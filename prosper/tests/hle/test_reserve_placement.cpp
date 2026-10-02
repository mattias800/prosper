// test_reserve_placement — guards the sceKernelReserveVirtualRange placement contract for HUGE
// non-fixed reservations (issues #312/#946, replacement for closed PR #982).
//
// DQ VII (PPSA17942, UE4) reserves its 512 GiB MallocBinned3 arena with hint=0x1000000000,
// flags=0, align=0x200000 (live-captured via PROSPER_MEMLOG), then a 64 MiB metadata pool with
// the same hint. A flag-less hint is a SEARCH START (BSD mmap / PS5 contract), and honoring it
// literally for the giant arena races prosper's own low guest-VA occupants: with the arena at
// 0x1000000000 the boot corrupts MallocBinned3 metadata (Linux #312) or wedges in a re-entrant
// __cxa_guard deadlock (Windows #946). The contract under test: a non-fixed reservation of
// >= 128 GiB is steered into the guest auto-map region instead of its low hint, WITHOUT widening
// the shared auto-map window (the widened-window approach re-admitted the PS5-libc-rejected
// 1-8 TiB gap and was rejected in #982 review). Smaller and MAP_FIXED reservations keep their
// existing hint semantics, and the vacated low hint stays available for the metadata pool.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include <cstdio>
#include <cstdint>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)
// Every reserve call logs its outcome so a platform-specific CI failure is attributable from the
// test output alone (address-space contents differ per host process; see the case-0 comment).
#define LOGV(tag, rc, val) printf("  %-10s rc=0x%llx addr=0x%llx\n", tag, \
                                  (unsigned long long)(rc), (unsigned long long)(val))

// Window bounds mirror hle_kernel_mem.cpp (kGuestAutoMapBase/Limit on Linux; kGuestAutoVaMin/Max
// on Windows). Huge hinted reserves retain their existing 128 GiB floor even when ordinary
// automatic maps prefer 256 GiB (#4064), so the low hint remains free for the metadata pool.
static constexpr uint64_t kAutoMin = 0x2000000000ull;      // 128 GiB huge-reserve floor
static constexpr uint64_t kAutoEndIncl = 0xfbffffffffull;  // inclusive (~1 TiB aperture ceiling)

// The below-threshold probe asserts "a free hint is honored", which requires the hint span to
// actually BE free in this host process — on Windows, high-entropy ASLR scatters small host
// allocations (image/stacks/heap) across the low terabyte, so any fixed ~128 GiB span can be
// occupied on a given run (empirically defeated 0x1200000000 AND would defeat any single
// alternative probabilistically — #1084 review). Verify the precondition with VirtualQuery over
// candidate hints and report the first blocking region; on POSIX the first candidate is free by
// construction in a fresh test process.
static bool span_is_free(uint64_t base, uint64_t len) {
#ifdef _WIN32
    uint64_t cur = base;
    while (cur < base + len) {
        MEMORY_BASIC_INFORMATION mbi{};
        if (!VirtualQuery(reinterpret_cast<void*>(static_cast<uintptr_t>(cur)),
                          &mbi, sizeof mbi)) return false;
        if (mbi.State != MEM_FREE) {
            printf("  [occupant] base=0x%llx size=0x%llx state=0x%lx (probe span 0x%llx)\n",
                   (unsigned long long)(uintptr_t)mbi.BaseAddress,
                   (unsigned long long)mbi.RegionSize, mbi.State, (unsigned long long)base);
            return false;
        }
        cur = (uint64_t)(uintptr_t)mbi.BaseAddress + mbi.RegionSize;
    }
#elif defined(__linux__)
    FILE* maps = std::fopen("/proc/self/maps", "r");
    if (!maps) return false;
    char line[512];
    bool free = true;
    while (std::fgets(line, sizeof(line), maps)) {
        unsigned long long lo = 0, hi = 0;
        if (std::sscanf(line, "%llx-%llx", &lo, &hi) != 2) { free = false; break; }
        if (lo < base + len && hi > base) {
            printf("  [occupant] base=0x%llx end=0x%llx (probe span 0x%llx)\n", lo, hi,
                   (unsigned long long)base);
            free = false;
            break;
        }
    }
    if (std::ferror(maps)) free = false;
    std::fclose(maps);
    return free;
#else
    (void)base; (void)len;
#endif
    return true;
}

static constexpr uint64_t kArenaHint  = 0x1000000000ull;   // 64 GiB — DQ7's live hint
static constexpr uint64_t kArenaAlign = 0x200000ull;       // 2 MiB — DQ7's live align
static constexpr uint64_t kHugeMin    = 0x2000000000ull;   // 128 GiB redirect threshold
// DQ7's real arena is 512 GiB, but the redirect CONTRACT only needs a reservation >= the 128 GiB
// threshold — and a 512 GiB reservation is fragile on the Windows CI runner: host high-entropy
// ASLR scatters small allocations across the 880 GiB auto window, so a single mid-window occupant
// can fragment it below 512 GiB contiguous and the reserve legitimately ENOMEMs (observed flaking
// case 1). A 160 GiB "huge" reservation clears the threshold and reliably fits a free span in the
// window regardless of occupancy, keeping the test deterministic while still exercising the huge
// path (the real 512 GiB size is not load-bearing for any assertion here).
static constexpr uint64_t kArenaLen   = 0x2800000000ull;   // 160 GiB (>= 128 GiB threshold)

#ifdef __APPLE__
// The macOS CI job runs x86_64 tests under Rosetta 2, whose VM tracking makes this test's
// operations pathologically slow: multi-hundred-GiB PROT_NONE reservations, and the POSIX
// reserve path's 64 KiB-stride MAP_FIXED_NOREPLACE probing — emulated on Darwin as a full
// mmap+munmap per probe (see the #983 cursor rationale in hle_kernel_mem.cpp) — can crawl for
// hours across host-occupied spans (observed: the CI Test step hung >30 min vs a ~5 min norm).
// The reserve-placement contract is platform-shared and fully exercised by the Linux and
// Windows jobs; skip here like test_raw_syscall does off-Linux.
int main() { printf("reserve_placement: skipped under Rosetta (giant-reservation VM-tracking "
                    "pathology); contract covered by the Linux and Windows jobs\n"); return 0; }
#else
int main() {
    printf("== test_reserve_placement ==\n");
    register_builtin_hle();
    auto reserve = Hle::lookup(nid_hash("sceKernelReserveVirtualRange"));
    auto unmap   = Hle::lookup(nid_hash("sceKernelMunmap"));
    CHECK(reserve && unmap, "reserve + munmap HLEs registered");
    if (!reserve || !unmap) { printf("fails=%d\n", fails); return 1; }

    // 0) Threshold boundary, just-below side FIRST, on pristine address space: later cases
    //    reserve/unmap huge in-window spans, and on Windows an unmapped span's VA stays
    //    OS-reserved as a free placeholder — probing a hint whose span crosses such VA would
    //    test placeholder geometry, not the redirect contract (#1084 review finding). Prefer the
    //    BELOW-window hint so this case's own free-placeholder residue never sits inside the
    //    window and cramps case 2b's whole-window fallback; verify the free-hint precondition
    //    (span_is_free above) instead of assuming any fixed span is free under Windows ASLR.
    // Single below-window candidate by review: in-window alternates carry latent geometry
    // hazards for later cases (a freed span at 0x6000000000 fragments the 512 GiB arena's
    // band+window fit deterministically; one at 0x4000000000 can leave case 2b only an
    // exact-fit gap). Both retained geometries — free below-window hint, and the relaxed
    // occupied-hint fallback — are validated by actual CI runs.
    const uint64_t belowLen = kHugeMin - 0x10000;
    const uint64_t belowCands[] = {0x1200000000ull};
    uint64_t belowHint = 0;
    for (uint64_t cand : belowCands)
        if (span_is_free(cand, belowLen)) { belowHint = cand; break; }
    uint64_t rc;
    if (belowHint) {
        uint64_t below = belowHint;
        rc = reserve((uint64_t)&below, belowLen, 0, 0x10000, 0, 0);
        LOGV("case0", rc, below);
        CHECK(rc == 0 && below == belowHint,
              "just-below-threshold reserve still honors its free hint");
        unmap(below, belowLen, 0, 0, 0, 0);
    } else {
        // Every candidate span is host-occupied (occupants printed above): the exact-hint
        // equality has no free-hint precondition to stand on. Still assert non-huge success.
        uint64_t below = belowCands[0];
        rc = reserve((uint64_t)&below, belowLen, 0, 0x10000, 0, 0);
        LOGV("case0-relaxed", rc, below);
        CHECK(rc == 0, "just-below-threshold reserve succeeds (hint occupancy: equality relaxed)");
        unmap(below, belowLen, 0, 0, 0, 0);
    }

    // #4032/#4064: before ordinary automatic mappings, keep huge's floor distinct from their base.
    // The huge redirect retains its separate floor rather than becoming a literal-at-hint reserve.
    // Windows case 1 below already covers that floor; an extra earlier giant reservation leaves
    // a free placeholder that fragments its later 512 GiB allocation even after guest unmap.
#ifndef _WIN32
    uint64_t pristine = kArenaHint;
    uint64_t pristine_rc = reserve((uint64_t)&pristine, kHugeMin, 0, kArenaAlign, 0, 0);
    LOGV("pristine", pristine_rc, pristine);
    CHECK(pristine_rc == 0 && pristine != kArenaHint && pristine >= kAutoMin &&
              pristine <= kAutoEndIncl - kHugeMin + 1,
          "a pristine huge hinted reserve keeps its legacy band above the low hint");
    if (pristine_rc == 0) unmap(pristine, kHugeMin, 0, 0, 0, 0);
#endif

    // 1) The DQ7 arena reserve: huge + non-fixed must NOT land at its low hint, must land
    //    aligned inside the guest auto window.
    uint64_t base = kArenaHint;
    rc = reserve((uint64_t)&base, kArenaLen, 0, kArenaAlign, 0, 0);
    LOGV("case1", rc, base);
    CHECK(rc == 0, "huge non-fixed reserve succeeds");
    CHECK(base != kArenaHint, "huge reserve does not land at the low 0x1000000000 hint");
    CHECK(base >= kAutoMin, "huge reserve lands at or above the auto window base");
    CHECK(base + kArenaLen - 1 <= kAutoEndIncl, "huge reserve ends inside the auto window");
    CHECK((base % kArenaAlign) == 0, "huge reserve honors the requested 2 MiB alignment");

    // 2) The metadata pool: the low hint the arena vacated must remain available to the guest's
    //    next small reserve with the same hint (search-start semantics, first free range).
    uint64_t pool = kArenaHint;
    rc = reserve((uint64_t)&pool, 0x4000000, 0, 0x4000, 0, 0);
    LOGV("case2", rc, pool);
    CHECK(rc == 0, "small same-hint reserve succeeds");
    CHECK(pool == kArenaHint, "small reserve takes the vacated 0x1000000000 hint");

    // 2b) A SECOND huge-class reserve with the same hint, while the small pool occupies it, must
    //     not false-succeed via the idempotent re-reserve check (a hint-only match returned the
    //     hint backed by the 64 MiB pool's mostly-unreserved VA) — it must be redirected like any
    //     huge reserve. Sized at the 128 GiB threshold, not a second 512 GiB arena: with the
    //     first arena still reserved, the ~880 GiB Windows window cannot hold another 512 GiB
    //     (#1084 review), and the idempotency bug under test only needs the huge class.
    uint64_t again = kArenaHint;
    rc = reserve((uint64_t)&again, kHugeMin, 0, kArenaAlign, 0, 0);
    LOGV("case2b", rc, again);
    CHECK(rc == 0 && again != kArenaHint && again >= kAutoMin,
          "second same-hint huge reserve is redirected, not idempotent-matched to the pool");
    unmap(again, kHugeMin, 0, 0, 0, 0);
    unmap(pool, 0x4000000, 0, 0, 0, 0);
    // Release the arena so the remaining cases probe hints on clean address space — their
    // assertions guard UNCHANGED semantics and must hold both before and after the fix.
    unmap(base, kArenaLen, 0, 0, 0, 0);

    // 3) Non-huge hinted reserves keep literal-when-free search semantics.
    uint64_t small = 0x1800000000ull;
    rc = reserve((uint64_t)&small, 0x10000, 0, 0x4000, 0, 0);
    LOGV("case3", rc, small);
    CHECK(rc == 0 && small == 0x1800000000ull, "non-huge hinted reserve still honors a free hint");
    unmap(small, 0x10000, 0, 0, 0, 0);

    // 4) MAP_FIXED reserves are untouched by the redirect.
    uint64_t fixedBase = 0x1900000000ull;
    rc = reserve((uint64_t)&fixedBase, 0x10000, 0x10, 0x4000, 0, 0);
    LOGV("case4", rc, fixedBase);
    CHECK(rc == 0 && fixedBase == 0x1900000000ull, "MAP_FIXED reserve keeps its exact hint");

    // 4b) A FIXED reserve whose span EXCEEDS the existing same-base reservation must fail, not
    //     false-succeed through the hint-only idempotency match (span containment required).
    uint64_t oversize = 0x1900000000ull;
    rc = reserve((uint64_t)&oversize, 0x20000, 0x10, 0x4000, 0, 0);
    LOGV("case4b", rc, oversize);
    CHECK(rc != 0, "FIXED re-reserve larger than the owning range fails instead of false-succeeding");
    unmap(fixedBase, 0x10000, 0, 0, 0, 0);

    // 5) Threshold boundary, at-threshold side: exactly 128 GiB redirects. (On Windows this also
    //    exercises free-placeholder recycling of the released arena span.)
    uint64_t atThr = 0x1200000000ull;
    rc = reserve((uint64_t)&atThr, kHugeMin, 0, 0x10000, 0, 0);
    LOGV("case5", rc, atThr);
    CHECK(rc == 0 && atThr != 0x1200000000ull && atThr >= kAutoMin,
          "exactly-128GiB non-fixed reserve is redirected into the window");
    unmap(atThr, kHugeMin, 0, 0, 0, 0);

#ifdef __linux__
    // Force the shared cursor to wrap BELOW the ordinary floor. The entire ordinary tail is a
    // sparse PROT_NONE reservation; only [128,256) can fit the next huge hinted 128 GiB request.
    // Checking the host-free precondition first makes a blocked fixture an explicit failure.
    constexpr uint64_t ordinaryFloor = 0x4000000000ull;
    constexpr uint64_t apertureEnd = kAutoEndIncl + 1;
    constexpr uint64_t tailLen = apertureEnd - ordinaryFloor;  // 752 GiB, no physical writes
    const bool tailFree = span_is_free(ordinaryFloor, tailLen);
    const bool hugeBandFree = span_is_free(kAutoMin, ordinaryFloor - kAutoMin);
    CHECK(tailFree && hugeBandFree, "host-free precondition for the ordinary-tail/huge-wrap discriminator");
    if (tailFree && hugeBandFree) {
        uint64_t tail = 0;
        const uint64_t tailRc = reserve((uint64_t)&tail, tailLen, 0, kArenaAlign, 0, 0);
        LOGV("tail", tailRc, tail);
        CHECK(tailRc == 0 && tail == ordinaryFloor && tail + tailLen == apertureEnd,
              "ordinary null-hint reservation wraps to its floor and fills only the bounded tail");
        if (tailRc == 0) {
            uint64_t lowerHuge = kArenaHint;
            const uint64_t lowerRc = reserve((uint64_t)&lowerHuge, kHugeMin, 0, kArenaAlign, 0, 0);
            LOGV("huge-wrap", lowerRc, lowerHuge);
            CHECK(lowerRc == 0 && lowerHuge == kAutoMin && lowerHuge + kHugeMin == ordinaryFloor,
                  "huge hinted reservation wraps to 128 GiB independently of the ordinary 256 GiB floor");
            if (lowerRc == 0)
                CHECK(unmap(lowerHuge, kHugeMin, 0, 0, 0, 0) == 0, "release the lower huge-wrap reservation");
            CHECK(unmap(tail, tailLen, 0, 0, 0, 0) == 0, "release the whole ordinary-tail reservation");
        }

        // The higher ordinary floor must still leave a real 512 GiB huge fit after ordinary use.
        constexpr uint64_t huge512Len = 0x8000000000ull;
        uint64_t huge512 = kArenaHint;
        const uint64_t huge512Rc = reserve((uint64_t)&huge512, huge512Len, 0, kArenaAlign, 0, 0);
        LOGV("huge512", huge512Rc, huge512);
        CHECK(huge512Rc == 0 && huge512 >= kAutoMin && huge512 <= apertureEnd - huge512Len,
              "a 512 GiB huge hinted reservation still fits after ordinary-tail use and release");
        uint64_t metadata = kArenaHint;
        const uint64_t metadataRc = reserve((uint64_t)&metadata, 0x4000000, 0, 0x4000, 0, 0);
        LOGV("metadata", metadataRc, metadata);
        CHECK(metadataRc == 0 && metadata == kArenaHint,
              "small hinted metadata retains its 64 GiB address beside the 512 GiB arena");
        if (metadataRc == 0) CHECK(unmap(metadata, 0x4000000, 0, 0, 0, 0) == 0, "release the metadata pool");
        if (huge512Rc == 0) CHECK(unmap(huge512, huge512Len, 0, 0, 0, 0) == 0, "release the 512 GiB reservation");
    }
#endif

    printf("fails=%d\n", fails);
    return fails ? 1 : 0;
}
#endif // __APPLE__
