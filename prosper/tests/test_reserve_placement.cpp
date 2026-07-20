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
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Window bounds mirror hle_kernel_mem.cpp (kGuestAutoMapBase/Limit on Linux; kGuestAutoVaMin/Max
// on Windows). The huge-reserve band must stay inside the platform's guest auto window.
static constexpr uint64_t kAutoMin = 0x2000000000ull;      // 128 GiB
#ifdef _WIN32
static constexpr uint64_t kAutoEndIncl = 0xfbffffffffull;  // inclusive (~1 TiB aperture ceiling)
#else
static constexpr uint64_t kAutoEndIncl = 0x40000000000ull - 1;   // limit exclusive -> inclusive
#endif

static constexpr uint64_t kArenaHint  = 0x1000000000ull;   // 64 GiB — DQ7's live hint
static constexpr uint64_t kArenaLen   = 0x8000000000ull;   // 512 GiB
static constexpr uint64_t kArenaAlign = 0x200000ull;       // 2 MiB — DQ7's live align
static constexpr uint64_t kHugeMin    = 0x2000000000ull;   // 128 GiB redirect threshold

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
    //    test placeholder geometry, not the redirect contract (#1084 review finding).
    uint64_t below = 0x1200000000ull;
    uint64_t rc = reserve((uint64_t)&below, kHugeMin - 0x10000, 0, 0x10000, 0, 0);
    CHECK(rc == 0 && below == 0x1200000000ull,
          "just-below-threshold reserve still honors its free hint");
    unmap(below, kHugeMin - 0x10000, 0, 0, 0, 0);

    // 1) The DQ7 arena reserve: huge + non-fixed must NOT land at its low hint, must land
    //    aligned inside the guest auto window.
    uint64_t base = kArenaHint;
    rc = reserve((uint64_t)&base, kArenaLen, 0, kArenaAlign, 0, 0);
    CHECK(rc == 0, "huge non-fixed reserve succeeds");
    CHECK(base != kArenaHint, "huge reserve does not land at the low 0x1000000000 hint");
    CHECK(base >= kAutoMin, "huge reserve lands at or above the auto window base");
    CHECK(base + kArenaLen - 1 <= kAutoEndIncl, "huge reserve ends inside the auto window");
    CHECK((base % kArenaAlign) == 0, "huge reserve honors the requested 2 MiB alignment");

    // 2) The metadata pool: the low hint the arena vacated must remain available to the guest's
    //    next small reserve with the same hint (search-start semantics, first free range).
    uint64_t pool = kArenaHint;
    rc = reserve((uint64_t)&pool, 0x4000000, 0, 0x4000, 0, 0);
    CHECK(rc == 0, "small same-hint reserve succeeds");
    CHECK(pool == kArenaHint, "small reserve takes the vacated 0x1000000000 hint");

    // 2b) A SECOND huge reserve with the same hint, while the small pool occupies it, must not
    //     false-succeed via the idempotent re-reserve check (hint-containment alone matched the
    //     64 MiB pool and returned the hint backed by mostly-unreserved VA) — it must be
    //     redirected like any huge reserve.
    uint64_t again = kArenaHint;
    rc = reserve((uint64_t)&again, kArenaLen, 0, kArenaAlign, 0, 0);
    CHECK(rc == 0 && again != kArenaHint && again >= kAutoMin,
          "second same-hint huge reserve is redirected, not idempotent-matched to the pool");
    unmap(again, kArenaLen, 0, 0, 0, 0);
    unmap(pool, 0x4000000, 0, 0, 0, 0);
    // Release the arena so the remaining cases probe hints on clean address space — their
    // assertions guard UNCHANGED semantics and must hold both before and after the fix.
    unmap(base, kArenaLen, 0, 0, 0, 0);

    // 3) Non-huge hinted reserves keep literal-when-free search semantics.
    uint64_t small = 0x1800000000ull;
    rc = reserve((uint64_t)&small, 0x10000, 0, 0x4000, 0, 0);
    CHECK(rc == 0 && small == 0x1800000000ull, "non-huge hinted reserve still honors a free hint");
    unmap(small, 0x10000, 0, 0, 0, 0);

    // 4) MAP_FIXED reserves are untouched by the redirect.
    uint64_t fixedBase = 0x1900000000ull;
    rc = reserve((uint64_t)&fixedBase, 0x10000, 0x10, 0x4000, 0, 0);
    CHECK(rc == 0 && fixedBase == 0x1900000000ull, "MAP_FIXED reserve keeps its exact hint");

    // 4b) A FIXED reserve whose span EXCEEDS the existing same-base reservation must fail, not
    //     false-succeed through the hint-only idempotency match (span containment required).
    uint64_t oversize = 0x1900000000ull;
    rc = reserve((uint64_t)&oversize, 0x20000, 0x10, 0x4000, 0, 0);
    CHECK(rc != 0, "FIXED re-reserve larger than the owning range fails instead of false-succeeding");
    unmap(fixedBase, 0x10000, 0, 0, 0, 0);

    // 5) Threshold boundary, at-threshold side: exactly 128 GiB redirects. (On Windows this also
    //    exercises free-placeholder recycling of the released arena span.)
    uint64_t atThr = 0x1200000000ull;
    rc = reserve((uint64_t)&atThr, kHugeMin, 0, 0x10000, 0, 0);
    CHECK(rc == 0 && atThr != 0x1200000000ull && atThr >= kAutoMin,
          "exactly-128GiB non-fixed reserve is redirected into the window");
    unmap(atThr, kHugeMin, 0, 0, 0, 0);

    printf("fails=%d\n", fails);
    return fails ? 1 : 0;
}
