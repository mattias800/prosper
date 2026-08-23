// libSceAmpr AMM (asynchronous memory manager) — the seven NIDs Yakuza Kiwami (PPSA31334) drives
// its entire game heap through (#2864).
//
// What this protects, and why an unimplemented AMM is not a harmless stub. Every one of these NIDs
// fell to the dispatcher's return-0 default, which for a VALUE-RETURNING contract is not "nothing
// happened": three of them have OUT-PARAMETERS the guest reads back. The one that matters is
// sceAmprAmmGetVirtualAddressRanges — the guest's AMM initialiser (eboot+0xdbf390) reaches it on a
// path that does not pre-zero its four-quadword struct, so a handler that returns 0 and writes
// nothing hands the allocator STACK RESIDUE as its virtual-address window. PPSA31334 then walked
// that window and died at 0.0 s: `movq $0x0,(%r12)` at eboot+0xdbc0f8, SIGSEGV at 0x1d0000.
//
// So the arms below are about the two directions that can go wrong:
//   * WRITE the out-parameters. `poison_survives` is the pre-fix behaviour, byte for byte.
//   * Never write OUTSIDE the window prosper reserved for AMM. A half-right AMM that maps where the
//     guest asks would MAP_FIXED over live guest memory, which is the #88 / #107 corruption class —
//     silent, and surfacing hours later somewhere unrelated.
//
// The window's own dimensions are asserted against the GUEST's gate, not against prosper's
// constant: the initialiser keeps min(requested, span - 4 GiB) and refuses to use the heap at all
// unless that is at least 0x8020 bytes (eboot+0xdbf571). Asserting the guest's inequality rather
// than kAmmWindowSize means resizing the window stays legal and shrinking it below what the guest
// can use does not.
#include "hle/dispatch/dispatch.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace prosper;

// The memory HLE's lazy-commit probe, as the SIGSEGV handler sees it. Declared here rather than
// pulled from a header because that is how every other caller declares it (exec_image_linux.cpp,
// hle_file.cpp).
extern "C" int prosper_reserved_range_state(uint64_t addr);

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

namespace {
constexpr uint64_t kPoison  = 0xdeadbeefdeadbeefull;   // what "wrote nothing" looks like
constexpr uint32_t kPoison32 = 0xdeadbeefu;
constexpr uint64_t kEinval  = 0x80020016ull;           // SCE_KERNEL_ERROR_EINVAL
constexpr uint64_t kEagain  = 0x80020010ull;           // the guest's retry sentinel — never return it
constexpr uint64_t kFourGiB = 0x100000000ull;
}

int main() {
    std::printf("== test_ampr_amm ==\n");
    register_builtin_hle();

    // sceAmprCommandBufferGetBufferBaseAddress is the generic accessor and is registered on every
    // platform; the six AMM entry points map guest pages and are POSIX-only for now (the Windows
    // arm's mapping primitives are a separate implementation). If this guard ever becomes
    // unnecessary, delete it — that is what "Windows AMM landed" looks like.
    HleFn set_buffer  = Hle::lookup("N-FSPA4S3nI");
    HleFn get_base    = Hle::lookup("RPCAhx-aabE");
    CHECK(set_buffer != nullptr, "sceAmprCommandBufferSetBuffer is registered");
    CHECK(get_base != nullptr,   "sceAmprCommandBufferGetBufferBaseAddress is registered");

    if (set_buffer && get_base) {
        // BUFFER flavor (a3 == a1): "this buffer already exists, just attach it". Deliberately not
        // the map flavor — this test is about the accessor, and the map flavor would ask the host
        // to place a mapping over the test's own storage.
        constexpr uint64_t kCb   = 0x7f0000010000ull;
        constexpr uint64_t kBuf  = 0x7f0000020000ull;
        constexpr uint64_t kSize = 0x40000ull;
        set_buffer(kCb, kBuf, kSize, kBuf, 0, 3);
        CHECK(get_base(kCb, 0, 0, 0, 0, 0) == kBuf,
              "GetBufferBaseAddress returns the buffer SetBuffer attached");
        CHECK(get_base(kCb + 0x1000, 0, 0, 0, 0, 0) == 0,
              "GetBufferBaseAddress answers 0 for a command buffer with no buffer attached");
    }

#if defined(__linux__) || defined(__APPLE__)
    HleFn get_ranges  = Hle::lookup("wkQR9+xTFKY");
    HleFn give_dmem   = Hle::lookup("Q07J7XpvhrU");
    HleFn amm_ctor    = Hle::lookup("EDq5bqCqYpA");
    HleFn amm_map     = Hle::lookup("JEVYGhDc97M");
    HleFn amm_submit  = Hle::lookup("OJf3vCckPAM");
    HleFn amm_wait    = Hle::lookup("HXymib4T8gc");
    CHECK(get_ranges && give_dmem && amm_ctor && amm_map && amm_submit && amm_wait,
          "all six libSceAmpr AMM NIDs are registered");
    if (!(get_ranges && give_dmem && amm_ctor && amm_map && amm_submit && amm_wait)) {
        std::printf("%s\n", fails ? "FAILED" : "PASSED");
        return fails ? 1 : 0;
    }

    // ---- the window ------------------------------------------------------------------------
    // Four out-parameters, pre-filled with residue exactly as the guest's stack slots are.
    uint64_t ranges[4] = { kPoison, kPoison, kPoison, kPoison };
    CHECK(get_ranges((uint64_t)&ranges[0], (uint64_t)&ranges[1],
                     (uint64_t)&ranges[2], (uint64_t)&ranges[3], 0, 0) == 0,
          "GetVirtualAddressRanges succeeds");
    CHECK(ranges[0] != kPoison && ranges[1] != kPoison,
          "GetVirtualAddressRanges WRITES the window start and end (the pre-fix defect: it did not)");
    CHECK(ranges[2] == 0 && ranges[3] == 0,
          "GetVirtualAddressRanges leaves no residue in the two out-parameters it cannot name");
    // Stop here rather than carry a poisoned window into the arms below. Without this the
    // no-write mutation does not FAIL the test, it SIGBUSes it on `mapped[0]` — which ctest scores
    // as a failure either way, but which reports a crash where the actual finding is "the handler
    // did not write its out-parameter". An instrument should say what it found.
    if (ranges[0] == kPoison || ranges[1] == kPoison) {
        std::printf("  [stop]  the window out-parameters were not written; the arms below would "
                    "dereference residue\n");
        std::printf("FAILED\n");
        return 1;
    }
    const uint64_t window_base = ranges[0], window_end = ranges[1];
    CHECK(window_base != 0 && window_end > window_base, "the window is a real, non-empty range");
    CHECK(window_end - window_base > kFourGiB + 0x8020ull,
          "the window clears the guest's own usability gate (span - 4 GiB >= 0x8020, eboot+0xdbf571)");

    uint64_t again[4] = { kPoison, kPoison, kPoison, kPoison };
    get_ranges((uint64_t)&again[0], (uint64_t)&again[1], (uint64_t)&again[2], (uint64_t)&again[3], 0, 0);
    CHECK(again[0] == window_base && again[1] == window_end,
          "the window is stable across calls (the guest caches r0 and classifies pointers against it)");

    // ---- the physical pool -----------------------------------------------------------------
    // Two properties of GiveDirectMemory are asserted below, and NEITHER can be observed against a
    // virgin direct-memory pool — which is how both of them were void in the first version of this
    // test, and is worth stating because the failure is invisible rather than wrong:
    //
    //   * the ALIGNMENT arm. The pool's first-fit gap starts at kDmemBase, and a fresh pool's
    //     first free offset is 16 KiB aligned by construction, so on any base that is ALSO 2 MiB
    //     aligned, honouring a 2 MiB alignment and silently falling back to the 16 KiB granule
    //     produce the SAME offset. Fix: hold a 16 KiB allocation so the next free offset is
    //     kDmemBase + 0x4000, which cannot be 2 MiB aligned, and where the two answers differ.
    //   * the ZEROING arm further down. prosper's pool is one process-wide memfd that retains bytes
    //     across release and reuse; in a fresh test process nothing has ever written it, so it reads
    //     back zero through sparseness whether or not the allocator punches the range. Fix: dirty
    //     the physical range first, then release it, then let the AMM pool land on it.
    //
    // Both were raised in review as same-source positive controls (the charter's instrument trap
    // 122): a control drawn from the same source as the null it validates tests the discriminator,
    // never the domain.
    HleFn alloc_dmem   = Hle::lookup("rTXw65xmLIA");   // sceKernelAllocateDirectMemory
    HleFn map_dmem     = Hle::lookup("L-Q3LEjIbgA");   // sceKernelMapDirectMemory
    // NIDs resolved from the PS5 3.20 firmware export database, not guessed — the first draft of
    // this line carried an invented NID for munmap and would have silently skipped the dirtying
    // step, leaving the zero arm exactly as void as it was before.
    HleFn munmap_fn    = Hle::lookup("cQke9UuBQOk");   // sceKernelMunmap
    HleFn release_dmem = Hle::lookup("MBuItvba6z8");   // sceKernelReleaseDirectMemory
    CHECK(alloc_dmem && map_dmem && munmap_fn && release_dmem,
          "the direct-memory NIDs this test builds its preconditions from are registered");
    if (!(alloc_dmem && map_dmem && munmap_fn && release_dmem)) {
        std::printf("FAILED\n");
        return 1;
    }

    // (1) Hold a 16 KiB block forever, so the pool's first free gap is no longer 2 MiB aligned.
    uint64_t pin = 0;
    CHECK(alloc_dmem(0, 16ull << 30, 0x4000, 0x4000, 0, (uint64_t)&pin) == 0,
          "a 16 KiB direct-memory block can be pinned to unalign the pool's first free gap");
    // The property this precondition needs is about the GAP, not about `pin` itself: whatever the
    // pool base is, a 16 KiB block held at the base must leave the next free offset off a 2 MiB
    // boundary, or the alignment arm below cannot discriminate. Asserting `pin`'s own 2 MiB
    // alignment stated that only for a base that happened to be 2 MiB aligned, and broke when
    // kDmemBase moved to 0x4000 (#2934) while the property it was standing in for still held.
    CHECK(pin != 0 && ((pin + 0x4000ull) & (0x200000ull - 1)) != 0,
          "...and the next free offset after it is NOT 2 MiB aligned");

    // (2) Dirty the physical range the AMM pool is about to be carved from, then release it. This
    // is what makes the zero-read arm below a statement about the allocator rather than about an
    // untouched file.
    constexpr uint64_t kDirtyLen = 0x400000ull;       // 4 MiB, spanning the next 2 MiB boundary
    uint64_t dirty = 0;
    CHECK(alloc_dmem(0, 16ull << 30, kDirtyLen, 0x4000, 0, (uint64_t)&dirty) == 0,
          "a scratch direct-memory block can be allocated over the pool's next gap");
    uint64_t dirty_va = 0;
    const uint64_t map_rc = map_dmem((uint64_t)&dirty_va, kDirtyLen, /*prot=*/0x3, /*flags=*/0,
                                     dirty, 0x4000);
    CHECK(map_rc == 0 && dirty_va != 0, "the scratch block maps");
    if (map_rc == 0 && dirty_va) {
        std::memset((void*)(uintptr_t)dirty_va, 0xa5, (size_t)kDirtyLen);
        munmap_fn(dirty_va, kDirtyLen, 0, 0, 0, 0);
    }
    release_dmem(dirty, kDirtyLen, 0, 0, 0, 0);

    constexpr uint64_t kPoolLen = 0x1000000ull;   // 16 MiB
    // A call that cannot deliver its result must not report success — and must not consume the pool
    // on the way to saying so. Both halves matter: the return value alone would still let a handler
    // take the memory first and refuse afterwards, which is how the guest ends up owning a pool it
    // was never told the address of. The second half is checked by the arms that follow: if this
    // call had consumed 16 MiB and published it, the real GiveDirectMemory below would be refused
    // as a non-contiguous second pool and every arm after it would redden. Raised in review.
    CHECK(give_dmem(0, 16ull << 30, kPoolLen, 0x200000, 1, /*out=*/0) == kEinval,
          "GiveDirectMemory refuses a call with no out-parameter instead of reporting success");

    // Shape taken verbatim from the live call under PROSPER_AMPRLOG:
    //   Q07J7XpvhrU a0=0x0 a1=0x400000000 a2=0x280000000 a3=0x200000 a4=0x1 a5=<out>
    // i.e. (searchStart, searchEnd, len, ALIGNMENT, MEMORY TYPE, off_t* out) — the kernel
    // allocator's order. A smaller len keeps the test from claiming the whole pool.
    uint64_t phys = kPoison;
    CHECK(give_dmem(0, 16ull << 30, kPoolLen, 0x200000, 1, (uint64_t)&phys) == 0,
          "GiveDirectMemory claims a physical pool");
    CHECK(phys != kPoison && phys != 0,
          "GiveDirectMemory WRITES the physical offset out-parameter");
    // Now discriminating, thanks to the pin above: the first free offset is kDmemBase + 0x4000,
    // which is never 2 MiB aligned, so an honoured 2 MiB alignment rounds up past it and a 16 KiB
    // fallback does not. Stated in terms of the base rather than in absolute offsets, because the
    // absolute ones went stale the moment kDmemBase moved (#2934) and a comment that describes the
    // world it replaced is believed instead of checked. Reading a3/a4 the other way round rejects
    // a3 = 1 as an alignment and reddens exactly here.
    CHECK(phys > pin && (phys & (0x200000ull - 1)) == 0,
          "the physical offset honours the 2 MiB alignment the guest asked for in a3");
    CHECK(phys >= dirty && phys + 0x40000ull <= dirty + kDirtyLen,
          "...and the pool's first 256 KiB lands inside the range the scratch block dirtied, so "
          "the zero arm below can actually fail");

    // ---- record, submit, wait, and then actually use the memory -----------------------------
    constexpr uint64_t kCb = 0x7f0000030000ull;
    amm_ctor(kCb, 0, 0, 0, 0, 0);
    constexpr uint64_t kMapLen = 0x40000ull;      // 256 KiB
    CHECK(amm_map(kCb, window_base, kMapLen, 0xb, 0xc3, 0) == 0,
          "AmmCommandBufferMap accepts a target inside the window");

    // Two ADJACENT dwords, because that is what the guest passes: it hands p1 = &state+0xed34 and
    // p2 = &state+0xed30 and reads the completion id back with a 32-bit load. Passing a3 = 0 here
    // leaves the neighbour poisoned on purpose: a 64-bit store of the id through p2 would erase it,
    // and that is precisely the mistake this arm exists to catch.
    volatile uint32_t slots[2] = { kPoison32, kPoison32 };
    const uint64_t submit_rc = amm_submit(/*bufferBase=*/0x7f0000020000ull, /*usedBytes=*/0x20,
                                          /*flags=*/0, /*p1=*/0, (uint64_t)&slots[0], 0);
    CHECK(submit_rc != kEagain,
          "SubmitCommandBuffer2 never answers EAGAIN (the guest spins on it, sleeping a second a turn)");
    CHECK(slots[0] != kPoison32 && slots[0] != 0,
          "SubmitCommandBuffer2 WRITES a nonzero completion id through its last out-parameter");
    CHECK(slots[1] == kPoison32,
          "the completion-id store is 32 bits wide and does not reach the adjacent dword");
    CHECK(amm_wait(slots[0], 0, 0, 0, 0, 0) == 0,
          "WaitCommandBufferCompletion accepts the id the submit handed out");
    // The same rule for the submit's own out-parameter: no slot to write the completion id into
    // means the guest would wait on residue, so this refuses rather than answering SCE_OK. And it
    // must not answer EAGAIN either — that is the retry sentinel the guest spins on.
    const uint64_t no_slot_rc = amm_submit(0x7f0000020000ull, 0x20, 0, 0, /*outCompletionId=*/0, 0);
    CHECK(no_slot_rc == kEinval,
          "SubmitCommandBuffer2 refuses a call with no completion-id out-parameter");
    CHECK(no_slot_rc != kEagain, "...and does not answer with the guest's retry sentinel");
    uint64_t ranges_bad[2] = { kPoison, kPoison };
    CHECK(get_ranges(0, (uint64_t)&ranges_bad[1], 0, 0, 0, 0) == kEinval &&
              ranges_bad[1] == kPoison,
          "GetVirtualAddressRanges refuses a call with no out-parameter for the window base");

    // The whole point: the guest can use the page. Fresh direct memory reads back zero on hardware,
    // and this range was filled with 0xa5 and released above, so a pool that skips the punch shows
    // it here rather than reading zero from an untouched file.
    volatile uint64_t* mapped = (volatile uint64_t*)(uintptr_t)window_base;
    bool all_zero = true;
    for (uint64_t i = 0; i < kMapLen / 8; ++i)
        if (mapped[i] != 0) { all_zero = false; break; }
    CHECK(all_zero,
          "the mapped range reads back zero across its whole length, over physical memory that "
          "held 0xa5 before it was released");
    mapped[0] = 0x0123456789abcdefull;
    mapped[(kMapLen / 8) - 1] = 0xfedcba9876543210ull;
    CHECK(mapped[0] == 0x0123456789abcdefull && mapped[(kMapLen / 8) - 1] == 0xfedcba9876543210ull,
          "the mapped range is writable and reads back what was written");

    // ---- the window declines lazy commit ------------------------------------------------------
    // Without this, the window is not inert: exec_image_linux.cpp's SIGSEGV handler backs any touch
    // of a tracked-but-uncommitted range above 0x1000000000 with a 64 KiB anonymous page, which
    // would turn every REFUSED map below into a silent substitution the guest cannot detect (it
    // does not test Map's return value). State 4 is what declines it — 4 and not 3, because 3 is
    // the Windows arm's "committed sparse direct page", a sub-case of a different parent. Both
    // points raised in review.
    CHECK(prosper_reserved_range_state(window_base + kMapLen) == 4,
          "an unmapped page of the AMM window declines lazy commit, so a refused map faults at the "
          "guest's own address instead of being backed with anonymous memory");
    CHECK(prosper_reserved_range_state(window_end - 0x4000) == 4,
          "...at the far end of the window too");
    CHECK(prosper_reserved_range_state(window_base) == 2,
          "...while a page this run actually mapped reads as committed");

    // ---- the corruption guard ---------------------------------------------------------------
    // A map whose target is OUTSIDE the window must be refused as a BAD ADDRESS, before any
    // physical page is carved and before the host is asked to place anything. The error code is
    // load-bearing: EINVAL says "not your address", and it is the answer that disappears if the
    // window check is deleted — the host mapping layer's own no-clobber refusal further down
    // reports ENOMEM instead, so the two are distinguishable and this arm can tell them apart.
    static uint64_t live[0x10000 / sizeof(uint64_t)];
    for (size_t i = 0; i < sizeof live / sizeof live[0]; ++i) live[i] = 0xa5a5a5a5a5a5a5a5ull;
    const uint64_t outside = ((uint64_t)(uintptr_t)live + 0x3fff) & ~0x3fffull;
    CHECK(amm_map(kCb, outside, 0x4000, 0xb, 0xc3, 0) == kEinval,
          "a map OUTSIDE the AMM window is refused as EINVAL, not attempted");
    bool intact = true;
    for (size_t i = 0; i < sizeof live / sizeof live[0]; ++i)
        if (live[i] != 0xa5a5a5a5a5a5a5a5ull) { intact = false; break; }
    CHECK(intact, "the refused target's contents are untouched");

    CHECK(amm_map(kCb, window_base + 1, kMapLen, 0xb, 0xc3, 0) == kEinval,
          "a misaligned map target is refused as EINVAL");
    CHECK(amm_map(kCb, window_base, kMapLen + 1, 0xb, 0xc3, 0) == kEinval,
          "a map length that is not a 16 KiB multiple is refused as EINVAL");
#else
    std::printf("  [skip] the six AMM entry points are POSIX-only for now (see the registration "
                "site in hle_kernel_mem.cpp)\n");
#endif

    std::printf("%s\n", fails ? "FAILED" : "PASSED");
    return fails ? 1 : 0;
}
