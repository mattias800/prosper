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
    // Shape taken verbatim from the live call under PROSPER_AMPRLOG:
    //   Q07J7XpvhrU a0=0x0 a1=0x400000000 a2=0x280000000 a3=0x200000 a4=0x1 a5=<out>
    // i.e. (searchStart, searchEnd, len, ALIGNMENT, MEMORY TYPE, off_t* out) — the kernel
    // allocator's order. A smaller len keeps the test from claiming the whole pool.
    constexpr uint64_t kPoolLen = 0x1000000ull;   // 16 MiB
    uint64_t phys = kPoison;
    CHECK(give_dmem(0, 16ull << 30, kPoolLen, 0x200000, 1, (uint64_t)&phys) == 0,
          "GiveDirectMemory claims a physical pool");
    CHECK(phys != kPoison && phys != 0,
          "GiveDirectMemory WRITES the physical offset out-parameter");
    // The alignment argument is honoured, which is the arm that separates the two readings of
    // a3/a4: with them swapped, a3 = 1 is rejected as an alignment and the carve falls back to the
    // 16 KiB granule, so a 2 MiB-aligned offset stops being guaranteed. (a4 = 1 as an alignment
    // would be rejected too, so this cannot pass by accident either way round.)
    CHECK((phys & (0x200000ull - 1)) == 0,
          "the physical offset honours the 2 MiB alignment the guest asked for in a3");

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

    // The whole point: the guest can use the page. Fresh direct memory reads back zero on hardware.
    volatile uint64_t* mapped = (volatile uint64_t*)(uintptr_t)window_base;
    CHECK(mapped[0] == 0 && mapped[(kMapLen / 8) - 1] == 0,
          "the mapped range reads back zero across its whole length");
    mapped[0] = 0x0123456789abcdefull;
    mapped[(kMapLen / 8) - 1] = 0xfedcba9876543210ull;
    CHECK(mapped[0] == 0x0123456789abcdefull && mapped[(kMapLen / 8) - 1] == 0xfedcba9876543210ull,
          "the mapped range is writable and reads back what was written");

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
