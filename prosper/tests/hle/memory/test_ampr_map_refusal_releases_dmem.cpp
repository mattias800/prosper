// sceAmprCommandBufferSetBuffer's map flavor must not retire direct memory it could not map (#2908).
//
// The handler claims a physical range from the pool and then asks the host to place it at the
// guest's VA. That placement is REFUSED whenever the no-clobber discipline (#137 / #88 / #107)
// declines the target — an unaligned VA, or a VA that is already live guest memory — and the
// refusal is the correct answer: those calls carry an ALREADY-EXISTING buffer, and real hardware
// does not touch its memory. What was wrong is what happened to the pool offset afterwards. It was
// simply dropped: nothing referenced it, nothing could ever release it, and because the claim is
// made at 64 KiB alignment every carcass retired a full 64 KiB stride however small the request.
//
// The cost is not theoretical. On *The First Berserker: Khazan* (PPSA20447) a 6 s boot produces
// 4,294 refusals — 605 of 0x40, 3,686 of 0x4000, 3 of 0x10000 — which is 268 MiB of pool gone,
// against the 300 MiB scratch block that is the only headroom left after UE4's halving probe has
// claimed the rest. The engine heap then cannot commit, and the guest prints its own
// `PS5 Out of Memory` while its stats still report 13 GiB free: those stats describe UE's own
// reserved pool, and the shortage is one layer below them, in prosper's.
//
// The arms below are built so that a passing result cannot be a coincidence:
//   * The LEVER arm maps at a genuinely free VA, succeeds, and shows the pool shrink. Without it, a
//     "nothing leaked" result would also be produced by a handler that never touches the pool at
//     all, and the test would be void rather than negative.
//   * The REFUSAL arms then show the pool unchanged across a refusal — and separately show, via a
//     canary, that the mapping really was refused rather than quietly succeeding.
// Deliberately NOT asserted: how much the lever arm consumes. That is an allocator-internal
// quantity; the contract under test is only that a refused map costs nothing.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

namespace {
constexpr uint64_t kAmprSentinel = 0xffffffffull;   // the map flavor's a4 sentinel
constexpr uint64_t kCanary       = 0x5eedfacecafe01ull;

// Largest free aligned block in the direct-memory pool, through the guest-facing query. Using the
// real entry point rather than an internal accessor keeps the instrument on the same side of the
// ABI as the thing it is measuring.
uint64_t largest_free(HleFn avail) {
    uint64_t phys = 0, size = 0;
    const uint64_t rc = avail(0, 0, 0x10000, (uint64_t)(uintptr_t)&phys,
                              (uint64_t)(uintptr_t)&size, 0);
    return rc == 0 ? size : 0;
}
}  // namespace

int main() {
    std::printf("== test_ampr_map_refusal_releases_dmem ==\n");
    register_builtin_hle();

    HleFn set_buffer = Hle::lookup("N-FSPA4S3nI");            // sceAmprCommandBufferSetBuffer
    HleFn avail      = Hle::lookup(nid_hash("sceKernelAvailableDirectMemorySize"));
    CHECK(set_buffer != nullptr, "sceAmprCommandBufferSetBuffer is registered");
    CHECK(avail != nullptr,      "sceKernelAvailableDirectMemorySize is registered");
    if (!set_buffer || !avail) { std::printf("%s\n", fails ? "FAILED" : "PASSED"); return fails ? 1 : 0; }

    const uint64_t pool_at_start = largest_free(avail);
    CHECK(pool_at_start > (64ull << 20), "the pool query reports a usable free block to measure against");

    // ---- LEVER: a map flavor at a FREE VA consumes pool -----------------------------------------
    // Obtain a VA that is genuinely unclaimed by asking the host for one and giving it straight
    // back. Single-threaded test, so nothing races in behind the munmap.
    {
        const uint64_t kLen = 0x10000ull;
        void* probe = mmap(nullptr, kLen, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHECK(probe != MAP_FAILED, "reserved a scratch VA to release as a known-free target");
        if (probe != MAP_FAILED) {
            const uint64_t free_va = (uint64_t)(uintptr_t)probe;
            munmap(probe, kLen);
            const uint64_t before = largest_free(avail);
            // map flavor: a3 != a1, a4 = the -1 sentinel.
            set_buffer(0x7f0000ab0000ull, free_va, kLen, free_va + 8, kAmprSentinel, 0);
            const uint64_t after = largest_free(avail);
            CHECK(after < before,
                  "LEVER: a map flavor that the host accepts really does consume the pool "
                  "(so a no-change result below is a measurement, not a dead instrument)");
            std::printf("         lever consumed %llu bytes\n",
                        (unsigned long long)(before - after));
        }
    }

    // ---- REFUSAL 1: an occupied, page-aligned target ---------------------------------------------
    {
        const uint64_t kLen = 0x4000ull;                       // Khazan's descriptor-buffer size
        void* live = mmap(nullptr, kLen, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHECK(live != MAP_FAILED, "allocated a live page to stand in for guest memory");
        if (live != MAP_FAILED) {
            *(volatile uint64_t*)live = kCanary;
            const uint64_t va = (uint64_t)(uintptr_t)live;
            const uint64_t before = largest_free(avail);
            set_buffer(0x7f0000ab0000ull, va, kLen, va + 8, kAmprSentinel, 0);
            const uint64_t after = largest_free(avail);
            CHECK(*(volatile uint64_t*)live == kCanary,
                  "REFUSAL 1: the live page is untouched, i.e. the mapping really was refused");
            CHECK(after == before,
                  "REFUSAL 1: a refused map over live memory leaves the pool unchanged");
            munmap(live, kLen);
        }
    }

    // ---- REFUSAL 2: the unaligned 0x40 completion record, 256 times ------------------------------
    // Khazan's most numerous refusal shape. Repeating it is what turns a 64 KiB discrepancy into a
    // 16 MiB one: with the defect present this arm alone retires more pool than the title has left.
    {
        const uint64_t kLen   = 0x40ull;
        const int      kReps  = 256;
        void* live = mmap(nullptr, 0x10000, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHECK(live != MAP_FAILED, "allocated a live page for the unaligned-record arm");
        if (live != MAP_FAILED) {
            *(volatile uint64_t*)live = kCanary;
            const uint64_t before = largest_free(avail);
            for (int i = 0; i < kReps; i++) {
                const uint64_t va = (uint64_t)(uintptr_t)live + 0x40 + (uint64_t)i * 0x40;  // unaligned
                set_buffer(0x7f0000ab0000ull, va, kLen, va + 8, kAmprSentinel, 0);
            }
            const uint64_t after = largest_free(avail);
            CHECK(*(volatile uint64_t*)live == kCanary,
                  "REFUSAL 2: the live page is untouched across 256 unaligned map requests");
            CHECK(after == before,
                  "REFUSAL 2: 256 refused 0x40 maps leave the pool unchanged");
            if (after != before)
                std::printf("         leaked %llu bytes over %d refusals (%llu per call)\n",
                            (unsigned long long)(before - after), kReps,
                            (unsigned long long)((before - after) / (uint64_t)kReps));
            munmap(live, 0x10000);
        }
    }

    std::printf("%s\n", fails ? "FAILED" : "PASSED");
    return fails ? 1 : 0;
}
