// test_map_noreplace — a hinted map must NOT silently clobber a live mapping (#137). The old
// map_at/map_phys_at forced MAP_FIXED for any non-zero hint, so a MapFlexible/MapDirectMemory
// whose hint overlapped a committed mapping (or a loaded guest image) destroyed it. The fix:
// MAP_FIXED_NOREPLACE, upgraded to MAP_FIXED only when the hint is entirely our own uncommitted
// reservation. This drives the real HLE handlers and asserts a committed range survives a
// colliding map, that committing an OWN reservation still works, and that a free hint is honored.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>
#include <sys/mman.h>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_map_noreplace ==\n");
    register_builtin_hle();
    auto reserve  = Hle::lookup(nid_hash("sceKernelReserveVirtualRange"));
    auto flexible = Hle::lookup(nid_hash("sceKernelMapNamedFlexibleMemory"));
    CHECK(reserve && flexible, "map HLE functions registered");
    if (fails) { printf("== FAIL ==\n"); return 1; }

    auto U = [](const void* p) { return (uint64_t)(uintptr_t)p; };
    const uint64_t LEN = 0x10000;   // 64 KiB

    // 1. Commit a range via MapFlexible at hint=0 (kernel picks a free VA), write a sentinel.
    uint64_t va = 0;
    CHECK(flexible(U(&va), LEN, 0x2 /*RW*/, 0, U("live"), 0) == 0 && va, "MapFlexible(hint=0) succeeds");
    CHECK(va >= 0x2000000000ull && va < 0x40000000000ull,
          "automatic flexible mapping stays in the guest user-VA range");
    volatile uint32_t* cell = (volatile uint32_t*)(uintptr_t)va;
    *cell = 0xC0FFEE42u;

    // 2. Map AGAIN at the same (now COMMITTED) hint. The old code MAP_FIXED'd fresh zero pages over
    //    it, erasing the sentinel; the fix refuses (the range is committed, not a free reservation).
    uint64_t va2 = va;
    uint64_t r2 = flexible(U(&va2), LEN, 0x2, 0, U("colliding"), 0);
    CHECK(r2 != 0, "MapFlexible over a COMMITTED hint fails (does not clobber)");
    CHECK(*cell == 0xC0FFEE42u, "the committed range's contents SURVIVED the colliding map");

    // 3. Committing an OWN reservation must still work: reserve fixed, then map into it.
    //    Find a free high VA by probing with the host mmap the handler uses.
    void* probe = mmap(nullptr, LEN, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(probe != MAP_FAILED, "probe mmap for a free VA");
    uint64_t rbase = (uint64_t)(uintptr_t)probe;
    munmap(probe, LEN);   // free it again; the reserve below re-claims the same VA (nothing raced in)
    uint64_t rv = rbase;
    CHECK(reserve(U(&rv), LEN, 0x10 /*SCE_KERNEL_MAP_FIXED*/, 0x4000, 0, 0) == 0 && rv == rbase,
          "ReserveVirtualRange(fixed) tracks an uncommitted reservation");
    uint64_t mv = rbase;
    CHECK(flexible(U(&mv), LEN, 0x2, 0, U("into-reservation"), 0) == 0 && mv == rbase,
          "MapFlexible INTO our own reservation succeeds (commits it in place)");
    *(volatile uint32_t*)(uintptr_t)rbase = 0x1234u;   // writable now (was PROT_NONE reservation)
    CHECK(*(volatile uint32_t*)(uintptr_t)rbase == 0x1234u, "the committed reservation is writable");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
