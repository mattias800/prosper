// test_dmem — guards sceKernelAvailableDirectMemorySize (issue #99). It must report the LARGEST
// free aligned direct-memory block within the caller's [searchStart, searchEnd) window and write
// BOTH out-params — it was aliased to the total-size stub, which returned success while leaving
// physAddrOut/sizeOut uninitialized (a caller sizing an allocation from *sizeOut read stack garbage).
// Drives the handlers through the NID registry exactly as the guest does. Each ctest binary is its
// own process, so the process-global direct-memory pool starts empty here.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Pool bounds mirror hle_kernel_mem.cpp (kDmemBase / kDmemTotal).
static constexpr uint64_t kBase  = 0x10000000ull;
static constexpr uint64_t kTotal = 8ull * 1024 * 1024 * 1024;
static constexpr uint64_t kEnd   = kBase + kTotal;

int main() {
    printf("== test_dmem ==\n");
#ifndef __linux__
    // The direct-memory HLE (hle_kernel_mem.cpp) is #ifdef __linux__ — its functions aren't
    // registered on other platforms (e.g. the Windows/MinGW CI build), so there is nothing to
    // exercise here. Skip cleanly rather than fail on the absent registrations.
    printf("  [skip] direct-memory HLE is Linux-only on this build\n== PASS ==\n");
    return 0;
#else
    register_builtin_hle();

    auto avail   = Hle::lookup(nid_hash("sceKernelAvailableDirectMemorySize"));
    auto alloc   = Hle::lookup(nid_hash("sceKernelAllocateDirectMemory"));
    auto release = Hle::lookup(nid_hash("sceKernelReleaseDirectMemory"));
    CHECK(avail && alloc && release, "dmem fns registered");
    if (!(avail && alloc && release)) { printf("== FAIL ==\n"); return 1; }

    // Fresh pool: the largest free block is the whole pool, and BOTH out-params are written.
    uint64_t phys = 0xdead, size = 0xdead;
    uint64_t r = avail(0, kEnd, 0x4000, (uint64_t)(uintptr_t)&phys, (uint64_t)(uintptr_t)&size, 0);
    CHECK(r == 0, "available(fresh) returns success");
    CHECK(phys == kBase, "available(fresh) phys == pool base");
    CHECK(size == kTotal, "available(fresh) size == whole pool");

    // Null out-pointers -> EINVAL, and it must NOT be the old success-with-garbage.
    CHECK((uint32_t)avail(0, kEnd, 0x4000, 0, (uint64_t)(uintptr_t)&size, 0) == 0x80020016u,
          "available(null physAddrOut) -> EINVAL");
    CHECK((uint32_t)avail(0, kEnd, 0x4000, (uint64_t)(uintptr_t)&phys, 0, 0) == 0x80020016u,
          "available(null sizeOut) -> EINVAL");

    // Allocate 1 MiB (args: searchStart, searchEnd, len, align, type, physOut) at the pool base.
    uint64_t ap = 0;
    uint64_t ar = alloc(0, kEnd, 0x100000, 0x4000, 0, (uint64_t)(uintptr_t)&ap);
    CHECK(ar == 0 && ap == kBase, "allocate 1MiB -> phys at pool base");

    // Now the largest free block starts just past the allocation and is the pool minus 1 MiB.
    phys = size = 0;
    avail(0, kEnd, 0x4000, (uint64_t)(uintptr_t)&phys, (uint64_t)(uintptr_t)&size, 0);
    CHECK(phys == kBase + 0x100000, "available after alloc -> phys past the allocation");
    CHECK(size == kTotal - 0x100000, "available after alloc -> size shrank by the allocation");

    // A search window entirely within free space clamps the reported block to the window.
    uint64_t wlo = 0x20000000ull, wsz = 0x40000ull;
    phys = size = 0;
    r = avail(wlo, wlo + wsz, 0x4000, (uint64_t)(uintptr_t)&phys, (uint64_t)(uintptr_t)&size, 0);
    CHECK(r == 0 && phys == wlo && size == wsz, "available(window in free space) clamps to the window");

    // A search window entirely inside the allocated region has nothing free -> ENOMEM.
    r = avail(kBase, kBase + 0x80000, 0x4000, (uint64_t)(uintptr_t)&phys, (uint64_t)(uintptr_t)&size, 0);
    CHECK((uint32_t)r == 0x8002000Cu, "available(window fully allocated) -> ENOMEM");

    // Alignment is honored: a 1 MiB-aligned search over free space returns a 1 MiB-aligned phys.
    phys = 0;
    avail(0x20000123ull, kEnd, 0x100000, (uint64_t)(uintptr_t)&phys, (uint64_t)(uintptr_t)&size, 0);
    CHECK((phys & 0xfffff) == 0 && phys >= 0x20000123ull, "available honors the requested alignment");

    // Release the allocation -> the whole pool is available again.
    release(kBase, 0x100000, 0, 0, 0, 0);
    phys = size = 0;
    avail(0, kEnd, 0x4000, (uint64_t)(uintptr_t)&phys, (uint64_t)(uintptr_t)&size, 0);
    CHECK(phys == kBase && size == kTotal, "available after release -> whole pool free again");

    // --- Allocate HONORS the search window (issue #108): a request constrained to a window well
    //     above the pool base must land INSIDE that window, not at the base as before. ---
    {
        uint64_t wlo = 0x40000000ull;                 // 1 GiB into the pool
        uint64_t p = 0;
        uint64_t rr = alloc(wlo, wlo + 0x200000, 0x100000, 0x4000, 0, (uint64_t)(uintptr_t)&p);
        CHECK(rr == 0 && p >= wlo && p + 0x100000 <= wlo + 0x200000,
              "allocate honors [searchStart,searchEnd) -> phys inside the window");
        release(p, 0x100000, 0, 0, 0, 0);
    }
    // A window too small for the request -> ENOMEM (not a base-of-pool fallback).
    {
        uint64_t wlo = 0x50000000ull; uint64_t p = 0xdead;
        uint64_t rr = alloc(wlo, wlo + 0x1000 /*window < 0x100000 request*/, 0x100000, 0x4000, 0,
                            (uint64_t)(uintptr_t)&p);
        CHECK((uint32_t)rr == 0x8002000Cu, "allocate with too-small window -> ENOMEM (no out-of-window fallback)");
    }

    if (fails) { printf("== FAIL: %d check(s) ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
#endif
}
