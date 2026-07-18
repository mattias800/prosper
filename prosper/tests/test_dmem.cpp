// test_dmem — guards sceKernelAvailableDirectMemorySize (issue #99). It must report the LARGEST
// free aligned direct-memory block within the caller's [searchStart, searchEnd) window and write
// BOTH out-params — it was aliased to the total-size stub, which returned success while leaving
// physAddrOut/sizeOut uninitialized (a caller sizing an allocation from *sizeOut read stack garbage).
// Drives the handlers through the NID registry exactly as the guest does. Each ctest binary is its
// own process, so the process-global direct-memory pool starts empty here.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#ifdef _WIN32
#include "../src/host/exec_image.hpp"
#include "../src/host/guest_memory_map.hpp"
#include "../src/gpu/gpu_execute.hpp"
#include <windows.h>
extern "C" int prosper_try_commit_dmem(uint64_t addr, uint64_t len, int write);
#endif
#include <cstdio>
#include <cstdint>
#ifdef __linux__
#include <sys/mman.h>
#endif

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Pool bounds mirror hle_kernel_mem.cpp (kDmemBase / kDmemTotal).
static constexpr uint64_t kBase  = 0x10000000ull;
static constexpr uint64_t kTotal = 16ull * 1024 * 1024 * 1024;
static constexpr uint64_t kEnd   = kBase + kTotal;
using Hle7Fn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                            uint64_t, uint64_t, uint64_t);
#ifdef _WIN32
static constexpr uint64_t kGuestAutoVaMin = 0x2000000000ull;
static constexpr uint64_t kGuestAutoVaMax = 0xfbffffffffull;
#endif

int main() {
    printf("== test_dmem ==\n");
#ifdef _WIN32
    // Windows first-touch handling must commit one 16 KiB guest page, not a 64 KiB allocation-
    // granularity span. The latter crosses this reservation and fails with ERROR_INVALID_ADDRESS.
    register_builtin_hle();
    auto reserve = Hle::lookup(nid_hash("sceKernelReserveVirtualRange"));
    auto unmap   = Hle::lookup(nid_hash("sceKernelMunmap"));
    auto alloc   = Hle::lookup(nid_hash("sceKernelAllocateDirectMemory"));
    auto alloc_main = Hle::lookup(nid_hash("sceKernelAllocateMainDirectMemory"));
    auto map     = Hle::lookup(nid_hash("sceKernelMapDirectMemory"));
    auto map2    = reinterpret_cast<Hle7Fn>(
        Hle::lookup(nid_hash("sceKernelMapDirectMemory2")));
    auto protect = Hle::lookup(nid_hash("sceKernelMprotect"));
    auto release = Hle::lookup(nid_hash("sceKernelReleaseDirectMemory"));
    auto query   = Hle::lookup(nid_hash("sceKernelVirtualQuery"));
    CHECK(nid_hash("sceKernelMapDirectMemory2") == "BQQniolj9tQ",
          "sceKernelMapDirectMemory2 hashes to the PS5 3.20 import NID");
    CHECK(reserve && unmap && alloc && alloc_main && map && map2 && protect && release && query,
          "memory HLE functions registered");
    if (fails) return 1;

    constexpr uint64_t len = 0x4000;
    uint64_t va = 0x30000000000ull;  // Fixed, 64 KiB-aligned, and above the VEH's heap threshold.
    CHECK(reserve((uint64_t)(uintptr_t)&va, len, 0x10 /* MAP_FIXED */, len, 0, 0) == 0,
          "ReserveVirtualRange creates an exact 16 KiB reservation");
    if (fails) return 1;

    install_trap_handler();
    volatile uint32_t* cell = (volatile uint32_t*)(uintptr_t)va;
    *cell = 0x6310CAFEu;
    CHECK(*cell == 0x6310CAFEu, "first touch commits one guest page and preserves the write");
    CHECK(unmap(va, len, 0, 0, 0, 0) == 0, "reserved page unmaps cleanly");

    // Direct memory is one physical pool, not private memory per virtual mapping. Dead Cells
    // releases and reuses small physical ranges aggressively; independent Windows allocations
    // let its allocator observe different metadata through two aliases and corrupt adjacent VA.
    constexpr uint64_t dlen = 0x10000;
    CHECK((uint32_t)alloc(0, kEnd, dlen, dlen, 0, 0) == 0x80020016u,
          "AllocateDirectMemory(null physAddrOut) -> EINVAL");
    CHECK((uint32_t)alloc_main(dlen, dlen, 0, 0, 0, 0) == 0x80020016u,
          "AllocateMainDirectMemory(null physAddrOut) -> EINVAL");
    uint64_t phys = 0, va1 = 0, va2 = 0, va_map2 = 0;
    CHECK(alloc(0, kEnd, dlen, dlen, 0, (uint64_t)(uintptr_t)&phys) == 0 && phys == kBase,
          "allocate one 64 KiB direct-memory page");
    CHECK(map((uint64_t)(uintptr_t)&va1, dlen, 0x2, 0, phys, dlen) == 0 && va1,
          "map first direct-memory view");
#ifdef _WIN32
    CHECK(va1 >= kGuestAutoVaMin && va1 <= kGuestAutoVaMax - dlen + 1,
          "automatic direct view lands in PS5 libc's valid low VA aperture");
#endif
    CHECK(map((uint64_t)(uintptr_t)&va2, dlen, 0x2, 0, phys, dlen) == 0 && va2 && va2 != va1,
          "map second direct-memory view");
    CHECK(map2((uint64_t)(uintptr_t)&va_map2, dlen, 0 /* type */, 0x2 /* RW */, 0,
               phys, dlen) == 0 && va_map2 && va_map2 != va1 && va_map2 != va2,
          "MapDirectMemory2 consumes shifted prot/flags/phys/alignment arguments");
    if (va1 && va2) {
        *(volatile uint64_t*)(uintptr_t)(va1 + 0x1230) = 0x6310CAFEDEADC0DEull;
        CHECK(*(volatile uint64_t*)(uintptr_t)(va2 + 0x1230) == 0x6310CAFEDEADC0DEull,
              "two virtual mappings of one physical offset alias the same bytes");
        CHECK(va_map2 && *(volatile uint64_t*)(uintptr_t)(va_map2 + 0x1230) ==
                         0x6310CAFEDEADC0DEull,
              "MapDirectMemory2 view aliases the requested physical range");
    }

    uint64_t hinted = va1;
    CHECK(map((uint64_t)(uintptr_t)&hinted, dlen, 0x2, 0, phys, dlen) == 0 &&
              hinted && hinted != va1,
          "non-fixed occupied hint relocates to a shared direct-memory view");
    if (hinted) {
        CHECK(*(volatile uint64_t*)(uintptr_t)(hinted + 0x1230) == 0x6310CAFEDEADC0DEull,
              "relocated hinted mapping retains physical aliasing");
    }

    CHECK(release(phys, dlen, 0, 0, 0, 0) == 0, "release direct-memory page");
    uint64_t reused_phys = 0, va3 = 0;
    CHECK(alloc(0, kEnd, dlen, dlen, 0, (uint64_t)(uintptr_t)&reused_phys) == 0 && reused_phys == phys,
          "released physical range is reused");
    CHECK(map((uint64_t)(uintptr_t)&va3, dlen, 0x2, 0, reused_phys, dlen) == 0 && va3,
          "map reused direct-memory page");
    if (va3) {
        CHECK(*(volatile uint64_t*)(uintptr_t)(va3 + 0x1230) == 0,
              "fresh direct-memory allocation is zeroed after physical reuse");
    }
    if (va1) CHECK(unmap(va1, dlen, 0, 0, 0, 0) == 0, "unmap first direct-memory view");
    if (va2) CHECK(unmap(va2, dlen, 0, 0, 0, 0) == 0, "unmap second direct-memory view");
    if (va3) CHECK(unmap(va3, dlen, 0, 0, 0, 0) == 0, "unmap reused direct-memory view");
    if (va_map2) CHECK(unmap(va_map2, dlen, 0, 0, 0, 0) == 0,
                       "unmap MapDirectMemory2 view");
    if (hinted) CHECK(unmap(hinted, dlen, 0, 0, 0, 0) == 0, "unmap relocated direct-memory view");
    if (reused_phys) release(reused_phys, dlen, 0, 0, 0, 0);

    // Large zero-hint alignments use a placeholder-backed SEC_RESERVE view. Dead Cells requests
    // one 3 GiB mapping at 2 MiB alignment; committing that whole range made private memory exceed
    // 4.5 GiB. Prove that placement is aligned, untouched pages remain sparse, host GPU reads
    // materialize exactly their range, and a second VA still aliases the same physical bytes.
    constexpr uint64_t sparse_len = 0x02000000;
    constexpr uint64_t sparse_align = 0x00200000;
    constexpr uint64_t sparse_window = kBase + 0x10000000;
    uint64_t sparse_phys = 0, sparse_va1 = 0, sparse_va2 = 0;
    CHECK(alloc(sparse_window, sparse_window + sparse_len, sparse_len, sparse_align, 0,
                (uint64_t)(uintptr_t)&sparse_phys) == 0 && sparse_phys == sparse_window,
          "allocate a virgin 32 MiB sparse direct-memory range");
    CHECK(map((uint64_t)(uintptr_t)&sparse_va1, sparse_len, 0x2, 0,
              sparse_phys, sparse_align) == 0 && sparse_va1 &&
              (sparse_va1 & (sparse_align - 1)) == 0,
          "large-alignment direct view uses a 2 MiB-aligned address");
    MEMORY_BASIC_INFORMATION near_before{}, far_before{};
    if (sparse_va1) {
        uint8_t guest_query[0x48]{};
        CHECK(query(sparse_va1 + 0x01000000, 0,
                    (uint64_t)(uintptr_t)guest_query, sizeof(guest_query), 0, 0) == 0 &&
                  *(uint32_t*)(guest_query + 0x20) == 0x10,
              "guest VirtualQuery reports the sparse direct view as committed");
        host::GuestReadableRange persistent_range{};
        CHECK(!host::guest_readable_mapping_containing(
                  sparse_va1 + 0x4000, sparse_va1 + 0x8000, persistent_range),
              "sparse view is excluded from cross-submit host-readability reuse");
        VirtualQuery((void*)(uintptr_t)(sparse_va1 + 0x4000), &near_before,
                     sizeof(near_before));
        VirtualQuery((void*)(uintptr_t)(sparse_va1 + 0x01000000), &far_before,
                     sizeof(far_before));
        CHECK(near_before.State == MEM_RESERVE && far_before.State == MEM_RESERVE,
              "untouched sparse direct pages carry no host commit");
        CHECK(gpu::guest_readable(sparse_va1 + 0x4000, 0x4000),
              "GPU guest-read guard materializes an untouched zero page");
        MEMORY_BASIC_INFORMATION near_after{}, far_after{};
        VirtualQuery((void*)(uintptr_t)(sparse_va1 + 0x4000), &near_after,
                     sizeof(near_after));
        VirtualQuery((void*)(uintptr_t)(sparse_va1 + 0x01000000), &far_after,
                     sizeof(far_after));
        CHECK(near_after.State == MEM_COMMIT && far_after.State == MEM_RESERVE,
              "host read commits only the requested 16 KiB guest page");
        CHECK(*(volatile uint32_t*)(uintptr_t)(sparse_va1 + 0x4120) == 0,
              "virgin sparse direct-memory page reads as zero");

        const uint64_t protected_page = sparse_va1 + 0x01000000;
        CHECK(protect(protected_page, 0x4000, 0x1, 0, 0, 0) == 0,
              "mprotect accepts an untouched sparse page");
        MEMORY_BASIC_INFORMATION protected_before{};
        VirtualQuery((void*)(uintptr_t)protected_page, &protected_before,
                     sizeof(protected_before));
        CHECK(protected_before.State == MEM_RESERVE,
              "mprotect does not materialize an untouched sparse page");
        CHECK(!prosper_try_commit_dmem(protected_page, 0x4000, 1),
              "read-only sparse mapping rejects host write materialization");
        CHECK(prosper_try_commit_dmem(protected_page, 0x4000, 0),
              "read-only sparse mapping permits host read materialization");
        MEMORY_BASIC_INFORMATION protected_after{};
        VirtualQuery((void*)(uintptr_t)protected_page, &protected_after,
                     sizeof(protected_after));
        CHECK(protected_after.State == MEM_COMMIT &&
                  (protected_after.Protect & 0xff) == PAGE_READONLY,
              "first touch applies the tracked read-only protection");
        CHECK(protect(protected_page, 0x4000, 0x2, 0, 0, 0) == 0,
              "mprotect restores read/write on a materialized sparse page");
        CHECK(prosper_try_commit_dmem(protected_page, 0x4000, 1),
              "mapping-generation invalidation permits a cached sparse-page write");
        MEMORY_BASIC_INFORMATION writable_after{};
        VirtualQuery((void*)(uintptr_t)protected_page, &writable_after,
                     sizeof(writable_after));
        CHECK(writable_after.State == MEM_COMMIT &&
                  (writable_after.Protect & 0xff) == PAGE_READWRITE,
              "mprotect updates an existing sparse host page");
    }
    CHECK(map((uint64_t)(uintptr_t)&sparse_va2, sparse_len, 0x2, 0,
              sparse_phys, sparse_align) == 0 && sparse_va2 && sparse_va2 != sparse_va1,
          "map a second sparse view of the same physical range");
    if (sparse_va1 && sparse_va2) {
        *(volatile uint64_t*)(uintptr_t)(sparse_va1 + 0x5120) = 0x5A17C0DE6310CAFEull;
        CHECK(*(volatile uint64_t*)(uintptr_t)(sparse_va2 + 0x5120) ==
                  0x5A17C0DE6310CAFEull,
              "sparse views retain physical alias coherence");
    }
    if (sparse_va1) CHECK(unmap(sparse_va1, sparse_len, 0, 0, 0, 0) == 0,
                          "unmap first sparse direct-memory view");
    if (sparse_va2) CHECK(unmap(sparse_va2, sparse_len, 0, 0, 0, 0) == 0,
                          "unmap second sparse direct-memory view");
    if (sparse_phys) release(sparse_phys, sparse_len, 0, 0, 0, 0);

    if (fails) { printf("== FAIL: %d check(s) ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
#elif !defined(__linux__)
    // The direct-memory HLE (hle_kernel_mem.cpp) is #ifdef __linux__ — its functions aren't
    // registered on this platform, so there is nothing to exercise here. Skip cleanly.
    printf("  [skip] direct-memory HLE is Linux-only on this build\n== PASS ==\n");
    return 0;
#else
    register_builtin_hle();

    auto avail   = Hle::lookup(nid_hash("sceKernelAvailableDirectMemorySize"));
    auto alloc   = Hle::lookup(nid_hash("sceKernelAllocateDirectMemory"));
    auto alloc_main = Hle::lookup(nid_hash("sceKernelAllocateMainDirectMemory"));
    auto map     = Hle::lookup(nid_hash("sceKernelMapDirectMemory"));
    auto map2    = reinterpret_cast<Hle7Fn>(
        Hle::lookup(nid_hash("sceKernelMapDirectMemory2")));
    auto release = Hle::lookup(nid_hash("sceKernelReleaseDirectMemory"));
    CHECK(nid_hash("sceKernelMapDirectMemory2") == "BQQniolj9tQ",
          "sceKernelMapDirectMemory2 hashes to the PS5 3.20 import NID");
    CHECK(avail && alloc && alloc_main && map && map2 && release, "dmem fns registered");
    if (!(avail && alloc && alloc_main && map && map2 && release)) { printf("== FAIL ==\n"); return 1; }

    // Both allocation APIs require their physical-address output. Reject before taking from
    // the pool so an unobservable allocation cannot leak capacity.
    CHECK((uint32_t)alloc(0, kEnd, 0x4000, 0x4000, 0, 0) == 0x80020016u,
          "AllocateDirectMemory(null physAddrOut) -> EINVAL");
    CHECK((uint32_t)alloc_main(0x4000, 0x4000, 0, 0, 0, 0) == 0x80020016u,
          "AllocateMainDirectMemory(null physAddrOut) -> EINVAL");

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

    // A zero-hint direct mapping must honor its explicit VA alignment. Linux mmap only promises
    // 4 KiB, which made HashLink reject a mapping requested at 64 KiB alignment during startup.
    {
        uint64_t p = 0, va = 0;
        uint64_t rr = alloc(0, kEnd, 0x10000, 0x10000, 0, (uint64_t)(uintptr_t)&p);
        CHECK(rr == 0, "allocate 64KiB-aligned direct page succeeds");
        rr = map((uint64_t)(uintptr_t)&va, 0x10000, 0x2 /*RW*/, 0, p, 0x10000);
        CHECK(rr == 0 && va && (va & 0xffff) == 0,
              "map direct memory honors requested 64KiB VA alignment");
        if (va) munmap((void*)(uintptr_t)va, 0x10000);
        if (p) release(p, 0x10000, 0, 0, 0, 0);
    }

    // MapDirectMemory2 inserts `type` before prot/flags/phys/alignment. Type zero deliberately
    // differs from RW protection so an unshifted six-argument alias cannot pass this write/alias
    // check, and the seventh alignment argument is exercised by the real fixed-arity prototype.
    {
        uint64_t p = 0, va = 0, alias = 0;
        uint64_t rr = alloc(0, kEnd, 0x10000, 0x10000, 0,
                            (uint64_t)(uintptr_t)&p);
        CHECK(rr == 0, "allocate direct page for MapDirectMemory2");
        rr = map2((uint64_t)(uintptr_t)&va, 0x10000, 0 /* type */, 0x2 /* RW */, 0,
                  p, 0x10000);
        CHECK(rr == 0 && va && (va & 0xffff) == 0,
              "MapDirectMemory2 consumes shifted arguments and seventh-argument alignment");
        rr = map((uint64_t)(uintptr_t)&alias, 0x10000, 0x2, 0, p, 0x10000);
        CHECK(rr == 0 && alias, "map ordinary alias for MapDirectMemory2 physical range");
        if (va && alias) {
            *(volatile uint64_t*)(uintptr_t)(va + 0x120) = 0xB002D1EC7A11A5ull;
            CHECK(*(volatile uint64_t*)(uintptr_t)(alias + 0x120) == 0xB002D1EC7A11A5ull,
                  "MapDirectMemory2 maps the requested physical offset as writable shared memory");
        }
        if (va) munmap((void*)(uintptr_t)va, 0x10000);
        if (alias) munmap((void*)(uintptr_t)alias, 0x10000);
        if (p) release(p, 0x10000, 0, 0, 0, 0);
    }

    // A non-fixed direct-memory address is a search hint. Astro Bot supplies the eboot base as
    // its preferred address; treating every non-zero input as fixed returned ENOMEM instead of
    // relocating the mapping. A genuinely fixed collision must still fail without clobbering it.
    {
        constexpr uint64_t len = 0x10000;
        uint64_t p = 0;
        void* live = mmap(nullptr, len, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHECK(live != MAP_FAILED, "create a live mapping for direct-memory hint collision");
        if (live != MAP_FAILED) {
            *(volatile uint32_t*)live = 0xA57B0782u;
            uint64_t hinted = (uint64_t)(uintptr_t)live;
            uint64_t rr = alloc(0, kEnd, len, len, 0, (uint64_t)(uintptr_t)&p);
            CHECK(rr == 0 && p, "allocate direct memory for occupied-hint test");
            if (rr == 0 && p) {
                rr = map((uint64_t)(uintptr_t)&hinted, len, 0x2 /*RW*/, 0,
                         p, len);
                CHECK(rr == 0 && hinted && hinted != (uint64_t)(uintptr_t)live,
                      "non-fixed occupied direct-memory hint relocates");
                CHECK(hinted >= 0x2000000000ull && hinted < 0x40000000000ull,
                      "relocated direct mapping stays in the guest user-VA range");
                CHECK(*(volatile uint32_t*)live == 0xA57B0782u,
                      "relocated direct mapping does not clobber the live hint");
                if (hinted && hinted != (uint64_t)(uintptr_t)live)
                    munmap((void*)(uintptr_t)hinted, len);

                uint64_t fixed_hint = (uint64_t)(uintptr_t)live;
                rr = map((uint64_t)(uintptr_t)&fixed_hint, len, 0x2 /*RW*/,
                         0x10 /*SCE_KERNEL_MAP_FIXED*/, p, len);
                CHECK(rr != 0, "fixed occupied direct-memory hint fails");
                CHECK(*(volatile uint32_t*)live == 0xA57B0782u,
                      "failed fixed direct mapping preserves the live range");
                release(p, len, 0, 0, 0, 0);
            }
            munmap(live, len);
        }
    }

    if (fails) { printf("== FAIL: %d check(s) ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
#endif
}
