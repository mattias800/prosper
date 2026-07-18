// test_retrack_reservation (#343) -- changing protection on an uncommitted guest reservation
// must not make the tracking layer report that the pages were committed. Guests use
// VirtualQuery to decide whether a later commit is necessary; a false committed result skips that
// commit and leaves the first access faulting.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#ifdef _WIN32
#include <windows.h>
#endif

using namespace prosper;

extern "C" int prosper_reserved_range_state(uint64_t addr);

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_retrack_reservation ==\n");
    register_builtin_hle();
    auto reserve  = Hle::lookup(nid_hash("sceKernelReserveVirtualRange"));
    auto protect  = Hle::lookup(nid_hash("sceKernelMprotect"));
    auto mtypeprotect = Hle::lookup(nid_hash("sceKernelMtypeprotect"));
    auto batch    = Hle::lookup(nid_hash("sceKernelBatchMap"));
    auto query    = Hle::lookup(nid_hash("sceKernelVirtualQuery"));
    auto flexible = Hle::lookup(nid_hash("sceKernelMapNamedFlexibleMemory"));
    auto unmap    = Hle::lookup(nid_hash("sceKernelMunmap"));
    CHECK(reserve && protect && mtypeprotect && batch && query && flexible && unmap,
          "memory HLE functions registered");
    if (fails) return 1;

    constexpr uint64_t page = 0x10000;
    constexpr uint64_t len = page * 3;
    uint64_t base = 0;
    CHECK(reserve((uint64_t)(uintptr_t)&base, len, 0, page, 0, 0) == 0 && base,
          "reserve three uncommitted pages");
    if (!base) return 1;

    // #387 F3: a host protection failure must reach the guest and must not retag the tracked
    // reservation. The old handler returned success and split/renamed the mapping even though no
    // host protection changed. Use a span the platform cannot protect as one reservation.
#ifdef _WIN32
    const uint64_t invalid_start = base + len - page / 2; // crosses the reserved allocation boundary
#else
    const uint64_t invalid_start = base + 1;              // POSIX mprotect requires page alignment
#endif
    CHECK((uint32_t)protect(invalid_start, page, 0x2, 0, 0, 0) == 0x80020016u,
          "mprotect maps an invalid host span to SCE_KERNEL_ERROR_EINVAL");
    uint8_t failed_info[0x48]{};
    CHECK(query(base + len - page / 4, 0, (uint64_t)(uintptr_t)failed_info,
                sizeof(failed_info), 0, 0) == 0 &&
              *(uint64_t*)(failed_info + 0x00) == base &&
              *(uint64_t*)(failed_info + 0x08) == base + len &&
              *(uint32_t*)(failed_info + 0x20) == 0,
          "failed mprotect leaves reservation tracking unchanged");

#ifdef _WIN32
    // #926: MEM_COMMIT says that a host allocation exists, not that it belongs to the guest. Put
    // one tracked guest page and one unrelated host-committed page in a single host reservation so
    // the boundary-crossing failure is deterministic regardless of the process address layout.
    void* host_reservation = VirtualAlloc(nullptr, page * 2, MEM_RESERVE, PAGE_NOACCESS);
    CHECK(host_reservation != nullptr, "reserve deterministic guest/host ownership test span");
    if (host_reservation) {
        const uint64_t host_base = (uint64_t)(uintptr_t)host_reservation;
        uint64_t guest_page = host_base;
        const bool guest_map_succeeded =
            flexible((uint64_t)(uintptr_t)&guest_page, page, 0x2, 0,
                     (uint64_t)(uintptr_t)"mprotect-ownership", 0) == 0 && guest_page;
        const bool guest_mapped = guest_map_succeeded && guest_page == host_base;
        CHECK(guest_mapped, "commit and track the first page as guest memory");

        void* foreign_page = nullptr;
        if (guest_mapped) {
            foreign_page = VirtualAlloc((void*)(uintptr_t)(host_base + page), page,
                                        MEM_COMMIT, PAGE_READWRITE);
        }
        CHECK(foreign_page == (void*)(uintptr_t)(host_base + page),
              "commit the adjacent page outside the guest tracker");

        if (guest_mapped && foreign_page) {
            volatile uint32_t* guest_cell =
                (volatile uint32_t*)(uintptr_t)(host_base + page / 4);
            volatile uint32_t* foreign_cell =
                (volatile uint32_t*)(uintptr_t)(host_base + page + page / 4);
            *guest_cell = 0x9260A11Cu;
            *foreign_cell = 0x9260F012u;

            const uint64_t crossing = host_base + page / 2;
            CHECK((uint32_t)protect(crossing, page, 0x1, 0, 0, 0) == 0x80020016u,
                  "mprotect rejects a committed tail outside guest ownership");
            CHECK((uint32_t)mtypeprotect(crossing, page, 0, 0x1, 0, 0) == 0x80020016u,
                  "mtypeprotect rejects the same unowned committed tail");

            alignas(8) uint8_t entry[0x20]{};
            *(uint64_t*)(entry + 0x00) = crossing;
            *(uint64_t*)(entry + 0x10) = page;
            entry[0x18] = 0x1;
            *(int32_t*)(entry + 0x1c) = 2; // PROTECT
            int32_t done = -1;
            CHECK((uint32_t)batch((uint64_t)(uintptr_t)entry, 1,
                                  (uint64_t)(uintptr_t)&done, 0, 0, 0) == 0x80020016u &&
                      done == 0,
                  "BatchMap protection failure is not counted or retracked");

            MEMORY_BASIC_INFORMATION guest_mbi{}, foreign_mbi{};
            VirtualQuery((const void*)(uintptr_t)host_base, &guest_mbi, sizeof(guest_mbi));
            VirtualQuery((const void*)(uintptr_t)(host_base + page), &foreign_mbi,
                         sizeof(foreign_mbi));
            CHECK(guest_mbi.State == MEM_COMMIT &&
                      (guest_mbi.Protect & 0xffu) == PAGE_READWRITE &&
                      *guest_cell == 0x9260A11Cu,
                  "failed protection calls leave the guest page unchanged");
            CHECK(foreign_mbi.State == MEM_COMMIT &&
                      (foreign_mbi.Protect & 0xffu) == PAGE_READWRITE &&
                      *foreign_cell == 0x9260F012u,
                  "failed protection calls leave unrelated host memory unchanged");
        }

        if (guest_map_succeeded)
            CHECK(unmap(guest_page, page, 0, 0, 0, 0) == 0,
                  "tracked ownership-test page unmaps cleanly");
        CHECK(VirtualFree(host_reservation, 0, MEM_RELEASE) != 0,
              "ownership-test host reservation releases cleanly");
    }
#endif

    const uint64_t middle = base + page;
    CHECK(protect(middle, page, 0x2 /* SCE_KERNEL_PROT_CPU_RW */, 0, 0, 0) == 0,
          "mprotect accepts the middle reservation page");

    uint8_t info[0x48]{};
    CHECK(query(middle, 0, (uint64_t)(uintptr_t)info, sizeof(info), 0, 0) == 0,
          "VirtualQuery accepts the protected reservation page");
    CHECK(*(uint64_t*)(info + 0x00) == middle &&
              *(uint64_t*)(info + 0x08) == middle + page,
          "protection tracking splits the reservation at the requested bounds");
    CHECK(*(int32_t*)(info + 0x18) == 0 && *(uint32_t*)(info + 0x20) == 0,
          "VirtualQuery still reports the protected reservation as uncommitted");
    CHECK(prosper_reserved_range_state(middle) == 1,
          "the lazy-commit probe still classifies the page as reserved");

    uint64_t committed = middle;
    CHECK(flexible((uint64_t)(uintptr_t)&committed, page, 0x2, 0,
                   (uint64_t)(uintptr_t)"retrack-commit", 0) == 0 && committed == middle,
          "MapFlexible can commit the protected reservation in place");
    if (committed == middle) {
        volatile uint32_t* cell = (volatile uint32_t*)(uintptr_t)committed;
        *cell = 0x343343u;
        CHECK(*cell == 0x343343u, "the committed page is writable");
    }
    memset(info, 0, sizeof(info));
    CHECK(query(middle, 0, (uint64_t)(uintptr_t)info, sizeof(info), 0, 0) == 0 &&
              *(uint32_t*)(info + 0x20) == 0x10,
          "VirtualQuery reports the page as committed only after MapFlexible");

    CHECK(protect(base, len, 0x1 /* SCE_KERNEL_PROT_CPU_READ */, 0, 0, 0) == 0,
          "mprotect accepts a range spanning reserved and committed pages");
    memset(info, 0, sizeof(info));
    CHECK(query(base, 0, (uint64_t)(uintptr_t)info, sizeof(info), 0, 0) == 0 &&
              *(uint32_t*)(info + 0x20) == 0,
          "range protection keeps the leading page uncommitted");
    memset(info, 0, sizeof(info));
    CHECK(query(middle, 0, (uint64_t)(uintptr_t)info, sizeof(info), 0, 0) == 0 &&
              *(uint32_t*)(info + 0x20) == 0x10,
          "range protection keeps the middle page committed");
    memset(info, 0, sizeof(info));
    CHECK(query(middle + page, 0, (uint64_t)(uintptr_t)info, sizeof(info), 0, 0) == 0 &&
              *(uint32_t*)(info + 0x20) == 0,
          "range protection keeps the trailing page uncommitted");

    CHECK(unmap(base, len, 0, 0, 0, 0) == 0, "the test reservation unmaps cleanly");
    if (fails) { printf("== FAIL: %d check(s) ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
