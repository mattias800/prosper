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

#ifdef _WIN32
static uint64_t find_free_guest_span(uint64_t len) {
    constexpr uint64_t kGranule = 0x10000;
    constexpr uint64_t kGuestVaBegin = 0x2000000000ull;
    constexpr uint64_t kGuestVaEnd = 0xfc00000000ull;
    uint64_t cursor = kGuestVaBegin;
    while (cursor <= kGuestVaEnd - len) {
        MEMORY_BASIC_INFORMATION mbi{};
        if (!VirtualQuery((void*)(uintptr_t)cursor, &mbi, sizeof(mbi))) break;
        const uint64_t region_base = (uint64_t)(uintptr_t)mbi.BaseAddress;
        const uint64_t region_end = region_base + (uint64_t)mbi.RegionSize;
        if (mbi.State == MEM_FREE) {
            uint64_t candidate = cursor > region_base ? cursor : region_base;
            candidate = (candidate + kGranule - 1) & ~(kGranule - 1);
            if (candidate <= kGuestVaEnd - len && candidate + len <= region_end)
                return candidate;
        }
        if (region_end <= cursor) break;
        cursor = (region_end + kGranule - 1) & ~(kGranule - 1);
    }
    return 0;
}
#endif

int main() {
    printf("== test_retrack_reservation ==\n");
    register_builtin_hle();
    auto reserve  = Hle::lookup(nid_hash("sceKernelReserveVirtualRange"));
    auto protect  = Hle::lookup(nid_hash("sceKernelMprotect"));
    auto mtypeprotect = Hle::lookup(nid_hash("sceKernelMtypeprotect"));
    auto batch    = Hle::lookup(nid_hash("sceKernelBatchMap"));
    auto query    = Hle::lookup(nid_hash("sceKernelVirtualQuery"));
    auto query_protection = Hle::lookup(nid_hash("sceKernelQueryMemoryProtection"));
    auto flexible = Hle::lookup(nid_hash("sceKernelMapNamedFlexibleMemory"));
    auto flexible_noname = Hle::lookup(nid_hash("sceKernelMapFlexibleMemory"));
    auto unmap    = Hle::lookup(nid_hash("sceKernelMunmap"));
    CHECK(nid_hash("sceKernelQueryMemoryProtection") == "WFcfL2lzido",
          "QueryMemoryProtection hashes to the PS5 3.20 import NID");
    CHECK(reserve && protect && mtypeprotect && batch && query && query_protection && flexible &&
              flexible_noname && unmap,
          "memory HLE functions registered");
    if (fails) return 1;

    constexpr uint64_t page = 0x10000;

    // #387 F6: without MAP_FIXED a nonzero address is a search hint. An occupied hint must
    // relocate forward without clobbering the existing mapping; MAP_FIXED retains exact-address
    // behavior and cannot be combined with a null address.
    uint64_t null_fixed = 0;
    CHECK((uint32_t)flexible((uint64_t)(uintptr_t)&null_fixed, page, 0x2,
                             0x10 /* SCE_KERNEL_MAP_FIXED */,
                             (uint64_t)(uintptr_t)"null-fixed", 0) == 0x80020016u &&
              null_fixed == 0,
          "MapFlexible rejects MAP_FIXED with a null address");

#ifdef _WIN32
    const uint64_t free_cover = find_free_guest_span(page * 2);
    uint64_t unaligned_fixed = free_cover ? free_cover + 0x4000 : 0;
    const bool unaligned_fixed_mapped = unaligned_fixed &&
        flexible((uint64_t)(uintptr_t)&unaligned_fixed, page, 0x2,
                 0x10 /* SCE_KERNEL_MAP_FIXED */,
                 (uint64_t)(uintptr_t)"unaligned-fixed", 0) == 0 &&
        unaligned_fixed == free_cover + 0x4000;
    CHECK(unaligned_fixed_mapped,
          "fixed free 16 KiB hint maps exactly instead of rounding down to 64 KiB");
    if (unaligned_fixed_mapped)
        CHECK(unmap(unaligned_fixed, page, 0, 0, 0, 0) == 0,
              "fixed 16 KiB flexible mapping unmaps cleanly");
#endif

    uint64_t first_flexible = 0;
    const bool first_flexible_mapped =
        flexible((uint64_t)(uintptr_t)&first_flexible, page, 0x2, 0,
                 (uint64_t)(uintptr_t)"flexible-hint-first", 0) == 0 &&
        first_flexible;
    CHECK(first_flexible_mapped, "map an automatic flexible page for hint collision");
    if (first_flexible_mapped) {
        volatile uint32_t* first_cell =
            (volatile uint32_t*)(uintptr_t)first_flexible;
        *first_cell = 0x387F6001u;

        uint64_t relocated = first_flexible;
        const bool relocated_mapped =
            flexible_noname((uint64_t)(uintptr_t)&relocated, page, 0x2, 0, 0, 0) == 0 &&
            relocated > first_flexible;
        CHECK(relocated_mapped,
              "non-fixed occupied flexible-memory hint relocates forward");
        CHECK(*first_cell == 0x387F6001u,
              "relocated flexible mapping preserves the occupied hint");

        uint64_t fixed_collision = first_flexible;
        CHECK(flexible_noname((uint64_t)(uintptr_t)&fixed_collision, page, 0x2,
                              0x10 /* SCE_KERNEL_MAP_FIXED */, 0, 0) != 0 &&
                  fixed_collision == first_flexible && *first_cell == 0x387F6001u,
              "fixed occupied flexible-memory hint fails without clobbering it");

        if (relocated_mapped)
            CHECK(unmap(relocated, page, 0, 0, 0, 0) == 0,
                  "relocated flexible page unmaps cleanly");
        CHECK(unmap(first_flexible, page, 0, 0, 0, 0) == 0,
              "original flexible page unmaps cleanly");
    }

#ifdef _WIN32
    // A loaded image or host allocation is not represented in the HLE mapping tracker. Windows
    // nevertheless lets VirtualAlloc(MEM_COMMIT) succeed on an already committed range, so the
    // exact-hint path must consult host VA state before it adopts and tracks an address.
    const uint64_t foreign_base = find_free_guest_span(page);
    void* foreign_page = foreign_base
        ? VirtualAlloc((void*)(uintptr_t)foreign_base, page,
                       MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
        : nullptr;
    CHECK(foreign_page != nullptr,
          "create an untracked committed page in the guest VA aperture");
    if (foreign_page) {
        auto* foreign_cell = (volatile uint32_t*)foreign_page;
        *foreign_cell = 0x387F60F1u;
        const uint64_t foreign_address = (uint64_t)(uintptr_t)foreign_page;

        uint64_t relocated = foreign_address;
        const bool foreign_relocated =
            flexible_noname((uint64_t)(uintptr_t)&relocated, page, 0x2, 0, 0, 0) == 0 &&
            relocated > foreign_address;
        CHECK(foreign_relocated,
              "non-fixed untracked committed hint relocates forward");
        CHECK(*foreign_cell == 0x387F60F1u,
              "relocation preserves the untracked committed page");

        uint64_t fixed_foreign = foreign_address;
        CHECK(flexible_noname((uint64_t)(uintptr_t)&fixed_foreign, page, 0x2,
                              0x10 /* SCE_KERNEL_MAP_FIXED */, 0, 0) != 0 &&
                  fixed_foreign == foreign_address && *foreign_cell == 0x387F60F1u,
              "fixed untracked committed hint fails without adopting it");

        if (foreign_relocated)
            CHECK(unmap(relocated, page, 0, 0, 0, 0) == 0,
                  "mapping relocated from an untracked page unmaps cleanly");
        CHECK(VirtualFree(foreign_page, 0, MEM_RELEASE) != 0,
              "untracked committed page releases cleanly");
    }
#endif

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
              failed_info[0x20] == 0,
          "failed mprotect leaves reservation tracking unchanged");

#ifdef _WIN32
    // #926: MEM_COMMIT says that a host allocation exists, not that it belongs to the guest. Keep
    // both pages in ONE host reservation, then use BatchMap's exact flexible operation to commit
    // and track only the first page. This makes a boundary-crossing ownership failure independent
    // of Windows' separate-allocation VirtualProtect restriction.
    void* host_reservation = VirtualAlloc(nullptr, page * 2, MEM_RESERVE, PAGE_NOACCESS);
    CHECK(host_reservation != nullptr, "reserve deterministic guest/host ownership test span");
    if (host_reservation) {
        const uint64_t host_base = (uint64_t)(uintptr_t)host_reservation;
        alignas(8) uint8_t map_entry[0x20]{};
        *(uint64_t*)(map_entry + 0x00) = host_base;
        *(uint64_t*)(map_entry + 0x10) = page;
        map_entry[0x18] = 0x2;
        *(int32_t*)(map_entry + 0x1c) = 3; // MAP_FLEXIBLE
        int32_t map_done = -1;
        const bool guest_mapped =
            batch((uint64_t)(uintptr_t)map_entry, 1,
                  (uint64_t)(uintptr_t)&map_done, 0, 0, 0) == 0 && map_done == 1;
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

        if (guest_mapped)
            CHECK(unmap(host_base, page, 0, 0, 0, 0) == 0,
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
    CHECK(*(int32_t*)(info + 0x18) == 0 && info[0x20] == 0,
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
              *(uint64_t*)(info + 0x10) == 0 && *(int32_t*)(info + 0x1c) == 0 &&
              *(int32_t*)(info + 0x18) == 0x2 &&
              info[0x20] == 0x11,
          "VirtualQuery reports flexible metadata without physical backing fields");
    uint64_t protection_start = UINT64_MAX;
    uint64_t protection_end = UINT64_MAX;
    uint32_t protection_value = UINT32_MAX;
    CHECK(query_protection(middle + page / 2, (uint64_t)(uintptr_t)&protection_start,
                           (uint64_t)(uintptr_t)&protection_end,
                           (uint64_t)(uintptr_t)&protection_value, 0, 0) == 0 &&
              protection_start == middle && protection_end == middle + page &&
              protection_value == 0x2,
          "QueryMemoryProtection returns the containing region and exact guest protection");
    CHECK(query_protection(middle + page / 2, 0, 0, 0, 0, 0) == 0,
          "QueryMemoryProtection permits optional output pointers");

    CHECK(protect(middle, page, 0x11 /* CPU_READ | GPU_READ */, 0, 0, 0) == 0,
          "mprotect accepts guest-only GPU protection bits");
    memset(info, 0, sizeof(info));
    CHECK(query(middle, 0, (uint64_t)(uintptr_t)info, sizeof(info), 0, 0) == 0 &&
              *(int32_t*)(info + 0x18) == 0x11 &&
              info[0x20] == 0x11,
          "VirtualQuery preserves guest-only GPU protection bits");
    protection_value = UINT32_MAX;
    CHECK(query_protection(middle + page / 2, 0, 0,
                           (uint64_t)(uintptr_t)&protection_value, 0, 0) == 0 &&
              protection_value == 0x11,
          "QueryMemoryProtection preserves guest-only GPU protection bits");

    CHECK(protect(base, len, 0x1 /* SCE_KERNEL_PROT_CPU_READ */, 0, 0, 0) == 0,
          "mprotect accepts a range spanning reserved and committed pages");
    memset(info, 0, sizeof(info));
    CHECK(query(base, 0, (uint64_t)(uintptr_t)info, sizeof(info), 0, 0) == 0 &&
              info[0x20] == 0,
          "range protection keeps the leading page uncommitted");
    memset(info, 0, sizeof(info));
    CHECK(query(middle, 0, (uint64_t)(uintptr_t)info, sizeof(info), 0, 0) == 0 &&
              *(int32_t*)(info + 0x18) == 0x1 &&
              info[0x20] == 0x11,
          "range protection keeps the middle page committed and reports read-only access");
    memset(info, 0, sizeof(info));
    CHECK(query(middle + page, 0, (uint64_t)(uintptr_t)info, sizeof(info), 0, 0) == 0 &&
              info[0x20] == 0,
          "range protection keeps the trailing page uncommitted");

    CHECK(unmap(base, len, 0, 0, 0, 0) == 0, "the test reservation unmaps cleanly");
    protection_start = UINT64_MAX;
    protection_end = UINT64_MAX;
    protection_value = UINT32_MAX;
    CHECK((uint32_t)query_protection(
              middle + page / 2, (uint64_t)(uintptr_t)&protection_start,
              (uint64_t)(uintptr_t)&protection_end,
              (uint64_t)(uintptr_t)&protection_value, 0, 0) == 0x8002000du &&
              protection_start == UINT64_MAX && protection_end == UINT64_MAX &&
              protection_value == UINT32_MAX,
          "QueryMemoryProtection rejects an untracked address without writing outputs");
    if (fails) { printf("== FAIL: %d check(s) ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
