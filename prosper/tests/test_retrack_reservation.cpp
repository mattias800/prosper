// test_retrack_reservation (#343) -- changing protection on an uncommitted guest reservation
// must not make the tracking layer report that the pages were committed. Guests use
// VirtualQuery to decide whether a later commit is necessary; a false committed result skips that
// commit and leaves the first access faulting.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>

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
    auto query    = Hle::lookup(nid_hash("sceKernelVirtualQuery"));
    auto flexible = Hle::lookup(nid_hash("sceKernelMapNamedFlexibleMemory"));
    auto unmap    = Hle::lookup(nid_hash("sceKernelMunmap"));
    CHECK(reserve && protect && query && flexible && unmap, "memory HLE functions registered");
    if (fails) return 1;

    constexpr uint64_t page = 0x10000;
    constexpr uint64_t len = page * 3;
    uint64_t base = 0;
    CHECK(reserve((uint64_t)(uintptr_t)&base, len, 0, page, 0, 0) == 0 && base,
          "reserve three uncommitted pages");
    if (!base) return 1;

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
