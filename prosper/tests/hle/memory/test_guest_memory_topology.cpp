// A protection split changes tracking records, not the physical identity of a direct mapping.
// Drive the real HLE mapping table so an unrelated resource cannot become an unknown alias.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/memory/guest_memory_topology.hpp"
#include <cstdint>
#include <cstdio>

using namespace prosper;

int main() {
    register_builtin_hle();
    const auto allocate = Hle::lookup(nid_hash("sceKernelAllocateDirectMemory"));
    const auto map = Hle::lookup(nid_hash("sceKernelMapDirectMemory"));
    const auto protect = Hle::lookup(nid_hash("sceKernelMprotect"));
    const auto unmap = Hle::lookup(nid_hash("sceKernelMunmap"));
    const auto release = Hle::lookup(nid_hash("sceKernelReleaseDirectMemory"));
    if (!allocate || !map || !protect || !unmap || !release) return 2;

    constexpr uint64_t page = 0x10000;
    constexpr uint64_t span = 4 * page;
    uint64_t physical = 0, split = 0, alias = 0, unrelated = 0;
    if (allocate(0, 0x200000000ull, 2 * span, span, 0,
                 reinterpret_cast<uint64_t>(&physical)) != 0 || !physical)
        return 2;
    const auto mapped = [&](uint64_t& address, uint64_t size, uint64_t offset) {
        return map(reinterpret_cast<uint64_t>(&address), size, 2, 0, offset, page) == 0 &&
               address != 0;
    };
    if (!mapped(split, span, physical) ||
        !mapped(alias, page, physical + page) ||
        !mapped(unrelated, span, physical + span)) {
        if (split) unmap(split, span, 0, 0, 0, 0);
        if (alias) unmap(alias, page, 0, 0, 0, 0);
        if (unrelated) unmap(unrelated, span, 0, 0, 0, 0);
        release(physical, 2 * span, 0, 0, 0, 0);
        return 2;
    }
    int failures = 0;
    const auto expect = [&](bool condition, const char* message) {
        if (!condition) {
            std::fprintf(stderr, "guest topology: %s\n", message);
            ++failures;
        }
    };
    expect(protect(split + page, page, 2, 0, 0, 0) == 0,
           "protection split succeeded");
    const auto relation = [](uint64_t a, uint64_t a_bytes,
                             uint64_t b, uint64_t b_bytes) {
        return guest_memory_topology_relation(a, a_bytes, b, b_bytes);
    };
    expect(relation(split, span, unrelated, span) ==
               GuestMemoryTopologyRelation::Disjoint,
           "fully mapped split range remains physically disjoint");
    expect(protect(unrelated + page, page, 2, 0, 0, 0) == 0 &&
           relation(split, span, unrelated, span) ==
               GuestMemoryTopologyRelation::Disjoint,
           "two fragmented but independent mappings remain disjoint");
    expect(relation(split, span, alias, page) ==
               GuestMemoryTopologyRelation::Overlap &&
           relation(alias, page, split, span) == GuestMemoryTopologyRelation::Overlap,
           "an alias of the second split segment overlaps in both operand orders");
    expect(relation(split, span, 0x1000, page) ==
               GuestMemoryTopologyRelation::Unknown,
           "untracked range remains unknown");
    expect(relation(split, span, unrelated + span - page, 2 * page) ==
               GuestMemoryTopologyRelation::Unknown,
           "range extending beyond its committed mapping remains unknown");
    expect(relation(split, span, UINT64_MAX - 3, page) ==
               GuestMemoryTopologyRelation::Unknown,
           "overflow remains unknown");
    unmap(split, span, 0, 0, 0, 0);
    unmap(alias, page, 0, 0, 0, 0);
    unmap(unrelated, span, 0, 0, 0, 0);
    release(physical, 2 * span, 0, 0, 0, 0);
    return failures ? 1 : 0;
}
