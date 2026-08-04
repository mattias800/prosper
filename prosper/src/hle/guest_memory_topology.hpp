#pragma once

#include <cstddef>
#include <cstdint>

namespace prosper {

// Relationship between two guest virtual ranges after consulting the kernel-memory mapping table.
// `Unknown` is intentionally distinct from `Disjoint`: consumers that need an ordering proof must
// keep fail-closed behavior when either range is untracked, uncommitted, malformed, or crosses a
// mapping boundary.
enum class GuestMemoryTopologyRelation : uint8_t {
    Unknown,
    Disjoint,
    Overlap,
};

GuestMemoryTopologyRelation guest_memory_topology_relation(
    uint64_t first_address, uint64_t first_size,
    uint64_t second_address, uint64_t second_size);

// Copy host bytes into a fully committed direct-memory mapping through prosper's authoritative
// physical backing. This models a device write without weakening the guest VA's CPU protection.
// Private, untracked, malformed, and cross-mapping destinations fail closed.
bool guest_memory_gpu_write_supported(uint64_t destination, size_t bytes);
bool guest_memory_gpu_write(uint64_t destination, const void* source, size_t bytes);

} // namespace prosper
