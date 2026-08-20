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

// Unit-test witness that the protected/backing-aware device-write path actually ran. Output bytes
// alone cannot prove that lever moved: a host memmove can coincidentally choose the correct direction
// for one pair of physically aliased but VA-disjoint views.
uint64_t guest_memory_gpu_write_successes_for_test();

// #2384: does an AMPR constructor's `a2` say the command buffer keeps a byte cursor? Exposed for
// test because the answer is a PREDICATE over guest-supplied bit patterns, and the only way to show
// the widening is strictly additive is to assert it over the shapes the working titles pass -- which
// a boot test cannot do without those titles' dumps. The implementation lives in hle_kernel_mem.cpp.
bool ampr_cb_tracks_offset_arg_for_test(uint64_t a2);

} // namespace prosper
