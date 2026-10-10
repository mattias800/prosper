#pragma once
#include "gpu/pm4/pm4_decode.hpp"
#include "host/memory/guest_memory_topology.hpp"
#include <cstdint>
#include <span>

namespace prosper::gpu {
struct PendingMemorySpan {
    uint64_t address = 0, bytes = 0;
};
PendingMemorySpan pending_memory_span(const Pm4Command& command);
bool pending_memory_overlaps(PendingMemorySpan first, PendingMemorySpan second);
bool pending_memory_completion(const Pm4Command& command);
enum class PendingMemoryOverlay { Disjoint, Applied, Unresolved };
struct PendingMemoryPatch {
    size_t source_offset = 0, target_offset = 0, bytes = 0;
};
struct PendingMemoryGeometry {
    PendingMemorySpan virtual_span;
    std::vector<GuestMemoryMappingSlice> mappings;
    GuestMemoryMappingCoverage coverage = GuestMemoryMappingCoverage::Untracked;
};
PendingMemoryGeometry pending_memory_geometry(const GuestMappingLease& lease,
                                              PendingMemorySpan span);
std::vector<PendingMemoryPatch> pending_memory_patches(const PendingMemoryGeometry& written,
                                                       const PendingMemoryGeometry& read);
PendingMemoryOverlay overlay_pending_memory_patch(const Pm4Command& command,
                                                  std::span<const uint8_t> captured_dma,
                                                  PendingMemoryPatch patch,
                                                  std::span<uint8_t> destination);
// Start with every completion span, then retain the complete footprint of each blocked resource
// write in FIFO order. Later partial overlaps cannot pass an earlier private write.
std::vector<bool> pending_renderer_selection(std::span<const Pm4Command> commands,
                                             std::span<const PendingMemoryGeometry> geometry);
// Apply only captured/fixed bytes, in command order. A timestamp or an uncaptured copy cannot
// be guessed from stale guest memory. GDS offsets occupy a separate address domain.
PendingMemoryOverlay overlay_pending_memory(const Pm4Command& command,
                                            std::span<const uint8_t> captured_dma, uint64_t address,
                                            std::span<uint8_t> destination);
} // namespace prosper::gpu
