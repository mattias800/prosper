#pragma once
#include "gpu/pm4/pm4_decode.hpp"
#include "host/memory/guest_memory_topology.hpp"
#include <cstdint>
#include <map>
#include <span>

namespace prosper::gpu {
struct PendingMemorySpan {
    uint64_t address = 0, bytes = 0;
};
struct PendingMemoryRange {
    size_t offset = 0, bytes = 0;
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
PendingMemoryGeometry pending_memory_subgeometry(const PendingMemoryGeometry& geometry,
                                                 PendingMemoryRange range);
std::vector<PendingMemoryRange>
pending_memory_complement(size_t bytes, std::span<const PendingMemoryRange> private_ranges);
// Union of exact VA and direct-backing intervals. Projection returns byte offsets in the queried
// resource, rather than making its whole footprint private because one byte overlaps a label.
class PendingMemoryIndex {
public:
    void add(const PendingMemoryGeometry& geometry);
    bool empty() const;
    std::vector<PendingMemoryRange> project(const PendingMemoryGeometry& geometry) const;

private:
    std::map<uint64_t, uint64_t> virtual_, physical_;
    bool incomplete_ = false;
};
// Unknown bytes can be made known by a later fixed producer. Refuse the final source only when
// some unknown byte survives the complete FIFO overlay, rather than at its first timestamp.
class PendingMemoryOverlayState {
public:
    void apply(const Pm4Command& command, std::span<const uint8_t> captured_dma,
               PendingMemoryPatch patch, std::span<uint8_t> destination);
    bool resolved() const { return !invalid_ && unknown_.empty(); }

private:
    std::map<uint64_t, uint64_t> unknown_;
    bool invalid_ = false;
};
// Apply only captured/fixed bytes, in command order. A timestamp or an uncaptured copy cannot
// be guessed from stale guest memory. GDS offsets occupy a separate address domain.
PendingMemoryOverlay overlay_pending_memory(const Pm4Command& command,
                                            std::span<const uint8_t> captured_dma, uint64_t address,
                                            std::span<uint8_t> destination);
} // namespace prosper::gpu
