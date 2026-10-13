#pragma once
#include "gpu/pm4/pending_memory_view.hpp"
#include <chrono>
#include <deque>
#include <memory>

namespace prosper::gpu {
struct PendingMemoryPayload {
    std::vector<uint32_t> words;
    std::vector<uint8_t> dma;
};
struct PendingMemoryWrite {
    Pm4Command cmd;
    std::shared_ptr<PendingMemoryPayload> payload = std::make_shared<PendingMemoryPayload>();
    uint64_t packet_addr = 0;
    std::chrono::steady_clock::time_point queued{};
    // Offsets in the ORIGINAL command and owned payload. Published bytes never rejoin this mask.
    std::vector<PendingMemoryRange> ranges;
    bool fragmented = false;
};
struct PendingMemorySourceOverlay {
    PendingMemoryWrite write;
    std::vector<PendingMemoryPatch> patches;
};
// Queue ownership and geometry only. The CP supplies synchronization, mapping leases and guarded
// effect application. New records are indexed/classified once while topology remains unchanged.
class PendingMemoryQueue {
public:
    bool empty() const { return writes_.empty(); }
    size_t size() const { return writes_.size(); }
    const PendingMemoryWrite& front() const { return writes_.front(); }
    auto begin() const { return writes_.begin(); }
    auto end() const { return writes_.end(); }
    void push_back(PendingMemoryWrite write);
    PendingMemoryWrite pop_front();
    std::vector<PendingMemoryWrite> renderer_writes(const GuestMappingLease& lease, bool batch);
    std::vector<PendingMemoryRange> private_ranges(const GuestMappingLease& lease,
                                                   const PendingMemoryGeometry& geometry);
    std::vector<PendingMemorySourceOverlay> source_overlays(const GuestMappingLease& lease,
                                                            const PendingMemoryGeometry& source);
    PendingMemoryGeometry geometry(const GuestMappingLease& lease, PendingMemorySpan span);
    bool topology_unresolved(const GuestMappingLease& lease);
    bool has_private_memory(const GuestMappingLease& lease);
    uint64_t records_examined() const { return records_examined_; }
    uint64_t geometry_queries() const { return geometry_queries_; }

private:
    void prepare_index(const GuestMappingLease& lease);
    void index_write(const GuestMappingLease& lease, const PendingMemoryWrite& write,
                     const PendingMemoryGeometry* whole = nullptr);
    std::deque<PendingMemoryWrite> writes_;
    PendingMemoryIndex index_;
    size_t indexed_ = 0, screened_ = 0;
    uint64_t generation_ = UINT64_MAX;
    uint64_t records_examined_ = 0, geometry_queries_ = 0;
    bool unresolved_ = false;
};
}   // namespace prosper::gpu
