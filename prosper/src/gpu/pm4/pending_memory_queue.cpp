// #2220: publish resource complements without replaying them when private completions retire.
// CONFIDENCE: HIGH on byte/FIFO identity under the mapping lease (actual ordered-executor tests).
// Live cross-title compatibility remains a separate validation gate.
#include "gpu/pm4/pending_memory_queue.hpp"
#include <algorithm>

namespace prosper::gpu {
void PendingMemoryQueue::push_back(PendingMemoryWrite write) {
    writes_.push_back(std::move(write));
}

PendingMemoryWrite PendingMemoryQueue::pop_front() {
    auto write = std::move(writes_.front());
    writes_.pop_front();
    if (screened_) --screened_;
    // Retirement removes blockers. A later submit must never inherit an old completion interval.
    generation_ = UINT64_MAX;
    return write;
}

PendingMemoryGeometry PendingMemoryQueue::geometry(const GuestMappingLease& lease,
                                                   PendingMemorySpan span) {
    ++geometry_queries_;
    return pending_memory_geometry(lease, span);
}

void PendingMemoryQueue::index_write(const GuestMappingLease& lease,
                                     const PendingMemoryWrite& write,
                                     const PendingMemoryGeometry* whole) {
    const auto span = pending_memory_span(write.cmd);
    if (!span.address || !span.bytes) return;
    if (pending_memory_completion(write.cmd)) {
        const auto shape = geometry(lease, span);
        index_.add(shape);
        unresolved_ |= shape.coverage == GuestMemoryMappingCoverage::Incomplete;
    } else if (write.fragmented) {
        for (const auto& range : write.ranges) {
            const auto shape = whole ? pending_memory_subgeometry(*whole, range)
                                     : geometry(lease, {span.address + range.offset, range.bytes});
            index_.add(shape);
            unresolved_ |= shape.coverage == GuestMemoryMappingCoverage::Incomplete;
        }
    }
}

void PendingMemoryQueue::prepare_index(const GuestMappingLease& lease) {
    const uint64_t generation = guest_memory_mapping_generation(lease);
    if (generation_ != generation) {
        index_ = {};
        indexed_ = 0;
        unresolved_ = false;
        generation_ = generation;
    }
    // Incremental insertion also bounds alternating completion/upload streams. On a topology
    // change, seed BOTH completions and retained resource fragments: their new aliases may differ.
    for (; indexed_ < writes_.size(); ++indexed_) {
        ++records_examined_;
        index_write(lease, writes_[indexed_]);
    }
}

std::vector<PendingMemoryRange>
PendingMemoryQueue::private_ranges(const GuestMappingLease& lease,
                                   const PendingMemoryGeometry& shape) {
    prepare_index(lease);
    return index_.project(shape);
}

bool PendingMemoryQueue::topology_unresolved(const GuestMappingLease& lease) {
    prepare_index(lease);
    return unresolved_;
}

bool PendingMemoryQueue::has_private_memory(const GuestMappingLease& lease) {
    prepare_index(lease);
    return !index_.empty();
}

std::vector<PendingMemoryWrite> PendingMemoryQueue::renderer_writes(const GuestMappingLease& lease,
                                                                    bool batch) {
    prepare_index(lease);
    std::vector<PendingMemoryWrite> ready;
    size_t retained = screened_, next = screened_;
    for (; next < writes_.size(); ++next) {
        ++records_examined_;
        auto write = std::move(writes_[next]);
        const auto span = pending_memory_span(write.cmd);
        const bool resource = write.cmd.kind == Pm4Command::Kind::WriteData ||
                              write.cmd.kind == Pm4Command::Kind::DmaData;
        if (!resource) {
            writes_[retained++] = std::move(write);
            continue;
        }
        const auto shape = geometry(lease, span);
        auto private_bytes = index_.project(shape);
        const auto public_bytes = pending_memory_complement(span.bytes, private_bytes);
        if (!public_bytes.empty() || !span.bytes) {
            ready.push_back(write);   // payload ownership is shared, never recopied for fragments
            ready.back().ranges = public_bytes;
            ready.back().fragmented = span.bytes != 0;
        }
        if (!private_bytes.empty()) {
            write.ranges = std::move(private_bytes);
            write.fragmented = true;
            index_write(lease, write, &shape);
            writes_[retained++] = std::move(write);
        }
        if (!batch && !ready.empty()) {
            ++next;
            break;
        }
    }
    screened_ = retained;
    // Compact only the newly appended suffix, never revisit the retained completion prefix.
    for (; next < writes_.size(); ++next) {
        if (retained != next) writes_[retained] = std::move(writes_[next]);
        ++retained;
    }
    writes_.resize(retained);
    indexed_ = writes_.size();
    return ready;
}

std::vector<PendingMemorySourceOverlay>
PendingMemoryQueue::source_overlays(const GuestMappingLease& lease,
                                    const PendingMemoryGeometry& source) {
    prepare_index(lease);
    std::vector<PendingMemorySourceOverlay> overlays;
    if (index_.project(source).empty()) return overlays;
    for (const auto& write : writes_) {
        ++records_examined_;
        const auto span = pending_memory_span(write.cmd);
        const auto collect = [&](PendingMemoryRange range) {
            auto patches = pending_memory_patches(
                geometry(lease, {span.address + range.offset, range.bytes}), source);
            for (auto& patch : patches) patch.source_offset += range.offset;
            if (!patches.empty()) overlays.push_back({write, std::move(patches)});
        };
        if (write.fragmented) {
            for (const auto& range : write.ranges) collect(range);
        } else if (span.address && span.bytes) {
            collect({0, static_cast<size_t>(span.bytes)});
        }
    }
    return overlays;
}
}   // namespace prosper::gpu
