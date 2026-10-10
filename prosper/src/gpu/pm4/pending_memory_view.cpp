#include "gpu/pm4/pending_memory_view.hpp"
#include "gpu/pm4/command_processor.hpp"
#include <algorithm>
#include <cstring>
#include <iterator>
#include <map>

namespace prosper::gpu {
// CONFIDENCE: HIGH on byte identity and FIFO closure: #2220's actual-submit regressions exercise
// distinct direct-memory VAs, protection splits, partial copies, and overlapping resource suffixes.
PendingMemorySpan pending_memory_span(const Pm4Command& command) {
    using K = Pm4Command::Kind;
    switch (command.kind) {
        case K::ReleaseMem:
            if (command.rel_data_sel == 3 ||
                (command.rel_value_valid &&
                 (command.rel_data_sel == 1 || command.rel_data_sel == 2)))
                return {command.rel_addr, command.rel_data_sel == 1 ? 4u : 8u};
            break;
        case K::EventWrite:
            if (command.event_addr) return {command.event_addr, 8};
            break;
        case K::WriteData:
            if (command.wd_valid && command.wd_data)
                return {command.wd_addr, uint64_t(command.wd_num) * 4};
            break;
        case K::DmaData:
            if (command.dd_valid && (command.dd_sels & 0xffu) != 1u)
                return {command.dd_dst, command.dd_bytes};
            break;
        default: break;
    }
    return {};
}

bool pending_memory_overlaps(PendingMemorySpan first, PendingMemorySpan second) {
    return first.address && first.bytes && second.address && second.bytes &&
           first.address <= UINT64_MAX - first.bytes &&
           second.address <= UINT64_MAX - second.bytes &&
           first.address < second.address + second.bytes &&
           second.address < first.address + first.bytes;
}

bool pending_memory_completion(const Pm4Command& command) {
    return command.kind == Pm4Command::Kind::ReleaseMem ||
           command.kind == Pm4Command::Kind::EventWrite;
}

PendingMemoryGeometry pending_memory_geometry(const GuestMappingLease& lease,
                                              PendingMemorySpan span) {
    PendingMemoryGeometry geometry{span, {}};
    // Wholly untracked host/loader memory retains its exact-VA behavior. A partially tracked or
    // uncommitted range cannot establish independence from a private physical alias.
    guest_memory_mapping_slices(lease, span.address, span.bytes, geometry.mappings,
                                &geometry.coverage);
    return geometry;
}

std::vector<PendingMemoryPatch> pending_memory_patches(const PendingMemoryGeometry& written,
                                                       const PendingMemoryGeometry& read) {
    std::vector<PendingMemoryPatch> patches;
    const auto add = [&](uint64_t a, uint64_t a_bytes, uint64_t a_offset, uint64_t b,
                         uint64_t b_bytes, uint64_t b_offset) {
        if (!a_bytes || !b_bytes || a > UINT64_MAX - a_bytes || b > UINT64_MAX - b_bytes) return;
        const uint64_t start = std::max(a, b), end = std::min(a + a_bytes, b + b_bytes);
        if (start >= end) return;
        PendingMemoryPatch patch{static_cast<size_t>(a_offset + start - a),
                                 static_cast<size_t>(b_offset + start - b),
                                 static_cast<size_t>(end - start)};
        if (std::none_of(patches.begin(), patches.end(), [&](const auto& other) {
                return other.source_offset == patch.source_offset &&
                       other.target_offset == patch.target_offset && other.bytes == patch.bytes;
            }))
            patches.push_back(patch);
    };
    if (written.virtual_span.address && read.virtual_span.address)
        add(written.virtual_span.address, written.virtual_span.bytes, 0, read.virtual_span.address,
            read.virtual_span.bytes, 0);
    for (const auto& a : written.mappings)
        if (a.direct)
            for (const auto& b : read.mappings)
                if (b.direct) add(a.physical, a.bytes, a.offset, b.physical, b.bytes, b.offset);
    return patches;
}

std::vector<bool> pending_renderer_selection(std::span<const Pm4Command> commands,
                                             std::span<const PendingMemoryGeometry> geometry) {
    struct Intervals {
        std::map<uint64_t, uint64_t> spans;
        bool overlaps(uint64_t begin, uint64_t bytes) const {
            if (!bytes || begin > UINT64_MAX - bytes) return true;
            const auto next = spans.upper_bound(begin);
            return (next != spans.begin() && std::prev(next)->second > begin) ||
                   (next != spans.end() && next->first < begin + bytes);
        }
        void add(uint64_t begin, uint64_t bytes) {
            if (!bytes || begin > UINT64_MAX - bytes) return;
            uint64_t end = begin + bytes;
            auto next = spans.lower_bound(begin);
            if (next != spans.begin() && std::prev(next)->second >= begin) --next;
            while (next != spans.end() && next->first <= end) {
                begin = std::min(begin, next->first);
                end = std::max(end, next->second);
                next = spans.erase(next);
            }
            spans.emplace(begin, end);
        }
    } virtual_spans, physical_spans;
    std::vector<bool> selected(commands.size(), false);
    if (commands.size() != geometry.size()) return selected;
    const auto retain = [&](const PendingMemoryGeometry& write) {
        if (write.virtual_span.address)
            virtual_spans.add(write.virtual_span.address, write.virtual_span.bytes);
        for (const auto& mapping : write.mappings)
            if (mapping.direct) physical_spans.add(mapping.physical, mapping.bytes);
    };
    bool uncertain_aliases = false;
    for (size_t i = 0; i < commands.size(); ++i)
        if (pending_memory_completion(commands[i])) {
            retain(geometry[i]);
            uncertain_aliases |= geometry[i].virtual_span.bytes &&
                                 geometry[i].coverage == GuestMemoryMappingCoverage::Incomplete;
        }
    for (size_t i = 0; i < commands.size(); ++i) {
        using K = Pm4Command::Kind;
        if (commands[i].kind != K::WriteData && commands[i].kind != K::DmaData) continue;
        // GDS offsets (including zero) cannot alias guest completion bytes. Prefix uploads still
        // have to execute before the first address-copy consumer, in their original FIFO order.
        if (commands[i].kind == K::DmaData && commands[i].dd_valid &&
            (commands[i].dd_sels & 0xffu) == 1u) {
            selected[i] = true;
            continue;
        }
        const auto& write = geometry[i];
        const bool incomplete = write.coverage == GuestMemoryMappingCoverage::Incomplete;
        bool blocked = uncertain_aliases || incomplete || !write.virtual_span.address ||
                       virtual_spans.overlaps(write.virtual_span.address, write.virtual_span.bytes);
        for (const auto& mapping : write.mappings)
            if (mapping.direct) blocked |= physical_spans.overlaps(mapping.physical, mapping.bytes);
        selected[i] = !blocked;
        if (blocked) retain(write);
        uncertain_aliases |= incomplete && write.virtual_span.bytes;
    }
    return selected;
}

PendingMemoryOverlay overlay_pending_memory_patch(const Pm4Command& command,
                                                  std::span<const uint8_t> captured_dma,
                                                  PendingMemoryPatch patch,
                                                  std::span<uint8_t> destination) {
    const auto written = pending_memory_span(command);
    if (!patch.bytes) return PendingMemoryOverlay::Disjoint;
    if (patch.source_offset > written.bytes || patch.bytes > written.bytes - patch.source_offset ||
        patch.target_offset > destination.size() ||
        patch.bytes > destination.size() - patch.target_offset)
        return PendingMemoryOverlay::Unresolved;
    const uint8_t* source = nullptr;
    using K = Pm4Command::Kind;
    if (command.kind == K::ReleaseMem && command.rel_data_sel != 3) {
        source = reinterpret_cast<const uint8_t*>(&command.rel_value);
    } else if (command.kind == K::WriteData) {
        source = reinterpret_cast<const uint8_t*>(command.wd_data);
    } else if (command.kind == K::DmaData) {
        if (captured_dma.size() >= written.bytes) {
            source = captured_dma.data();
        } else if (!(command.dd_sels & kDmaDataAddressSource) && command.dd_src <= UINT32_MAX &&
                   ((command.dd_sels >> 8u) & 0xffu) != 1u) {
            const uint32_t word = static_cast<uint32_t>(command.dd_src);
            const auto* pattern = reinterpret_cast<const uint8_t*>(&word);
            for (size_t n = 0; n < patch.bytes; ++n)
                destination[patch.target_offset + n] = pattern[(patch.source_offset + n) & 3u];
            return PendingMemoryOverlay::Applied;
        }
    }
    if (!source) return PendingMemoryOverlay::Unresolved;
    std::memcpy(destination.data() + patch.target_offset, source + patch.source_offset,
                patch.bytes);
    return PendingMemoryOverlay::Applied;
}

PendingMemoryOverlay overlay_pending_memory(const Pm4Command& command,
                                            std::span<const uint8_t> captured_dma, uint64_t address,
                                            std::span<uint8_t> destination) {
    const auto written = pending_memory_span(command);
    if (!pending_memory_overlaps(written, {address, destination.size()}))
        return PendingMemoryOverlay::Disjoint;
    const uint64_t start = std::max(address, written.address);
    const size_t count =
        std::min(address + destination.size(), written.address + written.bytes) - start;
    const size_t source_offset = start - written.address, target_offset = start - address;
    return overlay_pending_memory_patch(command, captured_dma,
                                        {source_offset, target_offset, count}, destination);
}
}   // namespace prosper::gpu
