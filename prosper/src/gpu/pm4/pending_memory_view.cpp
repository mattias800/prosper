#include "gpu/pm4/pending_memory_view.hpp"
#include "gpu/pm4/command_processor.hpp"
#include <algorithm>
#include <cstring>
#include <iterator>
#include <map>

namespace prosper::gpu {
// CONFIDENCE: HIGH on byte identity and FIFO fragments: #2220's actual-submit regressions exercise
// distinct direct-memory VAs, protection splits, partial copies, and overlapping resource suffixes.
PendingMemorySpan pending_memory_span(const Pm4Command& command) {
    using K = Pm4Command::Kind;
    const auto checked = [](uint64_t address, uint64_t bytes) -> PendingMemorySpan {
        return address && bytes && address <= UINT64_MAX - bytes ? PendingMemorySpan{address, bytes}
                                                                 : PendingMemorySpan{};
    };
    switch (command.kind) {
        case K::ReleaseMem:
            if (!command.rel_addr || (command.rel_addr & 3u)) break;
            if (command.rel_data_sel == 3 ||
                (command.rel_value_valid &&
                 (command.rel_data_sel == 1 || command.rel_data_sel == 2)))
                return checked(command.rel_addr, command.rel_data_sel == 1 ? 4u : 8u);
            break;
        case K::EventWrite:
            if (command.event_addr && !(command.event_addr & 3u))
                return checked(command.event_addr, 8);
            break;
        case K::WriteData:
            if (command.wd_valid && command.wd_data && command.wd_num && command.wd_addr &&
                !(command.wd_addr & 3u))
                return checked(command.wd_addr, uint64_t(command.wd_num) * 4);
            break;
        case K::DmaData: {
            const bool immediate =
                !(command.dd_sels & kDmaDataAddressSource) && command.dd_src <= UINT32_MAX;
            // Match dma_data_form's static no-store shapes before a private overlay can supply
            // their bytes. Runtime access checks still occur at the actual guarded store.
            if (command.dd_valid && command.dd_bytes && command.dd_bytes <= 0x10000000u &&
                command.dd_dst >= 0x10000 && (command.dd_sels & 0xffu) != 1u &&
                ((command.dd_sels >> 8u) & 0xffu) != 1u &&
                (!immediate || (!(command.dd_dst & 3u) && command.dd_bytes <= 0x1000000u)))
                return checked(command.dd_dst, command.dd_bytes);
            break;
        }
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

PendingMemoryGeometry pending_memory_subgeometry(const PendingMemoryGeometry& shape,
                                                 PendingMemoryRange range) {
    PendingMemoryGeometry result{
        {shape.virtual_span.address + range.offset, range.bytes}, {}, shape.coverage};
    for (const auto& mapping : shape.mappings) {
        const uint64_t begin = std::max<uint64_t>(range.offset, mapping.offset);
        const uint64_t end =
            std::min<uint64_t>(range.offset + range.bytes, mapping.offset + mapping.bytes);
        if (begin < end)
            result.mappings.push_back(
                {begin - range.offset, end - begin,
                 mapping.direct ? mapping.physical + begin - mapping.offset : 0, mapping.direct});
    }
    return result;
}

namespace {
void add_interval(std::map<uint64_t, uint64_t>& spans, uint64_t begin, uint64_t bytes) {
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
}   // namespace

void PendingMemoryIndex::add(const PendingMemoryGeometry& shape) {
    if (!shape.virtual_span.address || !shape.virtual_span.bytes) return;
    add_interval(virtual_, shape.virtual_span.address, shape.virtual_span.bytes);
    for (const auto& mapping : shape.mappings)
        if (mapping.direct) add_interval(physical_, mapping.physical, mapping.bytes);
    incomplete_ |= shape.coverage == GuestMemoryMappingCoverage::Incomplete;
}

bool PendingMemoryIndex::empty() const {
    return !incomplete_ && virtual_.empty() && physical_.empty();
}

std::vector<PendingMemoryRange>
PendingMemoryIndex::project(const PendingMemoryGeometry& shape) const {
    const size_t bytes = shape.virtual_span.bytes;
    if (!bytes || empty()) return {};
    if (incomplete_ || shape.coverage == GuestMemoryMappingCoverage::Incomplete)
        return {{0, bytes}};
    std::map<uint64_t, uint64_t> offsets;
    const auto intersect = [&](const auto& spans, uint64_t begin, uint64_t count, uint64_t offset) {
        if (!count || begin > UINT64_MAX - count) return;
        auto next = spans.upper_bound(begin);
        if (next != spans.begin()) --next;
        for (; next != spans.end() && next->first < begin + count; ++next) {
            const uint64_t start = std::max(begin, next->first);
            const uint64_t end = std::min(begin + count, next->second);
            if (start < end) add_interval(offsets, offset + start - begin, end - start);
        }
    };
    intersect(virtual_, shape.virtual_span.address, bytes, 0);
    for (const auto& mapping : shape.mappings)
        if (mapping.direct) intersect(physical_, mapping.physical, mapping.bytes, mapping.offset);
    std::vector<PendingMemoryRange> result;
    result.reserve(offsets.size());
    for (const auto& [begin, end] : offsets) result.push_back({begin, end - begin});
    return result;
}

std::vector<PendingMemoryRange>
pending_memory_complement(size_t bytes, std::span<const PendingMemoryRange> private_ranges) {
    std::vector<PendingMemoryRange> result;
    size_t begin = 0;
    for (const auto& range : private_ranges) {
        if (begin < range.offset) result.push_back({begin, range.offset - begin});
        begin = range.offset + range.bytes;
    }
    if (begin < bytes) result.push_back({begin, bytes - begin});
    return result;
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

void PendingMemoryOverlayState::apply(const Pm4Command& command,
                                      std::span<const uint8_t> captured_dma,
                                      PendingMemoryPatch patch, std::span<uint8_t> destination) {
    if (patch.target_offset > destination.size() ||
        patch.bytes > destination.size() - patch.target_offset) {
        invalid_ = true;
        return;
    }
    const auto outcome = overlay_pending_memory_patch(command, captured_dma, patch, destination);
    const uint64_t begin = patch.target_offset, end = begin + patch.bytes;
    if (outcome == PendingMemoryOverlay::Unresolved) {
        add_interval(unknown_, begin, patch.bytes);
        return;
    }
    if (outcome != PendingMemoryOverlay::Applied) return;
    auto next = unknown_.upper_bound(begin);
    if (next != unknown_.begin()) --next;
    while (next != unknown_.end() && next->first < end) {
        const auto [old_begin, old_end] = *next;
        if (old_end <= begin) {
            ++next;
            continue;
        }
        next = unknown_.erase(next);
        if (old_begin < begin) unknown_.emplace(old_begin, begin);
        if (old_end > end) {
            unknown_.emplace(end, old_end);
            break;
        }
    }
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
