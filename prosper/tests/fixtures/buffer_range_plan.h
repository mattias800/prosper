#pragma once
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <vector>

namespace prosper::test {
// Metadata only: callers prove readable, immutable, direct guest backing. Connected overlaps
// never read a gap. Equal address residues make every child offset satisfy the device alignment.
struct BufferRangeSpan {
    uint64_t address = 0;
    uint64_t bytes = 0;
    bool operator==(const BufferRangeSpan&) const = default;
};
struct BufferRangeGroup {
    uint64_t address = 0;
    uint64_t bytes = 0;
    uint64_t distinct_bytes = 0;
};
inline std::vector<BufferRangeGroup> plan_buffer_ranges(
    std::vector<BufferRangeSpan> spans, uint64_t alignment, uint64_t maximum_bytes) {
    if (!alignment || maximum_bytes < 4096) return {};
    std::erase_if(spans, [=](const auto& s) {
        return !s.address || s.bytes < 4096 || s.bytes > maximum_bytes ||
               s.address > std::numeric_limits<uint64_t>::max() - s.bytes;
    });
    std::sort(spans.begin(), spans.end(), [=](const auto& a, const auto& b) {
        if (a.address % alignment != b.address % alignment)
            return a.address % alignment < b.address % alignment;
        if (a.address != b.address) return a.address < b.address;
        return a.bytes < b.bytes;
    });
    spans.erase(std::unique(spans.begin(), spans.end()), spans.end());
    std::vector<BufferRangeGroup> groups;
    BufferRangeGroup current;
    auto finish = [&] {
        if (current.distinct_bytes - current.bytes >= 4096) groups.push_back(current);
    };
    for (const auto& span : spans) {
        const uint64_t end = span.address + span.bytes;
        const uint64_t current_end = current.address + current.bytes;
        if (current.bytes && span.address % alignment == current.address % alignment &&
            span.address < current_end && end - current.address <= maximum_bytes &&
            current.distinct_bytes <= std::numeric_limits<uint64_t>::max() - span.bytes) {
            current.bytes = std::max(current_end, end) - current.address;
            current.distinct_bytes += span.bytes;
        } else {
            finish();
            current = {span.address, span.bytes, span.bytes};
        }
    }
    finish();
    return groups;
}
// Capped groups may overlap. Only return a group containing the complete requested span.
inline std::size_t find_buffer_range(const std::vector<BufferRangeGroup>& groups,
                               uint64_t address, uint64_t bytes, uint64_t alignment) {
    if (!alignment || !bytes) return groups.size();
    auto it = std::upper_bound(groups.begin(), groups.end(), address,
        [=](uint64_t a, const BufferRangeGroup& b) {
            if (a % alignment != b.address % alignment)
                return a % alignment < b.address % alignment;
            return a < b.address;
        });
    if (it == groups.begin()) return groups.size();
    --it;
    if (it->address % alignment != address % alignment || address < it->address ||
        address - it->address > it->bytes || bytes > it->bytes - (address - it->address))
        return groups.size();
    return static_cast<std::size_t>(it - groups.begin());
}
} // namespace prosper::test
