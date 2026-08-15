#include "compute_tree_watch.hpp"

#include <cstdlib>
#include <string>

namespace prosper::gpu {
namespace {

// Shared strict field parser: rejects an empty field, trailing garbage, and a value wider than the
// field. Returning the end pointer lets the caller demand the exact separator it expects rather
// than accepting any punctuation.
bool parse_field(const char*& cursor, uint64_t& out, uint64_t limit) {
    if (!cursor || !*cursor) return false;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(cursor, &end, 0);
    if (end == cursor) return false;
    if (value > limit) return false;
    out = value;
    cursor = end;
    return true;
}

bool expect_separator(const char*& cursor) {
    if (!cursor || *cursor != ':') return false;
    ++cursor;
    return true;
}

} // namespace

std::optional<ComputeTreeWatchSelector> parse_compute_tree_watch_selector(const char* value) {
    if (!value) return std::nullopt;
    const char* cursor = value;
    ComputeTreeWatchSelector selector;

    uint64_t addr = 0;
    if (!parse_field(cursor, addr, UINT64_MAX)) return std::nullopt;
    if (!addr) return std::nullopt;
    if (!expect_separator(cursor)) return std::nullopt;

    uint64_t records = 0;
    // A dword count, and one large enough to allocate blindly per dispatch is a mistake worth
    // refusing at parse time rather than discovering as a stall in a routed run.
    constexpr uint64_t kMaximumRecords = 1u << 20u;
    if (!parse_field(cursor, records, kMaximumRecords)) return std::nullopt;
    if (!records) return std::nullopt;

    selector.addr = addr;
    selector.records = static_cast<uint32_t>(records);

    if (!*cursor) return selector;

    // Shift and mask are optional but arrive together: a shift without a mask would silently keep
    // the default mask, which is a different walk from the one the caller asked for.
    if (!expect_separator(cursor)) return std::nullopt;
    uint64_t shift = 0;
    if (!parse_field(cursor, shift, 31u)) return std::nullopt;
    if (!expect_separator(cursor)) return std::nullopt;
    uint64_t mask = 0;
    if (!parse_field(cursor, mask, UINT32_MAX)) return std::nullopt;
    if (!mask) return std::nullopt;
    if (*cursor) return std::nullopt;

    selector.index_shift = static_cast<uint32_t>(shift);
    selector.index_mask = static_cast<uint32_t>(mask);
    return selector;
}

std::vector<ComputeTreeWatchDelta> compute_tree_watch_deltas(
        std::span<const uint32_t> before, std::span<const uint32_t> after, uint32_t limit,
        uint32_t* total_changed) {
    std::vector<ComputeTreeWatchDelta> deltas;
    uint32_t changed = 0;
    // Comparing only the overlap keeps a resized observation from reporting every trailing slot as
    // a change; the caller sees the size difference in the record counts it already prints.
    const size_t common = std::min(before.size(), after.size());
    for (size_t i = 0; i < common; ++i) {
        if (before[i] == after[i]) continue;
        ++changed;
        if (deltas.size() < limit)
            deltas.push_back({static_cast<uint32_t>(i), before[i], after[i]});
    }
    if (total_changed) *total_changed = changed;
    return deltas;
}

const char* compute_tree_watch_transition_name(ComputeTreeWatchTransition transition) {
    switch (transition) {
        case ComputeTreeWatchTransition::Unchanged: return "unchanged";
        case ComputeTreeWatchTransition::CleanToClean: return "clean->clean";
        case ComputeTreeWatchTransition::CleanToCyclic: return "clean->cyclic";
        case ComputeTreeWatchTransition::CyclicToClean: return "cyclic->clean";
        case ComputeTreeWatchTransition::CyclicToCyclic: return "cyclic->cyclic";
    }
    return "unknown";
}

ComputeTreeWatchTransition classify_compute_tree_watch_transition(
        const ComputeParentWalkReport& before, const ComputeParentWalkReport& after,
        bool bytes_changed) {
    if (!bytes_changed) return ComputeTreeWatchTransition::Unchanged;
    const bool before_cyclic = before.distinct_cycles != 0u;
    const bool after_cyclic = after.distinct_cycles != 0u;
    if (!before_cyclic && !after_cyclic) return ComputeTreeWatchTransition::CleanToClean;
    if (!before_cyclic) return ComputeTreeWatchTransition::CleanToCyclic;
    if (!after_cyclic) return ComputeTreeWatchTransition::CyclicToClean;
    return ComputeTreeWatchTransition::CyclicToCyclic;
}

ComputeTreeSiblingReport analyze_compute_tree_siblings(std::span<const uint32_t> words) {
    ComputeTreeSiblingReport report;
    report.records = static_cast<uint32_t>(std::min<size_t>(words.size(), UINT32_MAX));
    for (size_t i = 0; i < words.size(); ++i) {
        if ((words[i] & kComputeTreeSiblingSideBit) == 0u) continue;
        ++report.side_bit_set;
        // A side-bit record whose predecessor is its exact mate forms a pair. Index 0 can never
        // pair (it has no predecessor), which is consistent with the root being the unpaired node.
        //
        // The head must NOT already carry the side bit. Without that clause two identical adjacent
        // records both carrying it would satisfy `words[i] == (words[i-1] | bit)` trivially and be
        // counted as a sibling pair -- inflating the metric with duplicates, which is the opposite
        // of what an unpaired-record count is for.
        if (i != 0 && (words[i - 1] & kComputeTreeSiblingSideBit) == 0u &&
            words[i] == (words[i - 1] | kComputeTreeSiblingSideBit))
            ++report.pairs;
        else
            ++report.unpaired_side;
    }
    return report;
}

} // namespace prosper::gpu
