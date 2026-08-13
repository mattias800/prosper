#include "compute_parent_walk.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {

namespace {

bool parse_field(const char*& cursor, uint64_t maximum, uint64_t& result, bool final) {
    if (!cursor || !*cursor || *cursor == '-') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(cursor, &end, 0);
    if (errno == ERANGE || end == cursor || parsed > maximum) return false;
    if (final) {
        if (*end) return false;
    } else {
        if (*end != ':') return false;
        ++end;
    }
    result = static_cast<uint64_t>(parsed);
    cursor = end;
    return true;
}

} // namespace

std::optional<ComputeParentWalkSelector> parse_compute_parent_walk_selector(
        const char* value) {
    if (!value || !*value) return std::nullopt;
    const char* cursor = value;
    uint64_t program = 0, fetch_pc = 0, shift = 0, mask = 0, deep = 64;
    if (!parse_field(cursor, UINT64_MAX, program, false) || !program ||
        !parse_field(cursor, UINT32_MAX, fetch_pc, false) ||
        !parse_field(cursor, 31u, shift, false))
        return std::nullopt;

    // MASK is the only field whose delimiter is optional.
    if (!cursor || !*cursor || *cursor == '-') return std::nullopt;
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed_mask = std::strtoull(cursor, &end, 0);
    if (errno == ERANGE || end == cursor || parsed_mask > UINT32_MAX || !parsed_mask)
        return std::nullopt;
    mask = parsed_mask;
    if (*end == ':') {
        cursor = end + 1;
        if (!parse_field(cursor, UINT32_MAX, deep, true) || !deep) return std::nullopt;
    } else if (*end) {
        return std::nullopt;
    }

    return ComputeParentWalkSelector{
        program,
        static_cast<uint32_t>(fetch_pc),
        static_cast<uint32_t>(shift),
        static_cast<uint32_t>(mask),
        static_cast<uint32_t>(deep),
    };
}

ComputeParentWalkReport analyze_compute_parent_walk(
        std::span<const uint32_t> words, uint32_t root_count,
        uint32_t index_shift, uint32_t index_mask) {
    ComputeParentWalkReport report;
    report.records = static_cast<uint32_t>(std::min<size_t>(words.size(), UINT32_MAX));
    report.roots = std::min(root_count, report.records);
    if (!report.records || !report.roots || index_shift >= 32u || !index_mask)
        return report;

    std::vector<uint32_t> seen_generation(report.records, 0u);
    std::vector<uint32_t> seen_depth(report.records, 0u);
    std::unordered_set<uint32_t> cycle_keys;
    for (uint32_t root = 0; root < report.roots; ++root) {
        const uint32_t generation = root + 1u;
        uint32_t index = root;
        uint32_t depth = 0;
        while (index != 0u) {
            if (index >= report.records) {
                // The raw buffer load itself executes and robustly returns zero.
                ++depth;
                ++report.oob_roots;
                break;
            }
            if (seen_generation[index] == generation) {
                ++report.cyclic_roots;
                const uint32_t cycle_depth = seen_depth[index];
                const uint32_t cycle_length = depth - cycle_depth;
                uint32_t canonical = index;
                uint32_t member = index;
                for (uint32_t step = 0; step < cycle_length; ++step) {
                    canonical = std::min(canonical, member);
                    member = (words[member] >> index_shift) & index_mask;
                }
                cycle_keys.insert(canonical);
                break;
            }
            seen_generation[index] = generation;
            seen_depth[index] = depth;
            index = (words[index] >> index_shift) & index_mask;
            ++depth;
        }
        if (depth > report.max_depth) {
            report.max_depth = depth;
            report.max_depth_root = root;
        }
    }
    report.distinct_cycles = static_cast<uint32_t>(cycle_keys.size());
    return report;
}

bool compute_parent_walk_suspicious(
        const ComputeParentWalkReport& report, uint32_t deep_threshold) {
    return report.distinct_cycles != 0u || report.max_depth > deep_threshold;
}

} // namespace prosper::gpu
