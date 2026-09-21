#pragma once

#include "hle/memory/renderer_tracked_mapping.hpp"
#include "host/memory/guest_memory_map.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iterator>
#include <mutex>
#include <vector>

namespace prosper::detail {

inline std::atomic<uint64_t>& renderer_tracked_mapping_generation() {
    static std::atomic<uint64_t> generation{1};
    return generation;
}

// Call after every membership-changing g_maps swap while its mutex remains held. Sequential
// consistency is stronger than the release/acquire contract this cache needs and keeps the
// cross-platform invalidation order explicit to static analyzers.
inline void advance_renderer_tracked_mapping_generation() {
    renderer_tracked_mapping_generation().fetch_add(1);
}

template <typename Mapping>
ProsperRendererTrackedMappingResult renderer_guest_address_tracked(
    uint64_t addr, std::mutex& mapping_mutex, const std::vector<Mapping>& mappings) {
    if (addr == UINT64_MAX) return ProsperRendererTrackedMappingResult::Untracked;

    static thread_local host::GuestReadableRangeCache ranges;
    constexpr size_t kMaxCachedTrackedRanges = 4096;
    uint64_t generation = renderer_tracked_mapping_generation().load();
    ranges.sync_generation(generation);
    if (ranges.contains(addr, addr + 1))
        return ProsperRendererTrackedMappingResult::CachedTracked;

    std::scoped_lock lock(mapping_mutex);
    generation = renderer_tracked_mapping_generation().load();
    ranges.sync_generation(generation);
    const auto after = std::upper_bound(
        mappings.begin(), mappings.end(), addr,
        [](uint64_t address, const Mapping& mapping) {
            return address < mapping.base;
        });
    if (after == mappings.begin()) return ProsperRendererTrackedMappingResult::Untracked;
    const Mapping& mapping = *std::prev(after);
    if (addr < mapping.base || addr - mapping.base >= mapping.size)
        return ProsperRendererTrackedMappingResult::Untracked;
    if (ranges.range_count() >= kMaxCachedTrackedRanges) ranges.clear();
    ranges.insert(mapping.base, mapping.base + mapping.size);
    return ProsperRendererTrackedMappingResult::AuthoritativeTracked;
}

} // namespace prosper::detail
