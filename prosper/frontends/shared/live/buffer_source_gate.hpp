#pragma once

#include "hle/memory/renderer_tracked_mapping.hpp"

#include <cstdint>

namespace prosper::frontend {

struct BufferSourceGateResult {
    bool unavailable = false;
    bool tracked_cache_hit = false;
    bool tracked_cache_fill = false;
    bool tracked_untracked_miss = false;
    bool reserved_state_query = false;
};

struct BufferSourceGateCounters {
    uint64_t tracked_cache_hits = 0;
    uint64_t tracked_cache_fills = 0;
    uint64_t tracked_untracked_misses = 0;
    uint64_t reserved_state_queries = 0;

    void record(const BufferSourceGateResult& result) {
        if (result.tracked_cache_hit) ++tracked_cache_hits;
        if (result.tracked_cache_fill) ++tracked_cache_fills;
        if (result.tracked_untracked_miss) ++tracked_untracked_misses;
        if (result.reserved_state_query) ++reserved_state_queries;
    }

    void add(const BufferSourceGateCounters& other) {
        tracked_cache_hits += other.tracked_cache_hits;
        tracked_cache_fills += other.tracked_cache_fills;
        tracked_untracked_misses += other.tracked_untracked_misses;
        reserved_state_queries += other.reserved_state_queries;
    }
};

// Keep the renderer's admission policy independently testable. The membership cache only avoids
// the numeric query after proving a positive g_maps membership; misses retain the original query
// because POSIX AMM-decline state is not represented by g_maps.
template <typename TrackedProbe, typename ReservedStateQuery>
inline BufferSourceGateResult classify_buffer_source(
    bool has_host_data, uint64_t gpu_addr, bool use_tracked_cache,
    TrackedProbe probe_tracked, ReservedStateQuery query_reserved_state) {
    BufferSourceGateResult result;
    if (has_host_data) return result;
    if (gpu_addr < 0x1000) {
        result.unavailable = true;
        return result;
    }

    ProsperRendererTrackedMappingResult tracked =
        ProsperRendererTrackedMappingResult::Untracked;
    if (use_tracked_cache) {
        tracked = probe_tracked(gpu_addr);
        result.tracked_cache_hit =
            tracked == ProsperRendererTrackedMappingResult::CachedTracked;
        result.tracked_cache_fill =
            tracked == ProsperRendererTrackedMappingResult::AuthoritativeTracked;
        result.tracked_untracked_miss =
            tracked == ProsperRendererTrackedMappingResult::Untracked;
    }
    if (tracked == ProsperRendererTrackedMappingResult::Untracked) {
        result.reserved_state_query = true;
        result.unavailable = query_reserved_state(gpu_addr) == 0;
    }
    return result;
}

} // namespace prosper::frontend
