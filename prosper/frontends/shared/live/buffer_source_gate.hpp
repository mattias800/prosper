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

// The draw path's gate for one resource. A replayed capture placeholder (#3807,
// src/gpu/capture/capture_source_gate.hpp) was refused by THIS gate in the live process, so it is
// refused here by its record, before the replay process's own mapping table is consulted: what is
// mapped in an offline tool says nothing about what was mapped in the game.
template <typename TrackedProbe, typename ReservedStateQuery>
inline BufferSourceGateResult classify_recorded_buffer_source(
    bool recorded_unavailable, bool has_host_data, uint64_t gpu_addr, bool use_tracked_cache,
    TrackedProbe probe_tracked, ReservedStateQuery query_reserved_state) {
    if (recorded_unavailable && !has_host_data) {
        BufferSourceGateResult result;
        result.unavailable = true;
        return result;
    }
    return classify_buffer_source(has_host_data, gpu_addr, use_tracked_cache, probe_tracked,
                                  query_reserved_state);
}

// The same decision as the capture asks it (set_gpu_capture_buffer_source_probe): the authoritative
// lookup, never the positive-only membership cache, for a resource with no host-owned bytes.
template <typename TrackedProbe, typename ReservedStateQuery>
inline bool capture_buffer_source_unavailable(uint64_t gpu_addr, TrackedProbe probe_tracked,
                                              ReservedStateQuery query_reserved_state) {
    return classify_buffer_source(false, gpu_addr, false, probe_tracked, query_reserved_state)
        .unavailable;
}

} // namespace prosper::frontend
