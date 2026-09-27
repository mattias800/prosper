#include "gpu/memory/texture_cache_budget.hpp"

#include <algorithm>

namespace prosper::gpu {

namespace {
uint64_t saturating_sub(uint64_t a, uint64_t b) { return a > b ? a - b : 0; }
// a * pct / 100 without overflowing for any heap a 64-bit size can describe.
uint64_t percent_of(uint64_t a, uint64_t pct) { return a / 100 * pct + a % 100 * pct / 100; }
}  // namespace

uint64_t texture_cache_heuristic_budget(uint64_t heap_size) {
    return std::clamp<uint64_t>(heap_size / kTextureCacheHeuristicDivisor,
                                kTextureCacheFloorBytes, kTextureCacheHeuristicCeilingBytes);
}

TextureCacheBudget texture_cache_budget(const TextureCacheBudgetInputs& in) {
    TextureCacheBudget out;
    const uint64_t heuristic = texture_cache_heuristic_budget(in.heap_size);
    if (!in.have_budget || in.heap_budget == 0) {
        out.bytes = heuristic;
        out.source = TextureCacheBudgetSource::heuristic;
        return out;
    }
    out.non_texture = saturating_sub(std::max(in.heap_usage, in.prosper_held), in.texture_bytes);
    out.target = percent_of(in.heap_budget, kTextureCacheTargetPercentOfBudget);
    const uint64_t live = saturating_sub(out.target, out.non_texture);
    uint64_t bytes = live;
    out.source = TextureCacheBudgetSource::live;
    const uint64_t ceiling = percent_of(in.heap_size, kTextureCacheCeilingPercentOfHeap);
    if (bytes > ceiling) {
        bytes = ceiling;
        out.source = TextureCacheBudgetSource::ceiling;
    }
    if (in.unified && bytes > heuristic) {
        bytes = heuristic;
        out.source = TextureCacheBudgetSource::unified;
    }
    if (bytes < kTextureCacheFloorBytes) {
        bytes = kTextureCacheFloorBytes;
        out.source = TextureCacheBudgetSource::floor;
    }
    out.bytes = bytes;
    return out;
}

bool texture_cache_budget_changed_materially(uint64_t current, uint64_t proposed) {
    const uint64_t delta = current > proposed ? current - proposed : proposed - current;
    return delta >= std::max<uint64_t>(128ull * kTextureCacheMiB, current / 16);
}

uint32_t texture_cache_budget_heap(const uint64_t* heap_sizes, const bool* heap_device_local,
                                   uint32_t heap_count, uint32_t recorded_heap) {
    if (recorded_heap < heap_count && heap_device_local[recorded_heap]) return recorded_heap;
    uint32_t best = UINT32_MAX;
    for (uint32_t i = 0; i < heap_count; ++i)
        if (heap_device_local[i] && (best == UINT32_MAX || heap_sizes[i] > heap_sizes[best]))
            best = i;
    return best;
}

uint64_t resolve_texture_cache_limit(bool have_override, uint64_t override_bytes,
                                     uint64_t policy_bytes) {
    if (have_override) return override_bytes;
    return policy_bytes ? policy_bytes : kTextureCacheFloorBytes;
}

const char* texture_cache_budget_source_name(TextureCacheBudgetSource source) {
    switch (source) {
        case TextureCacheBudgetSource::heuristic: return "heuristic";
        case TextureCacheBudgetSource::live: return "live";
        case TextureCacheBudgetSource::ceiling: return "heap-ceiling";
        case TextureCacheBudgetSource::unified: return "unified-cap";
        case TextureCacheBudgetSource::floor: return "floor";
    }
    return "?";
}

}  // namespace prosper::gpu
