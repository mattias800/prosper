#include "gpu/execute/derived_interpolation_geometry.hpp"
#include <map>
#include <mutex>
#include <utility>

namespace prosper::gpu {
namespace {
struct GeometryKey {
    std::weak_ptr<const ShaderCodeAnalysis> fragment;
    std::weak_ptr<const std::vector<uint32_t>> vertex;
    const ShaderCodeAnalysis* fragment_words = nullptr;
    const std::vector<uint32_t>* vertex_words = nullptr;
    std::vector<uint32_t> profile;
    bool operator<(const GeometryKey& other) const {
        if (fragment.owner_before(other.fragment)) return true;
        if (other.fragment.owner_before(fragment)) return false;
        if (vertex.owner_before(other.vertex)) return true;
        if (other.vertex.owner_before(vertex)) return false;
        if (fragment_words != other.fragment_words)
            return std::less<const ShaderCodeAnalysis*>{}(fragment_words, other.fragment_words);
        if (vertex_words != other.vertex_words)
            return std::less<const std::vector<uint32_t>*>{}(vertex_words, other.vertex_words);
        return profile < other.profile;
    }
};
// Process-wide: draws are realized on a worker pool, but their helper plans and collector
// pipelines are compiled and cached on the render thread keyed by this GS owner. Per-worker
// residence would give one identical profile up to configured_draw_realization_threads() cold
// plans and collector pipelines. The lock covers only lookup/insert, never the GS compile.
struct GeometryCache {
    std::mutex lock;
    std::map<GeometryKey, std::shared_ptr<const std::vector<uint32_t>>> entries;
    DerivedInterpolationGeometryStats stats;
};
GeometryCache& geometry_cache() {
    static GeometryCache cache;
    return cache;
}
std::vector<uint32_t> geometry_profile(const FragmentInterpolationLayout& layout, bool capture,
                                       bool rect, FloatTransportConfig transport) {
    // A nonexplicit transport's module also depends on the process float-controls verdict
    // (SpirvCompute::declare_float_controls), which a later publish can change. Explicit profiles
    // never read it. GS generation runs outside any CompilerChoiceScope, so this is the same live
    // value the compile consumes.
    const uint32_t float_controls =
        transport.explicit_nonfinite32() ? 2u : uint32_t(signed_zero_inf_nan_preserve_declared());
    std::vector<uint32_t> profile{layout.attribute_mask,
                                  layout.smooth_mask,
                                  layout.passthrough_mask,
                                  layout.flat_mask,
                                  uint32_t(layout.requires_geometry),
                                  uint32_t(layout.valid),
                                  uint32_t(capture),
                                  uint32_t(rect),
                                  uint32_t(transport.profile),
                                  float_controls};
    for (const auto& locations : layout.parameter_locations)
        profile.insert(profile.end(), locations.begin(), locations.end());
    profile.insert(profile.end(), layout.system_locations.begin(), layout.system_locations.end());
    return profile;
}
} // namespace

DerivedInterpolationGeometryStats derived_interpolation_geometry_stats() {
    auto& cache = geometry_cache();
    const std::lock_guard guard(cache.lock);
    return cache.stats;
}
std::shared_ptr<const std::vector<uint32_t>>
acquire_derived_interpolation_geometry(const std::shared_ptr<const ShaderCodeAnalysis>& fragment,
                                       const std::shared_ptr<const std::vector<uint32_t>>& vertex,
                                       const FragmentInterpolationLayout& layout, bool capture,
                                       bool rect, FloatTransportConfig transport) {
    auto& cache = geometry_cache();
    GeometryKey key{fragment, vertex, fragment.get(), vertex.get(),
                    geometry_profile(layout, capture, rect, transport)};
    // Missing immutable owners preserve ordinary uncached compilation. Same addresses/bytes,
    // public flags and unrelated draw entries cannot stand in for genuine producing versions.
    const bool cacheable = fragment && vertex;
    {
        const std::lock_guard guard(cache.lock);
        if (cacheable)
            if (const auto found = cache.entries.find(key); found != cache.entries.end()) {
                ++cache.stats.cache_hits;
                return found->second;
            }
        for (auto item = cache.entries.begin(); item != cache.entries.end();)
            if (item->first.fragment.expired() || item->first.vertex.expired()) {
                item = cache.entries.erase(item);
                ++cache.stats.retired;
            } else {
                ++item;
            }
        ++cache.stats.compile_calls;   // Actual emitter calls, including an empty/refused module.
    }
    auto result = std::make_shared<const std::vector<uint32_t>>(
        recompile_interpolation_geometry(layout, capture, rect, transport));
    // The compile re-reads the float-controls verdict the key captured. If a publish landed in
    // between, the bytes may belong to either value: return them, but never file them.
    if (!cacheable || geometry_profile(layout, capture, rect, transport) != key.profile)
        return result;
    const std::lock_guard guard(cache.lock);
    // Another worker may have compiled the same key meanwhile; keep ONE owner so the render
    // thread's plan/collector caches see a single generation.
    return cache.entries.emplace(std::move(key), result).first->second;
}
} // namespace prosper::gpu
