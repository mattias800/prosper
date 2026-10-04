#pragma once
#include <array>
#include <cstdint>
#include <memory>

namespace prosper::gpu {
// Calling-thread residency tracks immutable analysis generations, never guest addresses. Cached
// payloads own copied code and this ledger, NOT the source owners recorded here. Only producing
// draws/completions and the normal analysis cache keep those generations alive. Uncertain lifetime
// is deliberately retained: it is not permission to evict an admitted warmed working set.
class FragmentDrawSourceGenerations {
public:
    void remember(const std::shared_ptr<const void>& source) {
        if (!source) {
            indeterminate_ = true;
            return;
        }
        std::weak_ptr<const void>* vacant = nullptr;
        for (auto& owner : owners_) {
            if (!owner.owner_before(source) && !source.owner_before(owner)) return;
            if (owner.expired()) vacant = &owner;
        }
        if (vacant)
            *vacant = source;
        else
            indeterminate_ = true; // metadata capacity is NOT a shader-admission cap
    }
    bool live() const {
        if (indeterminate_) return true;
        for (const auto& owner : owners_)
            if (!owner.expired()) return true;
        return false;
    }
    bool indeterminate() const { return indeterminate_; }
    bool owns(const std::shared_ptr<const void>& source) const {
        if (!source) return false;
        for (const auto& owner : owners_)
            if (!owner.owner_before(source) && !source.owner_before(owner)) {
                const auto retained = owner.lock();
                if (retained && retained.get() == source.get()) return true;
            }
        return false;
    }

private:
    std::array<std::weak_ptr<const void>, 64> owners_{};
    bool indeterminate_ = false;
};

struct FragmentDrawCacheStats {
    uint64_t program_compile_calls = 0, program_hits = 0, program_retired = 0;
    uint64_t compute_cold_builds = 0, collector_cold_builds = 0, framebuffer_cold_builds = 0;
    uint64_t compute_retired = 0, collector_retired = 0, framebuffer_retired = 0;
    // Direct Vulkan calls are counted at their call sites, including failed attempts. The
    // separately counted checked-module entry point may refuse before invoking the driver.
    uint64_t vk_object_create_calls = 0, vk_pipeline_create_calls = 0;
    uint64_t checked_shader_module_calls = 0;
    // Actual once-per-draw shared resource uploads, not wave count or planned byte estimates.
    uint64_t scalar_bank_uploads = 0, scalar_bank_payload_bytes = 0;
    // Actual command-recording arguments, not a completion or coherent-memory pixel oracle.
    uint64_t scalar_bank_visibility_barriers = 0;
};
inline FragmentDrawCacheStats& fragment_draw_cache_stats() {
    static thread_local FragmentDrawCacheStats stats;
    return stats;
}
// Cold insertion only. No global mutex or per-draw scan of the live working set on a cache hit.
// Erasing residence never revokes a genuine outstanding completion's shared payload lease.
template <class Cache>
void retire_dead_fragment_draw_entries(Cache& cache, uint64_t& retired) {
    for (auto item = cache.begin(); item != cache.end();) {
        if (!item->second->source_live()) {
            item = cache.erase(item);
            ++retired;
        } else
            ++item;
    }
}
} // namespace prosper::gpu
