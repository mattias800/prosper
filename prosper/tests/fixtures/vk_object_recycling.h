#pragma once
// Bounded free lists for the small Vulkan objects the renderer used to create and destroy on every
// pass or batch: descriptor pools (per render pass and per GPU detile upload), batch fences, and the
// two-entry timestamp query pools each batch measures itself with. On RADV a descriptor pool and a
// query pool are each a kernel buffer object, and a fence is a DRM syncobj; creating and destroying
// them per pass showed up as ~9% of the GTA V submitting thread (2026-09-27, #3873).
#include "diagnostics/env_numeric.hpp"
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace prosper::test {

// ---------------------------------------------------------------------------------------------
// Bounded VkDescriptorPool free list -- the descriptor-pool twin of the command-pool list above.
//
// Every render pass used to create one descriptor pool sized exactly to its draws and destroy it
// when the pass completed. On RADV each pool is a kernel buffer object: create, GPU-VA map, CPU map,
// and on destroy the reverse. Measured on the routed Grand Theft Auto V window (2026-09-27, strace
// of the submitting thread): ~13,100 BO creates and as many closes per 10 s, most of them 48-464
// byte descriptor-pool buffers, and the BO lifecycle at ~9% of that thread's CPU. The per-pass pool
// was the largest single caller.
//
// A retired pool is RESET (which releases its sets but keeps its buffer object) and handed to the
// next pass whose requirement it covers. Capacities are rounded up to a power of two at creation so
// passes with slightly different draw counts share pools instead of each needing an exact fit.
//
// Retirement is completion-gated exactly like the command pool: the pass's cleanup, which runs only
// after the GPU has signalled the pass complete, is the only caller of release. vkResetDescriptorPool
// requires that no command buffer using its sets is still pending, which that ordering guarantees.
//
// BOUNDED: each retained pool keeps its buffer object, so the list is capped
// (PROSPER_RENDER_DESCRIPTOR_POOL_CACHE, default 32). PROSPER_NO_RENDER_DESCRIPTOR_POOL_REUSE restores the old
// exact-size create/destroy behaviour and is the negative control the test uses.
struct RenderDescriptorPoolCapacity {
    uint32_t sets = 0;
    uint32_t storage_buffers = 0;
    uint32_t sampled_images = 0;
    uint32_t storage_images = 0;
    bool covers(const RenderDescriptorPoolCapacity& need) const {
        return sets >= need.sets && storage_buffers >= need.storage_buffers &&
               sampled_images >= need.sampled_images && storage_images >= need.storage_images;
    }
    uint64_t total() const {
        return uint64_t(sets) + storage_buffers + sampled_images + storage_images;
    }
};

struct RenderDescriptorPoolLease {
    VkDescriptorPool pool = VK_NULL_HANDLE;
    RenderDescriptorPoolCapacity capacity{};
    explicit operator bool() const { return pool != VK_NULL_HANDLE; }
};

struct RenderDescriptorPoolEntry {
    VkDevice device = VK_NULL_HANDLE;
    RenderDescriptorPoolLease lease{};
};

struct RenderDescriptorPoolCache {
    std::mutex mutex;
    std::vector<RenderDescriptorPoolEntry> available;
    uint64_t hits = 0, misses = 0, retired = 0, destroyed = 0;
};

struct RenderDescriptorPoolStats {
    uint64_t hits = 0, misses = 0, retired = 0, destroyed = 0;
    size_t cached = 0;
};

inline RenderDescriptorPoolCache& render_descriptor_pool_cache() {
    static RenderDescriptorPoolCache cache;
    return cache;
}

inline bool render_descriptor_pool_reuse_enabled() {
    static const bool enabled = getenv("PROSPER_NO_RENDER_DESCRIPTOR_POOL_REUSE") == nullptr;
    return enabled;
}

inline size_t render_descriptor_pool_cache_limit() {
    static const size_t limit = []() -> size_t {
        const char* value = getenv("PROSPER_RENDER_DESCRIPTOR_POOL_CACHE");
        return static_cast<size_t>(prosper::diag::env_u64_or_default_capped(
            "PROSPER_RENDER_DESCRIPTOR_POOL_CACHE", value, 32ull, 256ull, "pools"));
    }();
    return limit;
}

// Round a requirement up so neighbouring passes can share one pool. The floor keeps tiny passes
// (one or two draws) from each minting their own size class.
inline uint32_t render_descriptor_pool_round_up(uint32_t value) {
    uint32_t rounded = 16;
    while (rounded < value && rounded < (1u << 30)) rounded <<= 1;
    return rounded;
}

inline RenderDescriptorPoolStats render_descriptor_pool_stats() {
    RenderDescriptorPoolCache& cache = render_descriptor_pool_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    return RenderDescriptorPoolStats{cache.hits, cache.misses, cache.retired, cache.destroyed,
                                     cache.available.size()};
}

// Returns a pool whose capacity covers `need`, with no sets allocated from it. A null lease means
// vkCreateDescriptorPool failed; the caller treats that exactly like the old failed create.
inline RenderDescriptorPoolLease acquire_render_descriptor_pool(
        VkDevice device, const RenderDescriptorPoolCapacity& need) {
    const bool reuse = render_descriptor_pool_reuse_enabled();
    if (reuse) {
        RenderDescriptorPoolCache& cache = render_descriptor_pool_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        // Smallest covering pool, so one oversized pool is not spent on a one-draw pass while a
        // large pass that needs it creates another.
        size_t best = cache.available.size();
        for (size_t i = 0; i < cache.available.size(); ++i) {
            const RenderDescriptorPoolEntry& entry = cache.available[i];
            if (entry.device != device || !entry.lease.capacity.covers(need)) continue;
            if (best == cache.available.size() ||
                entry.lease.capacity.total() < cache.available[best].lease.capacity.total())
                best = i;
        }
        if (best != cache.available.size()) {
            const RenderDescriptorPoolLease lease = cache.available[best].lease;
            cache.available[best] = cache.available.back();
            cache.available.pop_back();
            ++cache.hits;
            return lease;
        }
        ++cache.misses;
    }
    RenderDescriptorPoolCapacity capacity = need;
    if (reuse) {
        capacity.sets = render_descriptor_pool_round_up(need.sets);
        capacity.storage_buffers = render_descriptor_pool_round_up(need.storage_buffers);
        capacity.sampled_images = render_descriptor_pool_round_up(need.sampled_images);
        capacity.storage_images = render_descriptor_pool_round_up(need.storage_images);
    }
    VkDescriptorPoolSize sizes[3] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, std::max<uint32_t>(capacity.storage_buffers, 1)},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, std::max<uint32_t>(capacity.sampled_images, 1)},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, std::max<uint32_t>(capacity.storage_images, 1)},
    };
    VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    info.maxSets = std::max<uint32_t>(capacity.sets, 1);
    info.poolSizeCount = 3;
    info.pPoolSizes = sizes;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(device, &info, nullptr, &pool) != VK_SUCCESS || !pool) return {};
    return RenderDescriptorPoolLease{pool, capacity};
}

// Completion-gated: call only once no pending command buffer references a set from this pool.
inline void release_render_descriptor_pool(VkDevice device,
                                           const RenderDescriptorPoolLease& lease) {
    if (!lease.pool) return;
    if (render_descriptor_pool_reuse_enabled()) {
        RenderDescriptorPoolCache& cache = render_descriptor_pool_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.available.size() < render_descriptor_pool_cache_limit() &&
            vkResetDescriptorPool(device, lease.pool, 0) == VK_SUCCESS) {
            cache.available.push_back(RenderDescriptorPoolEntry{device, lease});
            ++cache.retired;
            return;
        }
        ++cache.destroyed;
    }
    vkDestroyDescriptorPool(device, lease.pool, nullptr);
}

// Destroys every retained pool for `device`. The device owner calls this before vkDestroyDevice;
// tests use it to start from a known-empty list.
inline void drain_render_descriptor_pool_cache(VkDevice device) {
    RenderDescriptorPoolCache& cache = render_descriptor_pool_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    for (size_t i = cache.available.size(); i-- > 0;) {
        if (cache.available[i].device != device) continue;
        vkDestroyDescriptorPool(device, cache.available[i].lease.pool, nullptr);
        cache.available[i] = cache.available.back();
        cache.available.pop_back();
        ++cache.destroyed;
    }
}

// ---------------------------------------------------------------------------------------------
// Batch fences and batch timestamp query pools.
//
// BackendSubmissionBatch::submit_and_wait created a fence and (when timing) a two-query timestamp
// pool per batch and destroyed both after the wait: ~3,600 syncobj create/destroy pairs per 10 s on
// the routed GTA V window. Both are recycled here once the batch is proven complete. A fence goes
// back UNSIGNALED (reset on release). A query pool needs no host reset: the batch records
// vkCmdResetQueryPool before its first timestamp, exactly as it did for a fresh pool.
//
// PROSPER_NO_RENDER_SYNC_OBJECT_REUSE restores create/destroy per batch.
inline bool render_sync_object_reuse_enabled() {
    static const bool enabled = getenv("PROSPER_NO_RENDER_SYNC_OBJECT_REUSE") == nullptr;
    return enabled;
}

struct RenderSyncObjectCache {
    std::mutex mutex;
    std::vector<std::pair<VkDevice, VkFence>> fences;
    std::vector<std::pair<VkDevice, VkQueryPool>> timestamp_pools;
    uint64_t fence_hits = 0, fence_misses = 0, pool_hits = 0, pool_misses = 0;
};

inline RenderSyncObjectCache& render_sync_object_cache() {
    static RenderSyncObjectCache cache;
    return cache;
}

// Enough for every batch that can be in flight at once; beyond it, release destroys.
inline constexpr size_t kRenderSyncObjectCacheLimit = 64;

// An UNSIGNALED fence, or VK_NULL_HANDLE if creation failed.
inline VkFence acquire_render_batch_fence(VkDevice device) {
    if (render_sync_object_reuse_enabled()) {
        RenderSyncObjectCache& cache = render_sync_object_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        for (size_t i = cache.fences.size(); i-- > 0;) {
            if (cache.fences[i].first != device) continue;
            const VkFence fence = cache.fences[i].second;
            cache.fences[i] = cache.fences.back();
            cache.fences.pop_back();
            ++cache.fence_hits;
            return fence;
        }
        ++cache.fence_misses;
    }
    VkFenceCreateInfo info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(device, &info, nullptr, &fence) != VK_SUCCESS) return VK_NULL_HANDLE;
    return fence;
}

// Call only when no queue operation can still signal `fence` (its wait succeeded, or it was never
// submitted).
inline void release_render_batch_fence(VkDevice device, VkFence fence) {
    if (!fence) return;
    if (render_sync_object_reuse_enabled()) {
        RenderSyncObjectCache& cache = render_sync_object_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.fences.size() < kRenderSyncObjectCacheLimit &&
            vkResetFences(device, 1, &fence) == VK_SUCCESS) {
            cache.fences.emplace_back(device, fence);
            return;
        }
    }
    vkDestroyFence(device, fence, nullptr);
}

// A two-entry TIMESTAMP query pool. The caller must record vkCmdResetQueryPool before using it.
inline VkQueryPool acquire_render_timestamp_pool(VkDevice device) {
    if (render_sync_object_reuse_enabled()) {
        RenderSyncObjectCache& cache = render_sync_object_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        for (size_t i = cache.timestamp_pools.size(); i-- > 0;) {
            if (cache.timestamp_pools[i].first != device) continue;
            const VkQueryPool pool = cache.timestamp_pools[i].second;
            cache.timestamp_pools[i] = cache.timestamp_pools.back();
            cache.timestamp_pools.pop_back();
            ++cache.pool_hits;
            return pool;
        }
        ++cache.pool_misses;
    }
    VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = 2;
    VkQueryPool pool = VK_NULL_HANDLE;
    if (vkCreateQueryPool(device, &info, nullptr, &pool) != VK_SUCCESS) return VK_NULL_HANDLE;
    return pool;
}

// Call only when no pending command buffer references `pool`.
inline void release_render_timestamp_pool(VkDevice device, VkQueryPool pool) {
    if (!pool) return;
    if (render_sync_object_reuse_enabled()) {
        RenderSyncObjectCache& cache = render_sync_object_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.timestamp_pools.size() < kRenderSyncObjectCacheLimit) {
            cache.timestamp_pools.emplace_back(device, pool);
            return;
        }
    }
    vkDestroyQueryPool(device, pool, nullptr);
}

} // namespace prosper::test
