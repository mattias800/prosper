// Renderer-owned mapped staging allocations shared by texture and compute uploads.
#pragma once
#include "diagnostics/env_numeric.hpp"
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "gpu/diagnostics/gpu_memory_budget_vk.hpp"  // #3533: count what we hold on each heap

namespace prosper::test {
// #3405: texture staging blocks cached together WITH their CPU mapping.
//
// The render-memory pool already avoids vkAllocateMemory, but every consumer still wrapped each
// use in vkMapMemory/vkUnmapMemory. Dropping the mapping returns the pages, so the next write
// faults all of them back in through amdgpu -- ~30% of the profile (amdgpu_gem_fault,
// ttm_bo_vm_fault_reserved, vmf_insert_pfn_prot, pfnmap_setup_cachemode) while memcpy'ing a
// 33 MB 4K staging texture. Keeping the mapping alive measured +12% guest flips/s.
//
// These blocks are deliberately NOT drawn from the shared pool: a retained mapping on shared
// memory is unsound, because the block returns to the pool and the next consumer issues its own
// vkMapMemory on an already-mapped allocation.
//
// `owned` and the per-acquisition lease are the load-bearing parts. The cache must be the ONLY code that can free a block it
// created: a caller that releases the same staging twice would otherwise fall through to the
// generic teardown, vkFreeMemory the allocation this cache still maps, and leave a dangling
// mapped pointer for the next acquire to memcpy into.
struct MappedStagingBlock {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDevice device = VK_NULL_HANDLE;   // #3405: a cached handle is only valid for ITS device
    uint64_t lease = 0;                // renewed on every acquire, including reuse
    uint64_t idle_order = 0;
};

using MappedStagingKey = std::tuple<VkDevice, VkDeviceSize, uint32_t>;
struct MappedStagingOwner {
    MappedStagingKey key;
    MappedStagingBlock block;
};

struct MappedStagingCache {
    std::mutex mutex;
    // Keyed by device as well as shape: VkBuffer/VkDeviceMemory belong to the VkDevice that
    // created them, and a process that builds more than one device would otherwise be handed a
    // block from a destroyed one.
    std::map<std::tuple<VkDevice, VkDeviceSize, uint32_t>,
             std::vector<MappedStagingBlock>> free_blocks;
    std::unordered_map<uint64_t, MappedStagingKey> in_use;
    std::unordered_map<uint64_t, MappedStagingOwner> owned;
    // Shared sequence for acquisition identities and oldest-idle ordering. Zero
    // means exhausted: never reuse a token, even if Vulkan reuses raw handles.
    uint64_t next_sequence = 1;
    uint64_t stale_releases = 0;
    uint64_t evictions = 0;
    uint64_t cross_device_skips = 0;
    VkDeviceSize cached_bytes = 0;
    uint64_t double_releases = 0;
};

inline MappedStagingCache& mapped_staging_cache() {
    static MappedStagingCache cache;
    return cache;
}

// Reuse is off when EITHER switch is set. PROSPER_NO_MEMORY_POOL has to keep working here: before
// this cache existed, staging came from allocate_transient_render_memory, so that flag disabled
// staging pooling too. Several `## Ruled out` rows in docs/GRAPHICS.md exonerate "prosper's caches,
// pools or arenas" using exactly that arm; if this cache ignored it, those arms would silently test
// less than they did when they were recorded.
inline bool mapped_staging_reuse_enabled() {
    static const bool enabled = std::getenv("PROSPER_NO_MAPPED_STAGING") == nullptr &&
                                std::getenv("PROSPER_NO_MEMORY_POOL") == nullptr;
    return enabled;
}

// This cache holds its own allocations and accounts them separately from RenderMemoryPool, so it
// must not silently borrow that pool's budget -- doing so would double peak retained host memory
// with nothing reporting the second half. Its own, smaller, explicitly named budget instead.
inline VkDeviceSize mapped_staging_limit() {
    static const VkDeviceSize limit = []() -> VkDeviceSize {
        const char* value = getenv("PROSPER_MAPPED_STAGING_MB");
        const uint64_t mib = prosper::diag::env_u64_or_default_capped(
            "PROSPER_MAPPED_STAGING_MB", value, 256ull,
            UINT64_MAX / (1024ull * 1024ull), "MiB");
        return static_cast<VkDeviceSize>(mib) * 1024ull * 1024ull;
    }();
    return limit;
}

// Called with cache.mutex held. Opt-in diagnosis; the counts describe the state at
// this decision, not a claim that an acquisition miss was caused by a budget limit.
// cached_bytes counts idle requested bytes, excluding live blocks and allocation padding.
inline void trace_mapped_staging_event(const MappedStagingCache& cache, const char* event,
        const std::tuple<VkDevice, VkDeviceSize, uint32_t>& key, VkDeviceMemory memory) {
    static const bool enabled = std::getenv("PROSPER_MAPPED_STAGING_LOG") != nullptr;
    if (!enabled) return;
    size_t live_same = 0;
    for (const auto& entry : cache.in_use) live_same += entry.second == key;
    const auto free = cache.free_blocks.find(key);
    const size_t free_same = free == cache.free_blocks.end() ? 0 : free->second.size();
    std::fprintf(stderr, "[mapped-staging] event=%s device=%p memory=0x%llx "
        "bytes=%llu usage=0x%x cached_bytes=%llu limit_bytes=%llu "
        "free_same=%zu live_same=%zu free_keys=%zu\n", event,
        static_cast<void*>(std::get<0>(key)),
        (unsigned long long)memory,
        static_cast<unsigned long long>(std::get<1>(key)), std::get<2>(key),
        static_cast<unsigned long long>(cache.cached_bytes),
        static_cast<unsigned long long>(mapped_staging_limit()),
        free_same, live_same, cache.free_blocks.size());
}

template <typename PickMemoryType>
inline MappedStagingBlock acquire_mapped_staging(VkDevice device, VkDeviceSize bytes,
                                                 VkBufferUsageFlags usage,
                                                 PickMemoryType&& pick, bool* reused = nullptr) {
    if (reused) *reused = false;
    if (!bytes) return {};
    MappedStagingCache& cache = mapped_staging_cache();
    const std::tuple<VkDevice, VkDeviceSize, uint32_t> key{device, bytes,
                                                           static_cast<uint32_t>(usage)};
    uint64_t lease;
    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (!cache.next_sequence) return {};
        lease = cache.next_sequence++;
        auto found = cache.free_blocks.find(key);
        if (found != cache.free_blocks.end() && !found->second.empty()) {
            MappedStagingBlock block = found->second.back();
            found->second.pop_back();
            if (found->second.empty()) cache.free_blocks.erase(found);
            cache.cached_bytes -= bytes;
            cache.owned.erase(block.lease);
            block.lease = lease;
            cache.owned.emplace(lease, MappedStagingOwner{key, block});
            cache.in_use.emplace(lease, key);
            if (reused) *reused = true;
            trace_mapped_staging_event(cache, "acquire-hit", key, block.memory);
            return block;                  // mapping intact: no fault-in on the next write
        }
        trace_mapped_staging_event(cache, "acquire-miss", key, VK_NULL_HANDLE);
    }

    MappedStagingBlock block;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = usage;
    if (vkCreateBuffer(device, &bci, nullptr, &block.buffer) != VK_SUCCESS) return {};
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, block.buffer, &requirements);
    const uint32_t type = pick(requirements.memoryTypeBits);
    if (type == UINT32_MAX) { vkDestroyBuffer(device, block.buffer, nullptr); return {}; }
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = type;
    if (prosper::gpu::allocate_device_memory(device, &allocation, &block.memory) != VK_SUCCESS) {
        vkDestroyBuffer(device, block.buffer, nullptr);
        return {};
    }
    if (vkBindBufferMemory(device, block.buffer, block.memory, 0) != VK_SUCCESS) {
        vkDestroyBuffer(device, block.buffer, nullptr);
        prosper::gpu::free_device_memory(device, block.memory);
        return {};
    }
    if (vkMapMemory(device, block.memory, 0, VK_WHOLE_SIZE, 0, &block.mapped) != VK_SUCCESS ||
        !block.mapped) {
        prosper::gpu::free_device_memory(device, block.memory);
        vkDestroyBuffer(device, block.buffer, nullptr);
        return {};
    }
    block.device = device;
    block.lease = lease;
    std::lock_guard<std::mutex> lock(cache.mutex);
    cache.in_use.emplace(lease, key);
    cache.owned.emplace(lease, MappedStagingOwner{key, block});
    trace_mapped_staging_event(cache, "acquire-new", key, block.memory);
    return block;
}

// Only same-device idle blocks may be destroyed: their last submission owner
// already released them, and the caller establishes that this device is live.
// A foreign raw VkDevice in the cache does not establish its lifetime.
// The caller holds cache.mutex through selection, accounting and destruction.
inline bool make_mapped_staging_room(MappedStagingCache& cache, VkDevice device,
                                     VkDeviceSize bytes) {
    const VkDeviceSize limit = mapped_staging_limit();
    if (bytes > limit) return false;
    if (cache.cached_bytes <= limit - bytes) return true;
    static const bool evict = std::getenv("PROSPER_NO_MAPPED_STAGING_EVICTION") == nullptr;
    if (!evict) return false;
    VkDeviceSize reclaimable = 0;
    for (const auto& [key, blocks] : cache.free_blocks)
        if (std::get<0>(key) == device) reclaimable += std::get<1>(key) * blocks.size();
    // Do not discard useful same-device entries if foreign occupancy still makes
    // the incoming block impossible to retain.
    if (cache.cached_bytes - reclaimable > limit - bytes) return false;
    while (cache.cached_bytes > limit - bytes) {
        auto victim = cache.free_blocks.end();
        for (auto it = cache.free_blocks.begin(); it != cache.free_blocks.end(); ++it)
            if (std::get<0>(it->first) == device && !it->second.empty() &&
                (victim == cache.free_blocks.end() ||
                 it->second.front().idle_order < victim->second.front().idle_order)) victim = it;
        const auto key = victim->first;
        const auto block = victim->second.front();
        victim->second.erase(victim->second.begin());
        if (victim->second.empty()) cache.free_blocks.erase(victim);
        cache.cached_bytes -= std::get<1>(key);
        cache.owned.erase(block.lease);
        ++cache.evictions;
        trace_mapped_staging_event(cache, "evict-idle", key, block.memory);
        vkUnmapMemory(device, block.memory);
        vkDestroyBuffer(device, block.buffer, nullptr);
        prosper::gpu::free_device_memory(device, block.memory);
    }
    return true;
}

// True means the caller must not free this allocation. A nonzero lease proves
// cache provenance even after eviction. Old leases cannot free a newer upload
// when the cache or Vulkan has reused the same buffer/memory handles.
inline bool release_mapped_staging(VkDevice device, VkBuffer buffer, VkDeviceMemory memory,
                                   void* mapped, uint64_t lease, bool allow_reuse = true) {
    if (!memory) return false;
    MappedStagingCache& cache = mapped_staging_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (!lease) {
        // Missing provenance must not invite generic teardown of known cache
        // memory. This path is rare; successful acquisitions always carry a lease.
        for (const auto& [id, owner] : cache.owned)
            if (owner.block.memory == memory) return true;
        return false;
    }
    const auto owner = cache.owned.find(lease);
    if (owner == cache.owned.end()) {
        ++cache.stale_releases;
        return true;
    }
    const auto& block = owner->second.block;
    const auto key = owner->second.key;
    if (block.device != device) {
        ++cache.cross_device_skips;
        trace_mapped_staging_event(cache, "wrong-device-release", key, memory);
        return true;
    }
    if (block.memory != memory || block.buffer != buffer || block.mapped != mapped) return true;
    const auto active = cache.in_use.find(lease);
    if (active == cache.in_use.end()) {
        ++cache.double_releases;
        trace_mapped_staging_event(cache, "duplicate-release", key, memory);
        return true;
    }
    cache.in_use.erase(active);
    const char* discard = nullptr;
    if (!allow_reuse) discard = "discard-caller-disabled";
    else if (!mapped_staging_reuse_enabled()) discard = "discard-pool-disabled";
    else if (!cache.next_sequence) discard = "discard-sequence-exhausted";
    else if (!make_mapped_staging_room(cache, device, std::get<1>(key))) discard = "discard-budget";
    if (discard) {
        cache.owned.erase(owner);
        trace_mapped_staging_event(cache, discard, key, memory);
        vkUnmapMemory(device, memory);
        vkDestroyBuffer(device, buffer, nullptr);
        prosper::gpu::free_device_memory(device, memory);
        return true;
    }
    auto idle = block;
    idle.idle_order = cache.next_sequence++;
    cache.free_blocks[key].push_back(idle);
    cache.cached_bytes += std::get<1>(key);
    trace_mapped_staging_event(cache, "retain", key, memory);
    return true;
}

} // namespace prosper::test
