// Renderer-owned mapped staging allocations shared by texture and compute uploads.
#pragma once
#include "diagnostics/env_numeric.hpp"
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

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
// `owned` is the load-bearing part. The cache must be the ONLY code that can free a block it
// created: a caller that releases the same staging twice would otherwise fall through to the
// generic teardown, vkFreeMemory the allocation this cache still maps, and leave a dangling
// mapped pointer for the next acquire to memcpy into.
struct MappedStagingBlock {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDevice device = VK_NULL_HANDLE;   // #3405: a cached handle is only valid for ITS device
};

struct MappedStagingCache {
    std::mutex mutex;
    // Keyed by device as well as shape: VkBuffer/VkDeviceMemory belong to the VkDevice that
    // created them, and a process that builds more than one device would otherwise be handed a
    // block from a destroyed one.
    std::map<std::tuple<VkDevice, VkDeviceSize, uint32_t>,
             std::vector<MappedStagingBlock>> free_blocks;
    std::unordered_map<VkDeviceMemory, std::tuple<VkDevice, VkDeviceSize, uint32_t>> in_use;
    std::unordered_map<VkDeviceMemory, std::tuple<VkDevice, VkDeviceSize, uint32_t>> owned;
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

template <typename PickMemoryType>
inline MappedStagingBlock acquire_mapped_staging(VkDevice device, VkDeviceSize bytes,
                                                 VkBufferUsageFlags usage,
                                                 PickMemoryType&& pick, bool* reused = nullptr) {
    if (reused) *reused = false;
    if (!bytes) return {};
    MappedStagingCache& cache = mapped_staging_cache();
    const std::tuple<VkDevice, VkDeviceSize, uint32_t> key{device, bytes,
                                                           static_cast<uint32_t>(usage)};
    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        auto found = cache.free_blocks.find(key);
        if (found != cache.free_blocks.end() && !found->second.empty()) {
            MappedStagingBlock block = found->second.back();
            found->second.pop_back();
            if (found->second.empty()) cache.free_blocks.erase(found);
            cache.cached_bytes -= bytes;
            cache.in_use.emplace(block.memory, key);
            if (reused) *reused = true;
            return block;                  // mapping intact: no fault-in on the next write
        }
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
    if (vkAllocateMemory(device, &allocation, nullptr, &block.memory) != VK_SUCCESS) {
        vkDestroyBuffer(device, block.buffer, nullptr);
        return {};
    }
    if (vkBindBufferMemory(device, block.buffer, block.memory, 0) != VK_SUCCESS) {
        vkDestroyBuffer(device, block.buffer, nullptr);
        vkFreeMemory(device, block.memory, nullptr);
        return {};
    }
    if (vkMapMemory(device, block.memory, 0, VK_WHOLE_SIZE, 0, &block.mapped) != VK_SUCCESS ||
        !block.mapped) {
        vkFreeMemory(device, block.memory, nullptr);
        vkDestroyBuffer(device, block.buffer, nullptr);
        return {};
    }
    block.device = device;
    std::lock_guard<std::mutex> lock(cache.mutex);
    cache.in_use.emplace(block.memory, key);
    cache.owned.emplace(block.memory, key);
    return block;
}

// True means "this cache owns the allocation; the caller must not free it". That answer is given
// for a block already back in the free list too, so a duplicate release is a no-op rather than a
// dangling mapping.
inline bool release_mapped_staging(VkDevice device, VkBuffer buffer, VkDeviceMemory memory,
                                   void* mapped, bool allow_reuse = true) {
    if (!memory) return false;
    MappedStagingCache& cache = mapped_staging_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (!cache.owned.count(memory)) return false;          // not ours: caller tears it down
    auto active = cache.in_use.find(memory);
    if (active == cache.in_use.end()) { ++cache.double_releases; return true; }
    const std::tuple<VkDevice, VkDeviceSize, uint32_t> key = active->second;
    if (std::get<0>(key) != device) {
        // Keep ownership for a later release through the correct device. Returning
        // false would invite the caller to free our allocation through the wrong one.
        ++cache.cross_device_skips;
        return true;
    }
    // Bisection seam: with this set the block is destroyed rather than retained, so acquire is
    // exercised exactly as before but nothing is ever reused. It separates "the new allocation
    // path is wrong" from "reusing a block is wrong".
    if (!allow_reuse || !mapped_staging_reuse_enabled()) {
        cache.owned.erase(memory);
        cache.in_use.erase(memory);
        vkUnmapMemory(device, memory);
        vkFreeMemory(device, memory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        return true;
    }
    const VkDeviceSize key_bytes = std::get<1>(key);
    cache.in_use.erase(active);
    if (cache.cached_bytes + key_bytes > mapped_staging_limit()) {
        cache.owned.erase(memory);
        vkUnmapMemory(device, memory);
        vkFreeMemory(device, memory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        return true;
    }
    cache.free_blocks[key].push_back(MappedStagingBlock{buffer, memory, mapped, device});
    cache.cached_bytes += key_bytes;
    return true;
}

} // namespace prosper::test
