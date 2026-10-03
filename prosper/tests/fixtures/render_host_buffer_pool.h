// Same-TU companion included at the unchanged declaration site inside prosper::test.
#pragma once

// Storage-buffer contents are rewritten for every synchronous render call, but their Vulkan object
// shapes repeat heavily. Keep capacity-class host-coherent buffers mapped between calls so the hot path
// only copies bytes. The backend normally packs call-local logical uploads into aligned slices of a few
// pooled arenas; the same pool also backs the per-upload fallback. A call or explicit submission batch
// completes before returning buffers, so no in-flight GPU work can observe a later upload. Descriptors
// retain exact logical offsets and ranges, so capacity padding and neighboring arena slices remain
// shader-inaccessible.
struct RenderHostBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize bytes = 0;
    VkDeviceSize allocation_bytes = 0;
    // Release order, stamped when the buffer enters the cache. Only meaningful for cached entries;
    // it is what makes eviction least-recently-used rather than arbitrary (#1284).
    uint64_t last_use = 0;
};

struct RenderHostBufferPool {
    // A deque per capacity class, ordered oldest-release at the front. Acquire takes the BACK (the
    // most recently released buffer of that class, so the hottest pages come back first) and
    // eviction takes the FRONT (the least recently released). A vector cannot do both in O(1).
    std::unordered_map<VkDeviceSize, std::deque<RenderHostBuffer>> available;
    VkDeviceSize cached_bytes = 0;
    size_t cached_buffers = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    // Monotonic release counter; see RenderHostBuffer::last_use.
    uint64_t release_clock = 0;
};

struct RenderHostBufferPoolStats {
    VkDeviceSize cached_bytes = 0;
    size_t cached_buffers = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
};

inline RenderHostBufferPool& render_host_buffer_pool() {
    static thread_local RenderHostBufferPool pool;
    return pool;
}

inline bool render_host_buffer_pool_enabled() {
    return getenv("PROSPER_NO_BACKEND_BUFFER_POOL") == nullptr;
}

// Which capacity class holds the least-recently-released cached buffer.
//
// Eviction used to take `pool.available.begin()` — an arbitrary `unordered_map` bucket — so under
// pressure the pool discarded whichever class the hash happened to order first, which is very often
// the class about to be needed again. That is the failure mode that survives any budget smaller than
// the working set, so it is fixed independently of the budget (#1284).
//
// Each deque is ordered oldest-release at the front, so only the fronts can be the global oldest and
// the scan is over the number of capacity classes (~20-30 power-of-two sizes), not cached entries.
// Pure over pool state so the policy is unit-testable without a Vulkan device.
inline bool render_host_buffer_pool_lru_key(const RenderHostBufferPool& pool,
                                            VkDeviceSize& key_out) {
    bool found = false;
    uint64_t oldest = 0;
    for (const auto& [capacity, entries] : pool.available) {
        if (entries.empty()) continue;
        const uint64_t stamp = entries.front().last_use;
        if (!found || stamp < oldest) {
            found = true;
            oldest = stamp;
            key_out = capacity;
        }
    }
    return found;
}

// Host physical memory, for the memory-aware pool budget below. Duplicated rather than shared with
// the frontend's identical helper because this header is included BY the frontend, so taking the
// dependency the other way would invert the include order.
inline uint64_t render_host_physical_memory_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? status.ullTotalPhys : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages <= 0 || page_size <= 0) return 0;
    return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
#endif
}

// Budget for retained host-visible staging buffers.
//
// This was a flat 256 MiB, which is not a cache for a 3D title: Blue Prince's per-submit staging
// working set measures 974 MiB across 503 buffers, so the pool ran permanently at its ceiling with
// evictions EXACTLY equal to misses (~216k of each) — one buffer destroyed for every one created.
// Raising it to 2 GiB on that title took the backend submit from 203.06 to 125.98 ms, -34.8 %
// normalised per draw, and dropped evictions to zero (#1284).
//
// Sized as a fraction of host RAM rather than a bigger constant, mirroring
// `texture_decode_cache_limit_bytes`. The floor is the historical 256 MiB, so no host is given LESS
// than before; the ceiling bounds the worst case. An explicit `PROSPER_BACKEND_BUFFER_POOL_MB` wins
// outright, including values below the floor, because it is also the A/B lever and a constrained-host
// escape hatch. Pure and separated from `getenv` so it can be unit-tested across host sizes.
inline VkDeviceSize render_host_buffer_pool_limit_bytes(const char* override_mib,
                                                        uint64_t physical_memory_bytes) {
    constexpr uint64_t kMiB = 1024ull * 1024ull;
    constexpr uint64_t kMinBytes = 256ull * kMiB;
    constexpr uint64_t kMaxBytes = 2048ull * kMiB;
    if (override_mib) {
        const uint64_t mib = strtoull(override_mib, nullptr, 10);
        if (mib > UINT64_MAX / kMiB) return VkDeviceSize{UINT64_MAX};
        return static_cast<VkDeviceSize>(mib * kMiB);
    }
    if (!physical_memory_bytes) return static_cast<VkDeviceSize>(kMinBytes);
    uint64_t bytes = std::clamp(physical_memory_bytes / 8u, kMinBytes, kMaxBytes);
    bytes -= bytes % kMiB;
    return static_cast<VkDeviceSize>(bytes);
}

inline VkDeviceSize render_host_buffer_pool_limit() {
    static const VkDeviceSize limit = []() -> VkDeviceSize {
        const uint64_t physical = render_host_physical_memory_bytes();
        const VkDeviceSize bytes =
            render_host_buffer_pool_limit_bytes(getenv("PROSPER_BACKEND_BUFFER_POOL_MB"), physical);
        fprintf(stderr,
                "[render] backend host-buffer pool budget = %.1f MiB (host physical %.1f GiB)\n",
                bytes / (1024.0 * 1024.0), physical / (1024.0 * 1024.0 * 1024.0));
        return bytes;
    }();
    return limit;
}

inline VkDeviceSize render_host_buffer_arena_size() {
    static const VkDeviceSize bytes = []() -> VkDeviceSize {
        // The max(4, ...) floor means a typo does not crash -- it silently builds a FOUR-BYTE
        // arena, which is the worst kind of wrong setting: plausible, survivable, and slow (#3267).
        const char* value = getenv("PROSPER_BACKEND_BUFFER_ARENA_KB");
        const uint64_t kib = prosper::diag::env_u64_or_default_capped(
            "PROSPER_BACKEND_BUFFER_ARENA_KB", value, 1024ull, UINT64_MAX / 1024ull, "KiB");
        return std::max<VkDeviceSize>(4, static_cast<VkDeviceSize>(kib) * 1024ull);
    }();
    return bytes;
}

inline void destroy_render_host_buffer(VkDevice device, RenderHostBuffer& buffer) {
    if (buffer.mapped) vkUnmapMemory(device, buffer.memory);
    if (buffer.buffer) vkDestroyBuffer(device, buffer.buffer, nullptr);
    if (buffer.memory) prosper::gpu::free_device_memory(device, buffer.memory);
    buffer = {};
}

inline VkDeviceSize render_host_buffer_capacity(VkDeviceSize bytes) {
    VkDeviceSize capacity = 4;
    while (capacity < bytes && capacity <= UINT64_MAX / 2) capacity *= 2;
    return capacity < bytes ? bytes : capacity;
}

inline RenderHostBuffer acquire_render_host_buffer(const RenderVkCtx& ctx, VkDeviceSize bytes) {
    if (!bytes) return {};
    const VkDeviceSize capacity = render_host_buffer_capacity(bytes);
    RenderHostBufferPool& pool = render_host_buffer_pool();
    auto found = pool.available.find(capacity);
    if (found != pool.available.end() && !found->second.empty()) {
        RenderHostBuffer buffer = found->second.back();
        found->second.pop_back();
        if (found->second.empty()) pool.available.erase(found);
        pool.cached_bytes -= buffer.allocation_bytes;
        --pool.cached_buffers;
        ++pool.hits;
        return buffer;
    }
    ++pool.misses;

    RenderHostBuffer buffer;
    buffer.bytes = capacity;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = capacity;
    // INDEX as well as STORAGE, so one pool serves both. Index data used to take a dedicated
    // vkCreateBuffer + vkAllocateMemory per indexed draw -- 55% of setup_fixed_ms and ~9% of a Blue
    // Prince gameplay frame (#2253, measured by #2252's partition).
    //
    // Widening the usage does not change how existing storage users BIND or WRITE these buffers; it
    // only makes the same memory legal for vkCmdBindIndexBuffer. It is not entirely free of the
    // storage path, though, and the honest statement is about ALLOCATION rather than binding: an
    // added usage bit can only narrow memoryRequirements.memoryTypeBits, so on a hypothetical device
    // with no HOST_VISIBLE type accepting INDEX usage the arena would fail to allocate and the
    // STORAGE path would fall back with it. That is a performance regression rather than a
    // correctness one -- the dedicated-buffer fallback covers both -- and no such device is known.
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    // TRANSFER_SRC only when PROSPER_BUFFER_ECHO is armed. The echo reads these slices back with
    // vkCmdCopyBuffer, and VUID-vkCmdCopyBuffer-srcBuffer-00118 requires the source to carry the
    // bit -- without it the diagnostic is itself invalid Vulkan, which validation reports 10 times
    // on one replay and which no amount of plausible-looking output would have revealed. Gated
    // rather than unconditional for the reason the comment above gives about INDEX: an added usage
    // bit can only narrow memoryRequirements.memoryTypeBits, and the default allocation path must
    // stay exactly as it is when the diagnostic is off.
    static const bool echo_usage = getenv("PROSPER_BUFFER_ECHO") != nullptr;
    if (echo_usage) info.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (vkCreateBuffer(ctx.dev, &info, nullptr, &buffer.buffer) != VK_SUCCESS) return {};
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(ctx.dev, buffer.buffer, &requirements);
    buffer.allocation_bytes = requirements.size;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = render_memory_type(ctx.phys, requirements.memoryTypeBits,
                                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (allocation.memoryTypeIndex == UINT32_MAX ||
        prosper::gpu::allocate_device_memory(ctx.dev, &allocation, &buffer.memory) != VK_SUCCESS ||
        vkBindBufferMemory(ctx.dev, buffer.buffer, buffer.memory, 0) != VK_SUCCESS ||
        vkMapMemory(ctx.dev, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped) != VK_SUCCESS) {
        destroy_render_host_buffer(ctx.dev, buffer);
        return {};
    }
    return buffer;
}

inline void release_render_host_buffer(VkDevice device, RenderHostBuffer buffer) {
    if (!buffer.buffer || !buffer.memory || !buffer.mapped) {
        destroy_render_host_buffer(device, buffer);
        return;
    }
    constexpr size_t max_cached_buffers = 4096;
    const VkDeviceSize limit = render_host_buffer_pool_limit();
    if (!limit || buffer.allocation_bytes > limit) {
        destroy_render_host_buffer(device, buffer);
        return;
    }

    std::vector<RenderHostBuffer> evicted;
    RenderHostBufferPool& pool = render_host_buffer_pool();
    while ((pool.cached_buffers >= max_cached_buffers ||
            pool.cached_bytes > limit - buffer.allocation_bytes) &&
           !pool.available.empty()) {
        VkDeviceSize victim_key = 0;
        if (!render_host_buffer_pool_lru_key(pool, victim_key)) break;
        auto victim = pool.available.find(victim_key);
        if (victim == pool.available.end() || victim->second.empty()) break;
        RenderHostBuffer old = victim->second.front();
        victim->second.pop_front();
        if (victim->second.empty()) pool.available.erase(victim);
        pool.cached_bytes -= old.allocation_bytes;
        --pool.cached_buffers;
        ++pool.evictions;
        evicted.push_back(old);
    }
    if (pool.cached_buffers < max_cached_buffers &&
        pool.cached_bytes <= limit - buffer.allocation_bytes) {
        buffer.last_use = ++pool.release_clock;
        pool.available[buffer.bytes].push_back(buffer);
        pool.cached_bytes += buffer.allocation_bytes;
        ++pool.cached_buffers;
        buffer = {};
    }
    for (RenderHostBuffer& old : evicted) destroy_render_host_buffer(device, old);
    if (buffer.buffer) destroy_render_host_buffer(device, buffer);
}
