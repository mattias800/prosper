// compute_buffer_cache_key.hpp -- identity of one resident compute buffer in the backend's cache.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

#include "shared/live/live_compute.hpp"

namespace prosper::frontend {

struct ComputeBufferCacheKey {
    uint64_t gpu_addr = 0;
    uintptr_t host_data = 0;
    uint32_t bytes = 0;
    ComputeBufferMaterializationDiscriminator materialization;
    bool operator==(const ComputeBufferCacheKey& other) const {
        return gpu_addr == other.gpu_addr && host_data == other.host_data &&
               bytes == other.bytes && materialization == other.materialization;
    }
};

struct ComputeBufferCacheKeyHash {
    size_t operator()(const ComputeBufferCacheKey& key) const {
        size_t result = std::hash<uint64_t>{}(key.gpu_addr);
        result ^= std::hash<uintptr_t>{}(key.host_data) << 1;
        result ^= std::hash<uint32_t>{}(key.bytes) << 2;
        result ^= std::hash<uint64_t>{}(key.materialization.logical_bytes) << 3;
        result ^= std::hash<uint64_t>{}(key.materialization.binding_bytes) << 4;
        result ^= std::hash<uint32_t>{}(
            static_cast<uint32_t>(key.materialization.semantic)) << 5;
        return result;
    }
};

}  // namespace prosper::frontend
