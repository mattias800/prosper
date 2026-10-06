// compute_image_cache_key.hpp -- identity of one resident compute image in the backend's cache.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

namespace prosper::frontend {

struct ComputeImageCacheKey {
    uint64_t gpu_addr = 0;
    // Replay/capture resources preserve the architectural address for descriptor identity, but
    // expose their bytes through owned host storage. Keep that storage identity in the key just as
    // the persistent buffer cache does: two loaded captures may reuse a guest address while their
    // owned byte arrays have unrelated lifetimes.
    uintptr_t host_data = 0;
    uint32_t guest_bytes = 0;
    uint32_t resource_bytes = 0;
    uint32_t width = 0, height = 0, depth = 0;
    uint32_t format = 0, components = 0, tile_mode = 0, img_dim = 0;
    uint32_t linear_row_pitch = 0;
    uint32_t layer_stride = 0, layer_mip_offset = 0;
    uint32_t mip_tail_offset = 0, mip_tail_bytes = 0;
    uint32_t mip_tail_x = 0, mip_tail_y = 0;
    uint32_t vk_format = 0;
    bool storage = false;
    bool in_mip_tail = false;
    bool srgb = false;
    bool depth_compare = false;
    // #3048: a cached image is created with exactly this many mip levels. Two T#s over the same
    // allocation can agree on every field above and declare different chain lengths, and handing a
    // one-level image to a binding whose module fetches level three is not a miss but a fault.
    // Appended LAST so `storage_image_cache_key`'s positional aggregate init keeps its meaning.
    uint32_t mip_levels = 1;
    // #657: a VK_IMAGE_VIEW_TYPE_CUBE view is only legal over an image created CUBE_COMPATIBLE, and
    // that flag is fixed at creation. Two T#s over the same allocation can agree on every field
    // above while one needs a cube view and the other does not, so handing the non-compatible image
    // to the cube binding is a FAULT, not a hit -- exactly the #3048 argument for `mip_levels`.
    // Measured: Sonic Frontiers' Cyber Space prefilter writes the cube through storage bindings and
    // a later dispatch samples it as `OpTypeImage Dim=Cube`; the reader found the writer's cached
    // image and its view failed VUID-VkImageViewCreateInfo-image-01003.
    // Appended LAST for the same positional-init reason.
    bool cube_compatible = false;

    bool operator==(const ComputeImageCacheKey& other) const = default;
};

struct ComputeImageCacheKeyHash {
    size_t operator()(const ComputeImageCacheKey& key) const {
        size_t result = std::hash<uint64_t>{}(key.gpu_addr);
        const auto mix = [&](uint64_t value) {
            result ^= std::hash<uint64_t>{}(value) + 0x9e3779b97f4a7c15ull +
                      (result << 6) + (result >> 2);
        };
        mix(key.host_data); mix(key.guest_bytes); mix(key.resource_bytes);
        mix(key.width); mix(key.height); mix(key.depth);
        mix(key.format); mix(key.components); mix(key.tile_mode); mix(key.img_dim);
        mix(key.linear_row_pitch); mix(key.layer_stride); mix(key.layer_mip_offset);
        mix(key.mip_tail_offset); mix(key.mip_tail_bytes);
        mix(key.mip_tail_x); mix(key.mip_tail_y); mix(key.vk_format);
        mix(key.storage); mix(key.in_mip_tail); mix(key.srgb); mix(key.depth_compare);
        mix(key.mip_levels); mix(key.cube_compatible);
        return result;
    }
};

}  // namespace prosper::frontend
