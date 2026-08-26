#pragma once

#include <cstddef>
#include <cstdint>

namespace prosper::frontend {

// Decide whether guest bytes are the authoritative source for a sampled-texture decode. Retained
// color and depth targets are already represented by Vulkan images and must bypass CPU decode-cache
// validation; captured host backing and non-texture resources follow their dedicated paths. Supported
// 3D volume inputs, the graphics backend's base-slice 2D-array view, and exact layered cubes are pure
// guest-byte decodes and may be retained. Non-BC cubes remain excluded by default: broad Float16 cube
// retention caused cache/write-watch churn on Plucky Squire. A caller may explicitly admit a narrow
// uncompressed cube class only after proving that its decoder reads an exact six-face source range.
constexpr bool texture_decode_cache_candidate(bool has_live_color_target,
                                               bool has_live_depth_target,
                                               bool has_captured_host_data,
                                               uint32_t image_dimension,
                                               bool is_sampled_texture,
                                               bool format_supported,
                                               bool block_compressed,
                                               bool exact_uncompressed_cube = false) {
    return !has_live_color_target && !has_live_depth_target &&
        !has_captured_host_data &&
        (image_dimension == 1u || image_dimension == 2u || image_dimension == 5u ||
         (image_dimension == 3u && (block_compressed || exact_uncompressed_cube))) &&
        is_sampled_texture && format_supported;
}

// Convert a layered cube's descriptor footprint into the exact contiguous validation span.
// The footprint covers the selected level in all six independently-strided face chains, including
// alignment gaps. Zero and overflowing ranges fail closed so cache reuse can never be proved against
// less memory than the layer-aware decoder reads.
constexpr size_t layered_cube_source_size(bool exact_layered_cube,
                                          uint64_t gpu_address,
                                          uint64_t descriptor_footprint) {
    if (!exact_layered_cube || !descriptor_footprint ||
        descriptor_footprint > SIZE_MAX ||
        gpu_address > UINT64_MAX - descriptor_footprint)
        return 0;
    return static_cast<size_t>(descriptor_footprint);
}

constexpr size_t block_compressed_cube_source_size(bool block_compressed_cube,
                                                    uint64_t gpu_address,
                                                    uint64_t descriptor_footprint) {
    return layered_cube_source_size(
        block_compressed_cube, gpu_address, descriptor_footprint);
}

} // namespace prosper::frontend
