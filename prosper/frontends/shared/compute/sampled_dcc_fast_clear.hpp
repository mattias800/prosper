// sampled_dcc_fast_clear.hpp -- materialize a sampled DCC fast-clear surface from its metadata.
//
// A DCC-compressed T# whose control plane holds one uniform fast-clear code has no meaningful base
// bytes: the plane IS the image. This returns that clear colour as the RGBA8 upload pixel, or false
// when the descriptor is not one of the shapes the exact decode covers. Moved verbatim out of
// live_compute.cpp, which is at its file-size cap.
#pragma once

#include <cstddef>
#include <cstdint>

#include "gpu/capture/gpu_capture.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/tile.hpp"

namespace prosper::frontend {

inline bool compute_sampled_dcc_fast_clear_rgba8(
    const prosper::gpu::ShaderResource& resource,
    bool ordinary_guest_backed_sampled_view,
    bool arrayed_sampled_view,
    bool disabled,
    uint8_t* rgba,
    size_t texel_count,
    const uint8_t* metadata,
    size_t metadata_bytes,
    uint8_t* clear_code) {
    const uint32_t components = resource.num_components ? resource.num_components : 1u;
    if (disabled || !ordinary_guest_backed_sampled_view || arrayed_sampled_view ||
        resource.cls != prosper::gpu::ResourceClass::Texture ||
        resource.format != prosper::gpu::DataFormat::Float16 || components != 4u ||
        resource.img_dim != 1u || resource.depth != 1u ||
        resource.declared_mip_levels != 1u || resource.in_mip_tail ||
        resource.layer_stride_bytes || resource.layer_mip_offset_bytes ||
        resource.srgb || resource.depth_compare || !resource.compression_enabled ||
        !resource.metadata_addr || !rgba || !texel_count)
        return false;
    const uint64_t expected_metadata = prosper::gpu::gpu_capture_dcc_metadata_footprint(resource);
    if (!expected_metadata || expected_metadata > SIZE_MAX ||
        metadata_bytes != static_cast<size_t>(expected_metadata))
        return false;
    return prosper::gpu::gfx10_dcc_fast_clear_rgba8(
        rgba, texel_count, metadata, metadata_bytes, components,
        resource.alpha_is_on_msb, clear_code);
}

} // namespace prosper::frontend
