#pragma once
// Image IDENTITY: do two bindings describe the same image, such that they must alias one Vulkan
// image rather than becoming two? (#3204 extraction of the predicates that decided #3205.)
//
// This is not a cache optimisation. Callers DEPEND on the aliasing: GTA V's 4K output shader writes
// the four 8x8 quadrants of each 16x16 block through four bindings at a single guest address, so a
// binding that fails to alias drops three of four stores and renders a checkerboard over the frame.
// #3205 is exactly that failure -- a field added to the comparison that described nothing for
// single-level images and was populated on only one of two construction paths.
//
// Extracted from a ~5,000-line function where these were unnamed local expressions, so that the
// contract can be stated, tested, and pointed at in review. Every function here is a pure
// comparison of two descriptors: no allocation, no lookup, no side effects, `inline` so the hot
// per-binding loop pays the same cost it did as an inline expression.
//
// The BoundImage-derived half of identity (representation, storage flag, texel depth, array layers)
// stays with the caller and arrives through `ComputeImageViewShape`, because BoundImage is a
// frontend-local type and dragging it into the resource layer would invert the dependency.
#include "gpu/resources/shader_resources.hpp"
#include "gpu/agc/agc_shader_layout.hpp"

namespace prosper::gpu {

// The non-descriptor half of image identity, supplied by the caller.
struct ComputeImageViewShape {
    bool storage = false;        // a storage image and a sampled image are never the same image
    uint32_t texel_depth = 0;    // realized depth, which may differ from the descriptor's
    uint32_t array_layers = 0;   // realized layer count
};

// DCC/metadata identity. Two descriptors that compress differently are different images even when
// every extent matches, because the compressed footprint is part of the allocation.
inline bool shader_resource_same_dcc_identity(const ShaderResource& a, const ShaderResource& b) {
    return a.max_uncompressed_block_size == b.max_uncompressed_block_size &&
           a.max_compressed_block_size == b.max_compressed_block_size &&
           a.meta_pipe_aligned == b.meta_pipe_aligned &&
           a.write_compress_enabled == b.write_compress_enabled &&
           a.compression_enabled == b.compression_enabled &&
           a.alpha_is_on_msb == b.alpha_is_on_msb &&
           a.color_transform == b.color_transform &&
           a.metadata_addr == b.metadata_addr &&
           a.dcc_metadata_size == b.dcc_metadata_size &&
           a.dcc_metadata_host_data == b.dcc_metadata_host_data &&
           a.dcc_metadata_host_data_size == b.dcc_metadata_host_data_size;
}

// Same guest bytes behind both descriptors.
//
// Pointer equality is NOT sufficient as the only test: a capture materializes each descriptor's
// bytes into an independently-owned blob, so two descriptors for one guest range get two pointers.
// The address+size arm is what keeps those aliasing -- it is the arm GTA V's quadrant writes need.
inline bool shader_resource_same_host_backing(const ShaderResource& a, const ShaderResource& b) {
    return a.host_data == b.host_data ||
           (a.gpu_addr != 0 && a.gpu_addr == b.gpu_addr && a.host_data_size == b.host_data_size);
}

// Full view identity. `same_backing_representation` is the caller's BoundImage-derived half, passed
// in rather than recomputed here.
inline bool shader_resource_same_view(const ShaderResource& a, const ShaderResource& b,
                                      const ComputeImageViewShape& sa,
                                      const ComputeImageViewShape& sb,
                                      bool same_backing_representation) {
    return same_backing_representation &&
           sa.storage == sb.storage &&
           a.gpu_addr == b.gpu_addr && a.size == b.size &&
           a.width == b.width && a.height == b.height && a.depth == b.depth &&
           sa.texel_depth == sb.texel_depth && sa.array_layers == sb.array_layers &&
           a.format == b.format && a.num_components == b.num_components &&
           a.tile_mode == b.tile_mode && a.img_dim == b.img_dim &&
           a.layer_stride_bytes == b.layer_stride_bytes &&
           a.layer_mip_offset_bytes == b.layer_mip_offset_bytes &&
           a.in_mip_tail == b.in_mip_tail &&
           a.mip_tail_x == b.mip_tail_x && a.mip_tail_y == b.mip_tail_y &&
           a.declared_mip_levels == b.declared_mip_levels &&
           // #3205: the derived chain provenance describes nothing for a single-level image, and is
           // populated on only one of two construction paths. See the predicate's own comment.
           shader_resource_mip_chain_provenance_matches(a, b) &&
           a.srgb == b.srgb &&
           shader_resource_same_host_backing(a, b) &&
           a.host_data_size == b.host_data_size &&
           shader_resource_same_dcc_identity(a, b);
}

// Sampler identity. Only meaningful for sampled bindings: a storage image has no sampler, so the
// caller skips this entirely rather than comparing fields that mean nothing.
inline bool shader_resource_same_sampler(const ShaderResource& a, const ShaderResource& b) {
    return a.mag_filter == b.mag_filter && a.min_filter == b.min_filter &&
           a.mip_filter == b.mip_filter &&
           a.addr_uvw[0] == b.addr_uvw[0] && a.addr_uvw[1] == b.addr_uvw[1] &&
           a.addr_uvw[2] == b.addr_uvw[2] &&
           a.border_color_type == b.border_color_type &&
           a.min_lod == b.min_lod && a.max_lod == b.max_lod &&
           a.lod_bias == b.lod_bias && a.max_aniso_ratio == b.max_aniso_ratio &&
           a.depth_compare_func == b.depth_compare_func &&
           a.depth_compare == b.depth_compare && a.unnormalized == b.unnormalized &&
           a.swizzle[0] == b.swizzle[0] && a.swizzle[1] == b.swizzle[1] &&
           a.swizzle[2] == b.swizzle[2] && a.swizzle[3] == b.swizzle[3];
}

}  // namespace prosper::gpu
