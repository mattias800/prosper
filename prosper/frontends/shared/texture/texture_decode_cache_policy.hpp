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

// The array equivalent, and the same invariant: a LAYERED array's decoder reads every layer, so the
// validation span must reach the last one. Returning a single surface let the persistent cache prove
// reuse against 262144 of 90177536 bytes -- 0.29% -- of a 256-layer world atlas, so every layer above
// the first changed invisibly and a decode taken while the atlas was nearly empty was reused for the
// whole run (#2998). Cubes have always spanned their six faces; arrays were added later without this.
//
// Fails closed exactly as the cube form does. A `surface_bytes` of 0 is the caller's "do not cache"
// signal and must never be widened into a cacheable span; a single layer leaves the surface size
// unchanged; a MISSING stride does not -- the decoder synthesizes one, so the span follows it (see
// below); an overflowing span yields 0 rather than a truncated range, because a span shorter than
// the decoder reads is precisely the defect this exists to prevent.
//
// The span deliberately OVER-covers by the per-layer mip chains (~23 MiB for the atlas above), and
// that should stay. Tightening it means writing a second model of the layout here, which can drift
// from the decoder's -- and two models disagreeing about one layout is exactly how this defect
// happened. An over-approximation sharing the decoder's own stride errs toward re-decoding, which
// is the safe direction; a tighter one that drifts errs toward serving stale pixels.
constexpr size_t layered_array_source_size(size_t surface_bytes,
                                           uint64_t layer_stride_bytes,
                                           uint32_t layers) {
    if (!surface_bytes) return 0;
    if (layers <= 1u) return surface_bytes;
    // Mirror the decoder's own fallback. `face_base` uses
    //     stride = layer_stride_bytes ? layer_stride_bytes : selected_span
    // so a descriptor that declares NO stride is still read as `layers` back-to-back surfaces --
    // and returning one surface here would validate 1/layers of what is read, which is the very
    // defect this function exists to prevent. Reachable: a BC 2D_ARRAY whose tile mode falls
    // outside the modelled set leaves `image_base_level_view` supported with layer_stride == 0.
    // (Scope: `selected_span` is `mip_tail_bytes` when the view sits in the mip tail, which this
    // cannot see. The zero-stride path is reached only with `in_mip_tail` false, where
    // `selected_span == surface_bytes`; a future tail-plus-zero-stride case would need the tail
    // size passed in rather than assumed.)
    const uint64_t stride = layer_stride_bytes ? layer_stride_bytes
                                               : static_cast<uint64_t>(surface_bytes);
    const uint64_t last_layer_start = stride * static_cast<uint64_t>(layers - 1u);
    if (last_layer_start / stride != (layers - 1u)) return 0;
    if (last_layer_start > UINT64_MAX - surface_bytes) return 0;
    const uint64_t span = last_layer_start + surface_bytes;
    if (span > SIZE_MAX) return 0;
    return static_cast<size_t>(span);
}

constexpr size_t block_compressed_cube_source_size(bool block_compressed_cube,
                                                    uint64_t gpu_address,
                                                    uint64_t descriptor_footprint) {
    return layered_cube_source_size(
        block_compressed_cube, gpu_address, descriptor_footprint);
}

// PROSPER_TEXCOMMIT (#3053): the guest-memory byte extent to scan from a sampled texture's SOURCE
// address. Must be a source-side quantity -- `source_footprint_bytes` is the caller's
// `gpu_capture_resource_footprint()` for this exact resource, which already folds in BC block
// bytes, tiling, and the array/cube layer span the same way `layered_array_source_size()` and
// `layered_cube_source_size()` above do -- and must NEVER be the DECODED byte count
// (`tw * th * decoded_bpp * layers`, the size a surface expands to once converted to the backend's
// RGBA8/RGBA16F decode format). For a block-compressed texture the decoded count is 4x+ the source
// (BC7 decodes 1 source byte/texel to 4 RGBA8 bytes), and a layered array multiplies that again by
// the layer count: Tomb Raider I-III Remastered's 512x512 BC7 256-layer world atlas measured a
// decoded count of 268,435,456 against a real source extent of 67,108,864 -- a 4x overshoot that
// both mis-measured "committed" guest memory (it counted whatever happened to follow the real
// texture) and read guest memory past the real allocation. This function's signature deliberately
// has no slot for a decoded byte count, so that value cannot be passed here by mistake.
constexpr size_t texcommit_scan_extent(uint64_t source_footprint_bytes) {
    return source_footprint_bytes > SIZE_MAX
        ? SIZE_MAX : static_cast<size_t>(source_footprint_bytes);
}

} // namespace prosper::frontend
