// avplayer_plane_policy.hpp — which sampled resource is AvPlayer's NV12 chroma plane, and why.
//
// AvPlayer exposes a decoded video frame as two guest-visible linear planes: an R8 luma plane
// followed by an interleaved RG8 U/V plane. The renderer must recognise the second one, because the
// alternative is the legacy narrow coverage path, which broadcasts a narrow surface's FIRST byte to
// every channel. For a chroma plane that makes the shader's V equal its U, and a picture whose luma,
// detail and geometry are all still exactly right collapses onto the single green<->magenta chroma
// axis through grey. Nothing in a draw census, a colour count or a non-black metric can see that,
// and the result reads as a shading bug rather than a plane-format one — so the recognition test
// keeps the clause that decided it, and the live renderer can log it (PROSPER_AVPCHROMA_LOG).
//
// The test is deliberately narrower than "any RG8 texture": several established game paths still
// rely on the historical coverage broadcast for their own two-channel surfaces.
#pragma once

#include "gpu/shader_resources.hpp"
#include "gpu/guest_texture_layout.hpp"
#include "gpu/tile.hpp"

#include <cstdint>
#include <vector>

namespace prosper::frontend {

enum class AvpChromaReason {
    MatchedRegisteredPitch,      // the HLE producer's exact row pitch covers the visible row
    MatchedSiblingLumaPlane,     // an adjacent R8 plane of matching pitch and 2:1 height
    NotNarrowLinearRg8,          // not a linear one-layer 2D 2-component Unorm8 texture at all
    UnrecognisedSwizzle,         // T# DST_SEL is neither (R,G,0,1) nor (X,X,X,Y)
    PitchShorterThanRow,         // the resolved row pitch cannot hold the visible row
    NoAdjacentLumaPlane,         // no sibling luma plane ends at this allocation
};

struct AvpChromaVerdict {
    bool match = false;
    AvpChromaReason reason = AvpChromaReason::NotNarrowLinearRg8;
    uint32_t row_bytes = 0;
    uint32_t registered_pitch = 0;
    uint32_t resolved_pitch = 0;
    uint64_t sibling_luma_addr = 0;
};

inline const char* avp_chroma_reason_name(AvpChromaReason reason) {
    switch (reason) {
        case AvpChromaReason::MatchedRegisteredPitch:  return "matched-registered-pitch";
        case AvpChromaReason::MatchedSiblingLumaPlane: return "matched-sibling-luma-plane";
        case AvpChromaReason::NotNarrowLinearRg8:      return "not-narrow-linear-rg8";
        case AvpChromaReason::UnrecognisedSwizzle:     return "unrecognised-swizzle";
        case AvpChromaReason::PitchShorterThanRow:     return "pitch-shorter-than-row";
        case AvpChromaReason::NoAdjacentLumaPlane:     return "no-adjacent-luma-plane";
    }
    return "unknown";
}

// An AvPlayer plane is an ordinary linear 2D surface, but a title may DECLARE it with DIM=2D_ARRAY
// and a single layer. That descriptor is byte-identical to a 2D image — the contract already stated
// by shader_resource_uses_ordinary_2d_image — and the rest of the sampled-texture path already
// treats it as one: the padded-linear row read, the sampled-2D source address and the created
// VK_IMAGE_VIEW_TYPE_2D view all admit img_dim 5. R-Type Delta (PPSA26414) declares BOTH NV12
// planes that way, so requiring DIM=2D here rejected a real chroma plane and dropped the whole
// opening movie into the coverage broadcast (#2005). A MULTI-layer array still fails: its slices
// are not one contiguous plane, so neither the pitch nor the sibling-adjacency reasoning holds.
//
// `depth == 1` exactly, matching shader_resource_uses_ordinary_2d_image, and NOT `depth <= 1`: the
// descriptor decoder produces `depth = LAST_ARRAY - BASE_ARRAY + 1` for an array type and **zero**
// when LAST_ARRAY < BASE_ARRAY (`agc_shader_layout.cpp`). Zero therefore means an inverted, malformed
// array range rather than a single layer, and admitting it would claim a descriptor prosper cannot
// interpret. Let it fail visibly instead.
constexpr bool avp_plane_is_one_layer_2d(const gpu::ShaderResource& r) {
    return r.img_dim == 1u ||
           (r.img_dim == 5u && r.depth == 1u && r.layer_stride_bytes == 0u &&
            r.layer_mip_offset_bytes == 0u);
}

// `table` is the draw's complete resource table, used only to look for the sibling luma plane when
// no exact HLE pitch provenance is available (a captured/replayed frame, or a title that stages the
// planes itself). Reading the live registry keeps live and replay on the same contract.
inline AvpChromaVerdict classify_avplayer_chroma_plane(
    const gpu::ShaderResource& r, uint32_t tw, uint32_t th,
    const std::vector<gpu::ShaderResource>& table) {
    using RC = gpu::ResourceClass;
    AvpChromaVerdict v;
    if (r.cls != RC::Texture || r.format != gpu::DataFormat::Unorm8 ||
        r.num_components != 2 || !avp_plane_is_one_layer_2d(r) || r.tile_mode != 0u ||
        r.compression_enabled || !tw || tw > UINT32_MAX / 2u) {
        v.reason = AvpChromaReason::NotNarrowLinearRg8;
        return v;
    }
    const bool rg01_swizzle =
        r.swizzle[0] == 4 && r.swizzle[1] == 5 && r.swizzle[2] == 0 && r.swizzle[3] == 1;
    const bool xxxy_swizzle =
        r.swizzle[0] == 4 && r.swizzle[1] == 4 && r.swizzle[2] == 4 && r.swizzle[3] == 5;
    if (!rg01_swizzle && !xxxy_swizzle) {
        v.reason = AvpChromaReason::UnrecognisedSwizzle;
        return v;
    }
    const uint32_t row_bytes = tw * 2u;
    v.row_bytes = row_bytes;
    v.registered_pitch = gpu::guest_linear_texture_row_pitch(r.gpu_addr, row_bytes);
    if (v.registered_pitch >= row_bytes) {
        v.match = true;
        v.reason = AvpChromaReason::MatchedRegisteredPitch;
        return v;
    }
    const uint32_t chroma_pitch = r.linear_row_pitch_bytes
        ? r.linear_row_pitch_bytes
        : static_cast<uint32_t>(gpu::linear_sampled_row_pitch(tw, 2u));
    v.resolved_pitch = chroma_pitch;
    if (chroma_pitch < row_bytes) {
        v.reason = AvpChromaReason::PitchShorterThanRow;
        return v;
    }
    for (const auto& luma : table) {
        if (luma.cls != RC::Texture || luma.format != gpu::DataFormat::Unorm8 ||
            luma.num_components != 1 || !avp_plane_is_one_layer_2d(luma) ||
            luma.tile_mode != 0u || luma.compression_enabled ||
            luma.width != row_bytes ||
            (static_cast<uint64_t>(luma.height) + 1u) / 2u != th)
            continue;
        const uint32_t luma_pitch = luma.linear_row_pitch_bytes
            ? luma.linear_row_pitch_bytes
            : static_cast<uint32_t>(gpu::linear_sampled_row_pitch(luma.width, 1u));
        if (luma_pitch != chroma_pitch) continue;
        const uint64_t luma_bytes = static_cast<uint64_t>(luma_pitch) * luma.height;
        if (luma.gpu_addr > UINT64_MAX - luma_bytes) continue;
        const uint64_t luma_end = luma.gpu_addr + luma_bytes;
        // The second allocation is either exactly adjacent or begins at the next 64 KiB boundary.
        if (luma_end == r.gpu_addr ||
            (luma_end <= UINT64_MAX - 0xffffu &&
             ((luma_end + 0xffffu) & ~uint64_t{0xffffu}) == r.gpu_addr)) {
            v.match = true;
            v.reason = AvpChromaReason::MatchedSiblingLumaPlane;
            v.sibling_luma_addr = luma.gpu_addr;
            return v;
        }
    }
    v.reason = AvpChromaReason::NoAdjacentLumaPlane;
    return v;
}

} // namespace prosper::frontend
