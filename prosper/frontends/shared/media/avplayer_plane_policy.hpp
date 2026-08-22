// avplayer_plane_policy.hpp — which sampled resource is a decoded video frame's NV12 chroma plane.
//
// A decoded video frame reaches the guest as two guest-visible planes: an R8 luma plane and an
// interleaved RG8 U/V plane. AvPlayer stages both itself, linear, in one contiguous buffer; a title
// that drives sceVideodec2 stages them wherever and however it likes -- separate allocations, and
// GPU-tiled rather than linear, are both live-observed. The renderer must recognise the second one
// in every case, because the
// alternative is the legacy narrow coverage path, which broadcasts a narrow surface's FIRST byte to
// every channel. For a chroma plane that makes the shader's V equal its U, and a picture whose luma,
// detail and geometry are all still exactly right collapses onto the single green<->magenta chroma
// axis through grey. Nothing in a draw census, a colour count or a non-black metric can see that,
// and the result reads as a shading bug rather than a plane-format one — so the recognition test
// keeps the clause that decided it, and the live renderer can log it (PROSPER_AVPCHROMA_LOG).
//
// The test is deliberately narrower than "any RG8 texture": several established game paths still
// rely on the historical coverage broadcast for their own two-channel surfaces. What keeps it narrow
// is the PAIR -- a two-channel surface is only claimed when the same draw also binds a
// single-channel partner of the same tile mode, same element format, one-layer 2D, whose width is
// exactly twice its own, whose height is (h+1)/2, and whose bytes do not overlap it. An ordinary
// two-channel game texture has no such partner.
//
// One clause listed here previously that is NOT doing the work, recorded so nobody credits it:
// "at the identical physical pitch". When neither descriptor declares `linear_row_pitch_bytes` --
// which is the case for both live fixtures -- both pitches are derived from the widths by
// `linear_sampled_row_pitch`, and the width relation two lines above has already forced them equal.
// The comparison then cannot fail, so it narrows nothing. It IS a real check when at least one
// descriptor declares a pitch, which is why it stays; it is simply not evidence in the measured
// case, and a file whose whole job is to record why the predicate is narrow must not overstate what
// is narrowing it.
#pragma once

#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/guest_texture_layout.hpp"
#include "gpu/texture/tile.hpp"

#include <cstdint>
#include <vector>

namespace prosper::frontend {

enum class AvpChromaReason {
    MatchedRegisteredPitch,      // the HLE producer's exact row pitch covers the visible row
    MatchedAdjacentLumaPlane,    // a sibling R8 plane of matching pitch and 2:1 geometry ENDS here
    MatchedSeparateLumaPlane,    // the same pair, staged in its own allocation (#2731)
    NotNarrowRg8Plane,           // not a one-layer 2D 2-component Unorm8 plane at all
    UnrecognisedSwizzle,         // T# DST_SEL is neither (R,G,0,1) nor (X,X,X,Y)
    PitchShorterThanRow,         // the resolved row pitch cannot hold the visible row
    NoSiblingLumaPlane,          // no co-bound luma plane pairs with this surface
};

struct AvpChromaVerdict {
    bool match = false;
    AvpChromaReason reason = AvpChromaReason::NotNarrowRg8Plane;
    uint32_t row_bytes = 0;
    uint32_t registered_pitch = 0;
    uint32_t resolved_pitch = 0;
    uint64_t sibling_luma_addr = 0;
};

inline const char* avp_chroma_reason_name(AvpChromaReason reason) {
    switch (reason) {
        case AvpChromaReason::MatchedRegisteredPitch:   return "matched-registered-pitch";
        case AvpChromaReason::MatchedAdjacentLumaPlane: return "matched-adjacent-luma-plane";
        case AvpChromaReason::MatchedSeparateLumaPlane: return "matched-separate-luma-plane";
        case AvpChromaReason::NotNarrowRg8Plane:        return "not-narrow-rg8-plane";
        case AvpChromaReason::UnrecognisedSwizzle:      return "unrecognised-swizzle";
        case AvpChromaReason::PitchShorterThanRow:      return "pitch-shorter-than-row";
        case AvpChromaReason::NoSiblingLumaPlane:       return "no-sibling-luma-plane";
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

// The sibling luma plane's own DST_SEL is deliberately NOT constrained. It reads as a free extra
// discriminator -- the three titles measured live all declare their luma plane (R,0,0,1) -- but
// GFX10 does not tie DST_SEL to a format's component count: the identity (R,G,B,A) remap is a
// perfectly ordinary descriptor for a one-component surface, and the hardware simply returns 0 for
// the absent channels and 1 for alpha. Rejecting it would silently re-create exactly the collapse
// this file exists to prevent, on whichever title happens to spell its luma T# that way. Two
// existing render arms in test_gpu_capture_render.cpp build their luma fixture that way and caught
// the over-tight version of this predicate immediately. (#2731)

// `table` is the draw's complete resource table, used only to look for the sibling luma plane when
// no exact HLE pitch provenance is available (a captured/replayed frame, or a title that stages the
// planes itself). Reading the live registry keeps live and replay on the same contract.
//
// MEMORY ADJACENCY IS EVIDENCE, NOT A REQUIREMENT (#2731). The first version of this search demanded
// that the luma plane END at the chroma plane's address (or at the 64 KiB boundary before it),
// because AvPlayer stages one contiguous NV12 buffer and R-Type Delta's descriptors are cut straight
// out of it. A title that receives NV12 through sceVideodec2 does its own staging, and there is no
// reason for its two plane allocations to touch: Tales of Graces f Remastered (PPSA19991) samples a
// 2048x1088 R8 luma plane and a 1024x544 RG8 chroma plane 0x111000 bytes past its end, and that gap
// alone sent every movie frame down the coverage broadcast. What makes the pair an NV12 pair is the
// GEOMETRY -- one luma texel per chroma byte across the row, two luma rows per chroma row, the same
// physical row pitch, both linear, both one-layer 2D, both Unorm8, and both bound by the same draw.
// Adjacency is kept as a separate, stronger verdict so the log still distinguishes the two routes.
inline AvpChromaVerdict classify_avplayer_chroma_plane(
    const gpu::ShaderResource& r, uint32_t tw, uint32_t th,
    const std::vector<gpu::ShaderResource>& table) {
    using RC = gpu::ResourceClass;
    AvpChromaVerdict v;
    const bool tiled = gpu::tile_mode_is_tiled(r.tile_mode);
    if (r.cls != RC::Texture || r.format != gpu::DataFormat::Unorm8 ||
        r.num_components != 2 || !avp_plane_is_one_layer_2d(r) ||
        (r.tile_mode != 0u && !tiled) ||
        r.compression_enabled || !tw || !th || tw > UINT32_MAX / 2u ||
        th > UINT32_MAX / 2u) {
        v.reason = AvpChromaReason::NotNarrowRg8Plane;
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
    // A TILED plane pair has no row pitch to reason about, so neither the HLE registry nor the
    // resolved-pitch comparison applies: the whole surface is a padded block of micro-tiles whose
    // size depends on the mode and the element width. Sonic Origins (PPSA05325) stages its decoded
    // 3840x2160 NV12 exactly that way -- both planes SW_64KB_S, declared as one-layer 2D arrays --
    // and its luma plane's TILED size (0x870000 for 3840x2160 at 1 B/texel, against 0x7e9000 tight)
    // lands precisely on the chroma plane's address, four allocations out of four. So the tiled
    // route keeps adjacency as a REQUIREMENT: it is available here, it is exact, and it is much the
    // strongest evidence a pair can carry. Measured live with PROSPER_AVPCHROMA_LOG (#2731).
    if (tiled) {
        for (const auto& luma : table) {
            if (luma.cls != RC::Texture || luma.format != gpu::DataFormat::Unorm8 ||
                luma.num_components != 1 || !avp_plane_is_one_layer_2d(luma) ||
                luma.tile_mode != r.tile_mode || luma.compression_enabled ||
                !luma.width || !luma.height ||
                (static_cast<uint64_t>(luma.width) + 1u) / 2u != tw ||
                (static_cast<uint64_t>(luma.height) + 1u) / 2u != th)
                continue;
            const uint64_t luma_bytes = gpu::tiled_surface_bytes(
                luma.width, luma.height, luma.tile_mode, 0, 1u);
            if (!luma_bytes || luma.gpu_addr > UINT64_MAX - luma_bytes) continue;
            const uint64_t luma_end = luma.gpu_addr + luma_bytes;
            if (luma_end == r.gpu_addr ||
                (luma_end <= UINT64_MAX - 0xffffu &&
                 ((luma_end + 0xffffu) & ~uint64_t{0xffffu}) == r.gpu_addr)) {
                v.match = true;
                v.reason = AvpChromaReason::MatchedAdjacentLumaPlane;
                v.sibling_luma_addr = luma.gpu_addr;
                return v;
            }
        }
        v.reason = AvpChromaReason::NoSiblingLumaPlane;
        return v;
    }
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
    const uint64_t chroma_bytes = static_cast<uint64_t>(chroma_pitch) * th;
    const gpu::ShaderResource* adjacent_luma = nullptr;
    const gpu::ShaderResource* separate_luma = nullptr;
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
        if (luma.gpu_addr > UINT64_MAX - luma_bytes ||
            r.gpu_addr > UINT64_MAX - chroma_bytes)
            continue;
        const uint64_t luma_end = luma.gpu_addr + luma_bytes;
        // Two planes of one picture are two DISJOINT ranges. Overlap means one allocation being
        // read two different ways, which is never an NV12 pair, so it is the one memory relation
        // that still disqualifies a candidate.
        if (!(luma_end <= r.gpu_addr || r.gpu_addr + chroma_bytes <= luma.gpu_addr)) continue;
        // Exactly adjacent, or beginning at the next 64 KiB boundary: one staged NV12 buffer.
        if (luma_end == r.gpu_addr ||
            (luma_end <= UINT64_MAX - 0xffffu &&
             ((luma_end + 0xffffu) & ~uint64_t{0xffffu}) == r.gpu_addr)) {
            adjacent_luma = &luma;
            break;
        }
        if (!separate_luma) separate_luma = &luma;
    }
    if (const gpu::ShaderResource* luma = adjacent_luma ? adjacent_luma : separate_luma) {
        v.match = true;
        v.reason = adjacent_luma ? AvpChromaReason::MatchedAdjacentLumaPlane
                                 : AvpChromaReason::MatchedSeparateLumaPlane;
        v.sibling_luma_addr = luma->gpu_addr;
        return v;
    }
    v.reason = AvpChromaReason::NoSiblingLumaPlane;
    return v;
}

} // namespace prosper::frontend
