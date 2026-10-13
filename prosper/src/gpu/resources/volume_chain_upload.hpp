// volume_chain_upload.hpp — per-level staging and upload for all-in-tail thick-3D chains.
//
// A base-zero chain view over a proven all-in-tail pyramid (tiled_volume_tail_layout)
// materializes every declared level into one Vulkan 3D image. Level zero uploads through
// the established single-level path; this file sizes and writes levels 1..N-1: each
// through its own in-block origin (detile_volume_tail_level), then either straight-copied
// (native formats, whose guest texels are already backend texels) or broadcast 1-to-4
// (R8, which the backend samples as RGBA8 -- the same broadcast the single-level R8 path
// performs, repeated per level).
//
// Pure CPU bytes, no Vulkan types: the backend maps the returned offsets onto its staging
// allocation and copies the regions they name. A level that fails to detile, an offset
// that overruns, or a conversion this file does not model fails the whole upload
// (fail-visible) rather than leaving staging-allocation contents behind as texels.
#pragma once

#include "gpu/resources/mip_chain_plan.hpp"

#include <cstdint>
#include <vector>

namespace prosper::gpu {

// True when `plan` is a placed all-in-tail volume chain: every level carries a shifted
// depth (thin-2D plans leave depth zero, so this also discriminates the two shapes for
// callers that must pick an uploader).
bool mip_chain_plan_is_volume_tail(const MipChainPlan& plan);

// The one conversion besides identity the uploader reproduces: R8 broadcast 1-to-4.
// Shared by image creation (which admits the backend width) and the upload cascade
// (which performs the broadcast), so the two cannot disagree about what a level costs.
// Narrow: Unorm8 x1 only. Unused by any other arm.
inline bool volume_chain_r8_broadcast_shape(DataFormat format, uint32_t components) {
    return format == DataFormat::Unorm8 && (components ? components : 1u) == 1u;
}

// Whether the backend's texel width for this chain is one the uploader reproduces:
// identity, or the proven R8 broadcast above. Anything else would leave levels 1+ as
// staging residue -- the drift detector's failure, made structural. In particular the
// old 2D rule (`backend == guest`, always) is wrong here: the admitted R8 chain reports
// one guest byte per texel against four backend bytes.
bool volume_chain_backend_width_agrees(const MipChainPlan& plan, uint32_t guest_bytes_per_texel,
                                       uint64_t backend_bytes_per_texel, bool broadcast_r8);

// Staging-materialization math for one backend image: the running byte offsets of every
// level's backend texels (offsets[0] is 0: level zero opens the upload) and the total.
// Single derivation shared by image creation (which sizes staging) and upload (which
// fills it). `alignment` is the caller's Vulkan bufferOffset granularity (multiple of
// four AND of the texel block size); 0 refuses.
struct VolumeChainStaging {
    std::vector<uint64_t> offsets;
    uint64_t total_bytes = 0;
};
VolumeChainStaging volume_chain_staging_layout(const MipChainPlan& plan,
                                               uint64_t backend_bytes_per_texel,
                                               uint64_t alignment);

// Fill levels 1..N-1 of `upload` (level zero is written by the single-level path).
// `allocation_base` is the shared tail block (plan.levels[*].byte_offset is 0 for every
// tail level); `allocation_bytes` bounds the detile reads and must cover the plan's whole
// span -- a short allocation is refused BEFORE any detile, because the detiler zero-fills
// short reads and reports success, which would bake zeros in as texels. `upload_bytes`
// is the destination capacity: every level must fit, and levels must not overlap.
// `staging_offsets` carries one backend-byte offset per planned level (plain uint64 words,
// so this header stays Vulkan-free; the backend passes its own offset vector's data).
// Native copies require guest == backend texel width; the R8 broadcast requires 1 guest
// byte to 4 backend bytes. Anything else -- an invalid or non-volume plan, a level count
// that disagrees with the level vector, a size/offset overrun, an overlap, an unmodelled
// conversion -- returns false having written nothing further.
//
// Trusted-precondition map (production proofs, not assumptions): the backend passes the
// plan's own span (readability-probed over that span before upload), generated offsets
// (single derivation with creation, above), and the mapped staging size; the unit tests
// below pin every refusal for hand-built violations of each one.
bool upload_volume_tail_chain_levels(const MipChainPlan& plan, const uint8_t* allocation_base,
                                     size_t allocation_bytes, uint32_t tile_mode,
                                     uint32_t guest_bytes_per_texel,
                                     uint64_t backend_bytes_per_texel, bool broadcast_r8,
                                     uint8_t* upload, size_t upload_bytes,
                                     const uint64_t* staging_offsets, size_t staging_count);

} // namespace prosper::gpu
