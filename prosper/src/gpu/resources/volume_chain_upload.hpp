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
// tail level); `allocation_bytes` bounds the detile reads. `staging_offsets` carries one
// backend-byte offset per planned level (plain uint64 words, so this header stays
// Vulkan-free; the backend passes its own offset vector's data). Native copies require
// guest == backend texel width; the R8 broadcast requires 1 guest byte to 4 backend
// bytes. Anything else -- an invalid or non-volume plan, a size/offset overrun, an
// unmodelled conversion -- returns false having written nothing further.
bool upload_volume_tail_chain_levels(const MipChainPlan& plan, const uint8_t* allocation_base,
                                     size_t allocation_bytes, uint32_t tile_mode,
                                     uint32_t guest_bytes_per_texel,
                                     uint64_t backend_bytes_per_texel, bool broadcast_r8,
                                     uint8_t* upload, const uint64_t* staging_offsets,
                                     size_t staging_count);

} // namespace prosper::gpu
