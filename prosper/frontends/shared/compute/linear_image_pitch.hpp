// linear_image_pitch.hpp -- the row pitch of a guest-backed LINEAR 2D image the compute backend reads
// and writes.
//
// GFX10 aligns a linear image's row pitch to 256 bytes unless a pitch is stated, so a 1920-wide R8
// plane occupies 2048 bytes per row and the last 128 bytes of every row are padding. The compute
// upload read such a plane as tight rows, so every row after the first started 128 bytes early and
// the padding showed as a diagonal band (Black Flag's warning-screen video planes came out in green
// stripes). The rule itself is gpu::resolved_linear_row_pitch (gpu/resources/linear_row_pitch.hpp):
// the descriptor's pitch, else a registered pitch (AvPlayer planes, VideoOut display buffers), else
// the 256-byte alignment. This header only decides which compute shapes it applies to.
//
// The live compute backend applies it to the sampled upload AND the storage seed and writeback
// (#4606), whoever owns the address -- a renderer-owned target included -- so one address keeps one
// layout across dispatches and agrees with the graphics path and the VideoOut presenter.
//
// Answers from plain metadata so it is unit-testable without a device.
#pragma once

#include <cstddef>
#include <cstdint>

#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/bc_decode.hpp"
#include "gpu/resources/linear_row_pitch.hpp"

namespace prosper::frontend {

// Bytes between the starts of consecutive rows of this image in guest memory, or 0 when the rows are
// tight or the image is not a shape this rule describes (tiled, array, cube, mip chain, compressed,
// block-compressed). `bytes_per_texel` is the texel size.
inline size_t compute_linear_row_pitch(const gpu::ShaderResource& r, uint32_t bytes_per_texel) {
    using gpu::ResourceClass;
    if (!bytes_per_texel || !r.width || !r.height || r.tile_mode != 0u || r.in_mip_tail ||
        r.compression_enabled || r.sample_count > 1u || r.declared_mip_levels > 1u ||
        r.layer_stride_bytes || r.layer_mip_offset_bytes || gpu::bc_block_bytes(r.format))
        return 0;
    if (r.cls != ResourceClass::Texture && r.cls != ResourceClass::StorageImage) return 0;
    if (!(r.img_dim == 1u || (r.img_dim == 5u && r.depth == 1u))) return 0;
    return gpu::padded_linear_row_pitch(r, r.width, bytes_per_texel);
}

using gpu::copy_linear_rows;

}   // namespace prosper::frontend
