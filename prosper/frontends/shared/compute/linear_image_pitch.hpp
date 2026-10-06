// linear_image_pitch.hpp -- the row pitch of a guest-backed LINEAR 2D image the compute backend reads.
//
// GFX10 lays a linear (tile mode 0) image out with a row pitch aligned to 256 bytes, so a 1920-wide
// R8 plane occupies 2048 bytes per row and the last 128 bytes of every row are padding. The T# does
// not encode that pitch: the same rule the graphics path applies (`linear_sampled_row_pitch`, or a
// pitch an HLE producer registered for the range) is the only source. The graphics upload reads such
// an image row by row; the compute upload read it as if the rows were tight, so every row after the
// first started `padding` bytes early and the padding appeared as a diagonal band of zeros. Black
// Flag's warning-screen video planes are 1920x1080 R8, so their background came out in green stripes.
//
// Answers from plain metadata so it is unit-testable without a device.
#pragma once

#include <cstddef>
#include <cstdint>

#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/bc_decode.hpp"
#include "gpu/texture/guest_texture_layout.hpp"
#include "gpu/texture/tile.hpp"

namespace prosper::frontend {

// Bytes between the starts of consecutive rows, or 0 when the rows are tight or the image is not a
// shape this rule describes (tiled, array, cube, mip chain, compressed, block-compressed, replay-owned
// bytes whose pitch the capture already resolved). `bytes_per_texel` is the sampled texel size.
inline size_t compute_linear_row_pitch(const gpu::ShaderResource& r, uint32_t bytes_per_texel) {
    using gpu::ResourceClass;
    if (!bytes_per_texel || !r.width || !r.height || r.tile_mode != 0u || r.in_mip_tail ||
        r.compression_enabled || r.sample_count > 1u || r.declared_mip_levels > 1u ||
        r.layer_stride_bytes || r.layer_mip_offset_bytes || gpu::bc_block_bytes(r.format))
        return 0;
    if (r.cls != ResourceClass::Texture && r.cls != ResourceClass::StorageImage) return 0;
    if (!(r.img_dim == 1u || (r.img_dim == 5u && r.depth == 1u))) return 0;
    const size_t tight = static_cast<size_t>(r.width) * bytes_per_texel;
    size_t pitch = r.linear_row_pitch_bytes;
    if (!pitch && !r.host_data) {
        if (tight > UINT32_MAX) return 0;
        pitch = gpu::guest_linear_texture_row_pitch(r.gpu_addr, static_cast<uint32_t>(tight));
        if (!pitch) pitch = gpu::linear_sampled_row_pitch(r.width, bytes_per_texel);
    }
    return pitch > tight ? pitch : 0;
}

}   // namespace prosper::frontend
