// linear_image_pitch.hpp -- the row pitch of a guest-backed LINEAR 2D image the compute backend reads
// and writes.
//
// A linear (tile mode 0) image's rows can be padded: AvPlayer publishes its 1920-wide R8 planes at a
// 2048-byte pitch, so the last 128 bytes of every row are padding. The compute upload read such a
// plane as tight rows, so every row after the first started 128 bytes early and the padding showed
// as a diagonal band (Black Flag's warning-screen video planes came out in green stripes).
//
// The rule is deliberately STATED-ONLY (#4586 review, #4606): a pitch is honoured when the guest
// states one -- the descriptor's own pitch field (a capture-resolved pitch) or an HLE producer's
// registration for the range (register_guest_linear_texture_layout, e.g. AvPlayer planes). It is
// never inferred from the GFX10 256-byte alignment. Inferring it would give every unaligned linear
// image a padded layout that the rest of prosper does not share: renderer-owned colour targets and
// the VideoOut presenter keep tight rows, and a compute result mirrored into a renderer target would
// change layout between dispatches. With a stated pitch the layout is a property of the address,
// so a compute producer and a compute consumer of it agree whether or not a retained result is
// borrowed.
//
// Answers from plain metadata so it is unit-testable without a device.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/bc_decode.hpp"
#include "gpu/texture/guest_texture_layout.hpp"

namespace prosper::frontend {

// Bytes between the starts of consecutive rows the guest STATED for this image, or 0 when it stated
// none, the stated pitch is tight, or the image is not a shape this rule describes (tiled, array,
// cube, mip chain, compressed, block-compressed). Replay-owned bytes (`host_data`) use only the
// descriptor's field, which is the capture's resolved pitch. `bytes_per_texel` is the texel size.
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
    if (!pitch && !r.host_data && tight <= UINT32_MAX)
        pitch = gpu::guest_linear_texture_row_pitch(r.gpu_addr, static_cast<uint32_t>(tight));
    return pitch > tight ? pitch : 0;
}

// The live compute backend applies compute_linear_row_pitch to BOTH the sampled read and the storage
// seed/writeback (live_compute.cpp, `live_guest_row_pitch`), so a compute producer and a compute
// consumer of one image agree (#4606). Two kinds of memory keep tight rows even under a stated
// pitch, because another part of prosper owns those bytes and reads them tight: a renderer-owned
// colour target and a registered VideoOut display buffer (the presenter reads width x height x 4).
// A stated-pitch result is never handed to the renderer either -- neither mirrored into a renderer
// target (the LinearPitch exact-result decline) nor published as a CPU RTT snapshot -- so one address
// keeps one layout across dispatches, and graphics reads it from guest memory at the stated pitch.
// CONFIDENCE: HIGH that a stated pitch is the guest's layout; MED that the two exemptions never
// meet a stated pitch on a real title (none is measured).

// Copy `rows` rows of `row_bytes` between two pitches: gather into tight rows (dst_pitch ==
// row_bytes) or scatter back to a padded pitch. Padding bytes in the destination are not written.
inline void copy_linear_rows(uint8_t* dst, size_t dst_pitch, const uint8_t* src, size_t src_pitch,
                             size_t row_bytes, uint32_t rows) {
    for (uint32_t y = 0; y < rows; ++y)
        std::memcpy(dst + y * dst_pitch, src + y * src_pitch, row_bytes);
}

}   // namespace prosper::frontend
