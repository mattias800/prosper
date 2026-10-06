// linear_row_pitch.hpp -- the one row-pitch rule for a LINEAR (tile mode 0) guest image, and the row
// copy that applies it.
//
// GFX10 aligns a linear surface's row pitch to 256 bytes, so a guest linear image's memory is padded
// unless a pitch is stated: a 1920-wide R8 plane occupies 2048 bytes per row. A pitch is STATED by
// the descriptor's own field (a capture-resolved pitch) or by an HLE producer that registered the
// allocation's layout (register_guest_linear_texture_layout: AvPlayer planes, VideoOut display
// buffers). Every path that reads or writes a linear guest image's bytes -- the graphics and compute
// sampled uploads, the compute storage seed and writeback, captures -- uses this rule, so one
// address has one layout whoever touches it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/guest_texture_layout.hpp"
#include "gpu/texture/tile.hpp"

namespace prosper::gpu {

// Bytes between consecutive rows of `r`'s guest memory for a `width`-element row of `bpt` bytes:
// the descriptor's pitch, else a registered pitch, else the GFX10 256-byte alignment. Replay-owned
// bytes (`host_data`) use only the descriptor's field and are otherwise tight. UINT32_MAX when the
// row does not fit.
inline uint32_t resolved_linear_row_pitch(const ShaderResource& r, uint32_t width, uint32_t bpt) {
    if (r.linear_row_pitch_bytes) return r.linear_row_pitch_bytes;
    const uint64_t tight = static_cast<uint64_t>(width) * bpt;
    if (tight > UINT32_MAX) return UINT32_MAX;
    if (r.host_data) return static_cast<uint32_t>(tight);
    if (const uint32_t registered =
            guest_linear_texture_row_pitch(r.gpu_addr, static_cast<uint32_t>(tight)))
        return registered;
    const size_t aligned = linear_sampled_row_pitch(width, bpt);
    return aligned > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(aligned);
}

// Copy `rows` rows of `row_bytes` between two pitches: gather into tight rows (dst_pitch ==
// row_bytes) or scatter back to a padded pitch. Padding bytes in the destination are not written.
inline void copy_linear_rows(uint8_t* dst, size_t dst_pitch, const uint8_t* src, size_t src_pitch,
                             size_t row_bytes, uint32_t rows) {
    for (uint32_t y = 0; y < rows; ++y)
        std::memcpy(dst + y * dst_pitch, src + y * src_pitch, row_bytes);
}

}   // namespace prosper::gpu
