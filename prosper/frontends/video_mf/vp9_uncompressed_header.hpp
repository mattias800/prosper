#pragma once

#include <cstddef>
#include <cstdint>

namespace prosper::video {

// The frame size a VP9 KEY frame declares in its uncompressed header (VP9 bitstream spec section 6.2,
// uncompressed_header / color_config / frame_size).
//
// Why the Media Foundation access-unit decoder needs this at all: sceVideodec2 hands prosper bare
// access units and never a container, so nothing but the bitstream says how big the pictures are,
// and an MF decoder is configured through an input media TYPE before it sees a byte. Handing it a
// size read off the key frame is the same thing a demuxer would do from a WebM/IVF header -- just
// taken from the one place the guest's stream actually carries it.
//
// Reads only what it has to: frame_marker, profile, show_existing_frame, frame_type, show_frame,
// error_resilient_mode, the sync code, color_config, then frame_width_minus_1 /
// frame_height_minus_1. Returns false for anything that is not a well-formed key-frame header --
// an inter frame, a show-existing frame, a wrong marker or sync code, or a truncated buffer -- so a
// caller can never mistake "no size here" for a size.
//
// A superframe needs no special case: its index is at the END of the access unit, and its first
// frame starts at byte 0, so a key frame inside one is read exactly like a bare key frame.
//
// CONFIDENCE: HIGH -- this is the published bitstream grammar, and the test decodes it from a
// committed libvpx-encoded asset whose size is known independently.
inline bool vp9_key_frame_size(const uint8_t* data, size_t bytes, uint32_t& width,
                               uint32_t& height) {
    if (!data || bytes == 0) return false;
    size_t bit = 0;
    bool overrun = false;
    auto read = [&](unsigned count) -> uint32_t {
        uint32_t value = 0;
        for (unsigned i = 0; i < count; ++i, ++bit) {
            if (bit / 8 >= bytes) { overrun = true; return 0; }
            value = (value << 1) | ((data[bit / 8] >> (7 - bit % 8)) & 1u);
        }
        return value;
    };
    if (read(2) != 2) return false;                        // frame_marker
    const uint32_t profile_low = read(1);
    const uint32_t profile = profile_low | (read(1) << 1);
    if (profile == 3 && read(1) != 0) return false;        // reserved_zero
    if (read(1)) return false;                             // show_existing_frame: no header
    if (read(1) != 0) return false;                        // frame_type: 0 = KEY_FRAME
    read(1);                                               // show_frame
    read(1);                                               // error_resilient_mode
    if (read(24) != 0x498342u) return false;               // frame_sync_code
    // color_config()
    if (profile >= 2) read(1);                             // ten_or_twelve_bit
    const uint32_t color_space = read(3);
    constexpr uint32_t kCsRgb = 7;
    if (color_space != kCsRgb) {
        read(1);                                           // color_range
        if (profile == 1 || profile == 3) read(3);         // subsampling_x, subsampling_y, reserved
    } else if (profile == 1 || profile == 3) {
        read(1);                                           // reserved_zero
    }
    // frame_size()
    const uint32_t w = read(16) + 1;
    const uint32_t h = read(16) + 1;
    if (overrun) return false;
    width = w;
    height = h;
    return true;
}

} // namespace prosper::video
