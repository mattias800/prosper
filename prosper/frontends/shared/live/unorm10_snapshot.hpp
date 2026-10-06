// unorm10_snapshot.hpp -- unpack a guest R10G10B10A2 UNORM linear image into the RGBA8 bytes the
// renderer keeps for the same render target.
//
// A compute pass may write a display buffer as 2_10_10_10 (R in bits [9:0], A in bits [31:30]).
// The renderer stores a 10:10:10:2 colour target as RGBA8 (backend_color_format falls back to it),
// so a result in the packed layout cannot be copied in; it is narrowed the same way a sampled
// upload of this format already is.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace prosper::frontend {

// Returns false when `bytes` is not exactly width*height packed words.
inline bool unpack_unorm10_to_rgba8(const uint8_t* packed, size_t packed_bytes, uint64_t width,
                                    uint64_t height, std::vector<uint8_t>& rgba8) {
    if (!packed || !width || !height || width > SIZE_MAX / height) return false;
    const size_t texels = static_cast<size_t>(width * height);
    if (texels > SIZE_MAX / 4 || packed_bytes != texels * 4) return false;
    rgba8.resize(texels * 4);
    for (size_t t = 0; t < texels; ++t) {
        uint32_t word;
        std::memcpy(&word, packed + t * 4, sizeof(word));
        const uint32_t r = word & 0x3ffu, g = (word >> 10) & 0x3ffu, b = (word >> 20) & 0x3ffu;
        const uint32_t a = word >> 30;
        rgba8[t * 4 + 0] = static_cast<uint8_t>((r * 255u + 511u) / 1023u);
        rgba8[t * 4 + 1] = static_cast<uint8_t>((g * 255u + 511u) / 1023u);
        rgba8[t * 4 + 2] = static_cast<uint8_t>((b * 255u + 511u) / 1023u);
        rgba8[t * 4 + 3] = static_cast<uint8_t>((a * 255u + 1u) / 3u);
    }
    return true;
}

} // namespace prosper::frontend
