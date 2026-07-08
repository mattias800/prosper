// bc_decode.hpp — S3TC/BCn block decompression to linear RGBA8.
//
// PS5 art textures are commonly stored block-compressed (BC1/BC2/BC3 = DXT1/3/5). The host upload
// path needs linear RGBA8 (our sampled-texture backend uploads 4-bytes/texel), and llvmpipe does not
// universally advertise BC sampling, so we decode CPU-side on upload. Pure + deterministic + testable:
// given the compressed block bytes for a WxH surface, produce W*H*4 RGBA8 (row-major, top-left origin).
//
// Standard S3TC decode (Khronos DXTn spec): a 4x4-texel block is 8 bytes (BC1/BC4) or 16 bytes
// (BC2/BC3/BC5/BC7). We implement the two the game uses first — BC1 and BC3 — plus BC2 (same color
// sub-block, explicit-alpha), which cost nothing extra. BC4/5/6/7 are left to a follow-up.
#pragma once
#include <cstdint>
#include "shader_resources.hpp"

namespace prosper::gpu {

// Bytes per 4x4 block for a block-compressed DataFormat (8 for BC1/BC4, 16 for the rest; 0 if `f` is
// not block-compressed).
uint32_t bc_block_bytes(DataFormat f);

// Decode a block-compressed surface `src` (bc_block_bytes(fmt) per 4x4 block, blocks row-major over a
// ceil(w/4) x ceil(h/4) grid) into `dst` as W*H*4 RGBA8. `dst` must hold width*height*4 bytes.
// `src_bytes` bounds the read (a short/absent source leaves the unreachable texels transparent-black).
// Returns true if `fmt` is a supported block format (BC1/BC2/BC3), false otherwise (dst untouched).
bool bc_decode_surface(uint8_t* dst, const uint8_t* src, size_t src_bytes,
                       uint32_t width, uint32_t height, DataFormat fmt);

} // namespace prosper::gpu
