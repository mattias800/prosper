// bc_decode.hpp — S3TC/BCn block decompression to linear RGBA8.
//
// PS5 art textures are commonly stored block-compressed (BC1/BC2/BC3 = DXT1/3/5). The host upload
// path needs linear RGBA8 (our sampled-texture backend uploads 4-bytes/texel), and llvmpipe does not
// universally advertise BC sampling, so we decode CPU-side on upload. Pure + deterministic + testable:
// given the compressed block bytes for a WxH surface, produce W*H*4 RGBA8 (row-major, top-left origin).
//
// Standard S3TC/BPTC decode (Khronos data-format spec §18-19, MS D3D11 §19.5): a 4x4-texel block is
// 8 bytes (BC1/BC4) or 16 bytes (BC2/BC3/BC5/BC6/BC7). Implemented: BC1/BC2/BC3 (Messenger UI/text),
// BC4/BC5 (single/dual-channel ramps) and BC7 (mode-switched; DOLL's material atlases) — #290.
// BC6H UF16 (HDR half-float; DOLL's title skybox/probes, #273) decodes too — then CLAMPS to
// UNORM8, losing >1.0 energy like the fp16-surface path. SIGNED variants (BC4/5 SNORM, BC6H SF16)
// stay skipped upstream.
#pragma once
#include <cstdint>
#include "gpu/resources/shader_resources.hpp"

namespace prosper::gpu {

// Bytes per 4x4 block for a block-compressed DataFormat (8 for BC1/BC4, 16 for the rest; 0 if `f` is
// not block-compressed).
uint32_t bc_block_bytes(DataFormat f);

// #325: is this sampled T# one prosper uploads as a real Vulkan array, with the shader indexing its
// layers? THE UPLOADER AND THE RECOMPILER MUST ANSWER THIS IDENTICALLY -- the uploader picks the
// image view type from it and the recompiler picks OpTypeImage's Arrayed flag from it, and a
// disagreement between the two is a descriptor mismatch rather than a wrong pixel. That is why it
// is written once here instead of being spelled out at each site.
//
// Block-compressed only, deliberately. The per-slice decoder handles tiled and linear BC and plain
// byte-per-texel surfaces; it does NOT carry the fp16/fp32/unorm16 narrowing that the single-surface
// decoder does, so a Float32 array decodes to black (measured). Widening this predicate therefore
// means teaching the slice loop those conversions first -- not merely relaxing the condition.
inline bool guest_texture_is_uploaded_array(uint32_t img_dim, uint32_t depth, DataFormat format) {
    return img_dim == 5u && depth > 1u && bc_block_bytes(format) != 0u;
}

// Decode a block-compressed surface `src` (bc_block_bytes(fmt) per 4x4 block, blocks row-major over a
// ceil(w/4) x ceil(h/4) grid) into `dst` as W*H*4 RGBA8. `dst` must hold width*height*4 bytes.
// `src_bytes` bounds the read (a short/absent source leaves the unreachable texels transparent-black).
// Returns true if `fmt` is a supported block format (BC1/BC2/BC3/BC4/BC5/BC6H/BC7), false otherwise
// (dst untouched). BC4 decodes to (R,0,0,255), BC5 to (R,G,0,255) — the hardware channel rule; the
// T# DST_SEL swizzle routes them. UNORM only (SNORM BC4/BC5 variants are kept skipped upstream).
bool bc_decode_surface(uint8_t* dst, const uint8_t* src, size_t src_bytes,
                       uint32_t width, uint32_t height, DataFormat fmt);

} // namespace prosper::gpu
