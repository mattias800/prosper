// agc_shader_layout.cpp — see agc_shader_layout.hpp. V# decode + the front-half resource-table build.
#include "agc_shader_layout.hpp"
#include <cstdio>
#include <cstdlib>

namespace prosper::gpu {

// RDNA2 (GFX10/PS5) V# dword3 carries a COMBINED 7-bit FORMAT at bits[18:12] — NOT the separate GCN/PS4
// NFMT[14:12]/DFMT[18:15] split (reading that on a PS5 V# yields garbage fields, so every descriptor came
// out DataFormat::Unknown and fell back to Float32 downstream — e.g. a UNORM8 vertex color 0xffffffff
// became a float32 NaN → the title composite multiplied to black). Cross-checked against Kyty
// (Shader.h:585 `Format()=(fields[3]>>12)&0x7F` on the Gen5 path). The value list follows the RDNA2 ISA
// buffer-format table; the four game-observed anchors 56/64/74/77 are Kyty-confirmed.
// CONFIDENCE: HIGH on the anchors, MED on the rest of the table.
void rdna2_buffer_format(uint32_t fmt, DataFormat* out_fmt, uint32_t* out_components) {
    DataFormat f = DataFormat::Unknown; uint32_t n = 0;
    switch (fmt) {
        case  1: f = DataFormat::Unorm8;  n = 1; break;
        case  2: f = DataFormat::Snorm8;  n = 1; break;
        case  5: f = DataFormat::Uint8;   n = 1; break;
        case  6: f = DataFormat::Sint8;   n = 1; break;
        case  7: f = DataFormat::Unorm16; n = 1; break;
        case  8: f = DataFormat::Snorm16; n = 1; break;
        case 11: f = DataFormat::Uint16;  n = 1; break;
        case 12: f = DataFormat::Sint16;  n = 1; break;
        case 13: f = DataFormat::Float16; n = 1; break;
        case 14: f = DataFormat::Unorm8;  n = 2; break;
        case 15: f = DataFormat::Snorm8;  n = 2; break;
        case 18: f = DataFormat::Uint8;   n = 2; break;
        case 19: f = DataFormat::Sint8;   n = 2; break;
        case 20: f = DataFormat::Uint32;  n = 1; break;
        case 21: f = DataFormat::Sint32;  n = 1; break;
        case 22: f = DataFormat::Float32; n = 1; break;
        case 23: f = DataFormat::Unorm16; n = 2; break;
        case 24: f = DataFormat::Snorm16; n = 2; break;
        case 27: f = DataFormat::Uint16;  n = 2; break;
        case 28: f = DataFormat::Sint16;  n = 2; break;
        case 29: f = DataFormat::Float16; n = 2; break;
        case 56: f = DataFormat::Unorm8;  n = 4; break;   // 8_8_8_8_UNORM (vertex colors) — Kyty-confirmed
        case 57: f = DataFormat::Snorm8;  n = 4; break;
        case 60: f = DataFormat::Uint8;   n = 4; break;
        case 61: f = DataFormat::Sint8;   n = 4; break;
        case 62: f = DataFormat::Uint32;  n = 2; break;
        case 63: f = DataFormat::Sint32;  n = 2; break;
        case 64: f = DataFormat::Float32; n = 2; break;   // 32_32_FLOAT (UVs) — Kyty-confirmed
        case 65: f = DataFormat::Unorm16; n = 4; break;
        case 66: f = DataFormat::Snorm16; n = 4; break;
        case 69: f = DataFormat::Uint16;  n = 4; break;
        case 70: f = DataFormat::Sint16;  n = 4; break;
        case 71: f = DataFormat::Float16; n = 4; break;
        case 72: f = DataFormat::Uint32;  n = 3; break;
        case 73: f = DataFormat::Sint32;  n = 3; break;
        case 74: f = DataFormat::Float32; n = 3; break;   // 32_32_32_FLOAT (positions) — Kyty-confirmed
        case 75: f = DataFormat::Uint32;  n = 4; break;
        case 76: f = DataFormat::Sint32;  n = 4; break;
        case 77: f = DataFormat::Float32; n = 4; break;   // 32_32_32_32_FLOAT — Kyty-confirmed
        default: break;                                    // Unknown -> caller/recompiler fallback
    }
    if (out_fmt) *out_fmt = f;
    if (out_components) *out_components = n;
}

DecodedBufferDescriptor decode_buffer_descriptor(const uint32_t v[4]) {
    DecodedBufferDescriptor d;
    d.base        = ((uint64_t)v[0] | ((uint64_t)v[1] << 32)) & 0xFFFFFFFFFFFFull;  // Base48
    d.stride      = (v[1] >> 16) & 0x3FFFu;                                          // 14-bit stride
    d.num_records = v[2];
    rdna2_buffer_format((v[3] >> 12) & 0x7Fu, &d.format, &d.num_components);
    // num_records is in units of `stride` when strided, else raw bytes. Compute in 64-bit and clamp so a
    // 32-bit wrap (num_records is a full 32-bit field) can't produce a small value that slips a bogus
    // buffer under the caller's `size_bytes > 0x10000000` plausibility guard.
    uint64_t sz = d.stride ? (uint64_t)d.num_records * d.stride : (uint64_t)d.num_records;
    d.size_bytes = sz > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)sz;
    return d;
}

DecodedImageDescriptor decode_image_descriptor(const uint32_t t[8]) {
    DecodedImageDescriptor d;
    d.base      = (((uint64_t)t[0] | ((uint64_t)t[1] << 32)) & 0xFFFFFFFFFFull) << 8;             // Base40
    d.width     = (uint32_t)(((t[1] >> 30) & 0x3u) | (((t[2] >> 0) & 0xFFFu) << 2)) + 1;          // Width5
    d.height    = (uint32_t)((t[2] >> 14) & 0x3FFFu) + 1;                                          // Height5
    d.format    = (t[1] >> 20) & 0x1FFu;                                                           // Format
    d.tile_mode = (t[3] >> 20) & 0x1Fu;                                                            // TileMode
    d.type      = (uint8_t)((t[3] >> 28) & 0xFu);                                                  // Type
    return d;
}

ShaderResourceTable build_shader_resources(const AgcShaderHeader& shdr,
                                           const uint32_t* user_sgprs, uint32_t num_user_sgprs,
                                           uint32_t user_sgpr_base) {
    ShaderResourceTable table;
    const AgcShaderUserData* ud = shdr.user_data;
    if (!ud || !user_sgprs) return table;

    uint32_t binding = 0;

    // Constant buffers: sharp_resource_offset[3] (storage-as-constant). Each slot's offset_dw points at
    // a 4-dword V# in the user-data SGPR block. srt_offset = the descriptor's byte offset within
    // user_data (offset_dw * 4) — the recompiler's provenance key.
    const AgcShaderSharp* cbufs = ud->sharp_resource_offset[3];
    if (cbufs) {
        for (uint16_t slot = 0; slot < ud->sharp_resource_count[3]; slot++) {
            const AgcShaderSharp& s = cbufs[slot];
            if (s.empty()) continue;
            uint32_t off = s.offset_dw();
            if ((uint64_t)off + 4 > num_user_sgprs) continue;   // descriptor must fit in the SGPR block
            DecodedBufferDescriptor d = decode_buffer_descriptor(&user_sgprs[off]);
            ShaderResource r;
            r.cls            = ResourceClass::ConstantBuffer;
            r.format         = d.format;
            r.num_components  = d.num_components;
            r.binding        = binding++;
            r.gpu_addr       = d.base;
            r.size           = d.size_bytes;
            r.stride         = d.stride;
            r.srt_offset     = off * 4;                 // byte offset within user_data (indirect path)
            r.sgpr_base      = user_sgpr_base + off;    // the shader SGPR holding this V# (s_buffer_load SBASE)
            table.resources.push_back(r);
        }
    }

    // Textures: sharp_resource_offset[0] (textures2D). Each slot's offset_dw points at an 8-dword T#
    // in the user-data SGPR block. The PS reads it directly in SGPRs (image_sample SRSRC), so DIRECT
    // provenance: sgpr_base = offset_dw. The paired sampler (sharp[2]) is folded into the combined
    // image-sampler at the same binding by the backend, so we don't emit a separate Sampler resource.
    const AgcShaderSharp* texs = ud->sharp_resource_offset[0];
    if (texs) {
        for (uint16_t slot = 0; slot < ud->sharp_resource_count[0]; slot++) {
            const AgcShaderSharp& s = texs[slot];
            if (s.empty()) continue;
            uint32_t off = s.offset_dw();
            if ((uint64_t)off + 8 > num_user_sgprs) continue;   // T# (8 dwords) must fit in the block
            DecodedImageDescriptor d = decode_image_descriptor(&user_sgprs[off]);
            if (d.base == 0 || d.width == 0 || d.height == 0 ||
                d.width > 16384 || d.height > 16384) continue;  // skip a garbage/degenerate T#
            if (getenv("PROSPER_GFXLOG")) {
                const uint32_t* t = &user_sgprs[off];
                fprintf(stderr, "[t#] %ux%u base=0x%llx tile_mode=%u type=%u fmt=%u | raw: %08x %08x %08x %08x %08x %08x %08x %08x\n",
                        d.width, d.height, (unsigned long long)d.base, d.tile_mode, d.type, d.format,
                        t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
            }
            ShaderResource r;
            r.cls           = ResourceClass::Texture;
            r.format        = DataFormat::Unorm8;   // sampled UI textures are 8-bit UNORM RGBA
            r.num_components = 4;
            r.binding       = binding++;
            r.gpu_addr      = d.base;
            r.width         = d.width;
            r.height        = d.height;
            r.tile_mode     = d.tile_mode;          // so the renderer can auto-detile a GPU-tiled surface
            r.size          = d.width * d.height * 4;
            r.sgpr_base     = user_sgpr_base + off;  // DIRECT provenance key (image_sample SRSRC SGPR)
            r.srt_offset    = 0xFFFFFFFFu;
            table.resources.push_back(r);
        }
    }

    // Vertex buffers (stage 2): Sony "direct" resources — the driver places the V# straight in the
    // user-data SGPRs (contract's DIRECT provenance). Kyty ShaderParseUsage2 usage types: 8 = vertex
    // buffer, 10 = vertex attrib; direct_resource_offset is indexed by usage type and the value is the
    // SGPR index where that V# sits (0xffff = absent). We emit a VertexBuffer keyed by sgpr_base so
    // the recompiler resolves each buffer_load_format_*'s SRSRC directly (no in-shader s_load).
    if (const uint16_t* dro = ud->direct_resource_offset) {
        for (uint16_t type = 0; type < ud->direct_resource_count; type++) {
            if (type != 8 && type != 10) continue;              // vertex buffer / vertex attrib
            uint32_t reg = dro[type];
            if (reg == 0xffff) continue;
            if ((uint64_t)reg + 4 > num_user_sgprs) continue;   // V# (4 dwords) must fit in the block
            DecodedBufferDescriptor d = decode_buffer_descriptor(&user_sgprs[reg]);
            // Plausibility guard: a real vertex-buffer V# has a non-zero base and a sane size. This
            // game's vertex fetch is bindless-dynamic (the V# is s_loaded from a table at a computed
            // offset, not placed directly at this SGPR), so direct_resource_offset here often points at
            // non-descriptor SGPRs -> a garbage decode (e.g. size ~1.4 GB). Skip those rather than emit
            // a bogus binding. (When the dynamic-fetch path is implemented, this direct case still holds
            // for shaders that DO place the V# inline.)
            if (d.base == 0 || d.size_bytes == 0 || d.size_bytes > 0x10000000u) continue;
            ShaderResource r;
            r.cls            = ResourceClass::VertexBuffer;
            r.format         = d.format;          // Float32 for this game (dfmt {4,11,13,14}, nfmt 7)
            r.num_components  = d.num_components;
            r.binding        = binding++;
            r.gpu_addr       = d.base;
            r.size           = d.size_bytes;
            r.stride         = d.stride;
            r.sgpr_base      = reg;               // DIRECT provenance key (SRSRC SGPR index)
            r.srt_offset     = 0xFFFFFFFFu;       // not s_loaded
            table.resources.push_back(r);
        }
    }
    return table;
}

} // namespace prosper::gpu
