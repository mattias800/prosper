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

// The Gen5 T# carries a 9-bit COMBINED IMG_FMT at dword1 bits[28:20] (the GFX10 image-format enum,
// not the GCN dfmt/nfmt split). Values 1..77 share the buffer-format numbering decoded by
// rdna2_buffer_format above; image-only values sit at 128+ — SRGB at 128..130, BC1..BC7 at 169..182 —
// per AMD's published GFX10 register database (mesa src/amd/registers/gfx10-rsrc.json, enum
// GFX10_FORMAT). Anchors: fmt=56 (8_8_8_8_UNORM) is Kyty-confirmed (Texture.cpp get_texture_format,
// gen5 path) AND is the title's live 1920x1080 composite T#; fmt=1 (8_UNORM) is the title's live
// 2048x1024 single-channel atlas. USCALED/SSCALED, 10/11-bit packed, depth, FMASK and video formats
// are deliberately left unmapped until a target needs them (callers log + skip, never assume RGBA8).
// CONFIDENCE: HIGH on 1..77 and the two live anchors, MED on the image-only rows (register-DB-derived,
// no game-observed instance yet).
bool gen5_image_format(uint32_t fmt, Gen5ImageFormatInfo* out) {
    Gen5ImageFormatInfo fi;
    auto plain = [&](DataFormat f, uint32_t n, bool srgb = false) {
        fi.format = f; fi.num_components = n; fi.bytes_per_block = data_format_bytes(f) * n; fi.srgb = srgb;
    };
    auto bcn = [&](DataFormat f, uint32_t n, uint32_t bpb, bool srgb = false) {
        fi.format = f; fi.num_components = n; fi.bytes_per_block = bpb;
        fi.block_width = fi.block_height = 4; fi.srgb = srgb;
    };
    if (fmt >= 1 && fmt <= 77) {                 // shared with the V# buffer-format numbering
        DataFormat f = DataFormat::Unknown; uint32_t n = 0;
        rdna2_buffer_format(fmt, &f, &n);
        if (f != DataFormat::Unknown) plain(f, n);
    } else switch (fmt) {
        case 128: plain(DataFormat::Unorm8, 1, true); break;   // 8_SRGB
        case 129: plain(DataFormat::Unorm8, 2, true); break;   // 8_8_SRGB
        case 130: plain(DataFormat::Unorm8, 4, true); break;   // 8_8_8_8_SRGB
        case 169: bcn(DataFormat::Bc1, 4,  8);        break;   // BC1_UNORM
        case 170: bcn(DataFormat::Bc1, 4,  8, true);  break;   // BC1_SRGB
        case 171: bcn(DataFormat::Bc2, 4, 16);        break;   // BC2_UNORM
        case 172: bcn(DataFormat::Bc2, 4, 16, true);  break;   // BC2_SRGB
        case 173: bcn(DataFormat::Bc3, 4, 16);        break;   // BC3_UNORM
        case 174: bcn(DataFormat::Bc3, 4, 16, true);  break;   // BC3_SRGB
        case 175: case 176: bcn(DataFormat::Bc4, 1,  8); break; // BC4_UNORM/_SNORM (snorm not modeled
        case 177: case 178: bcn(DataFormat::Bc5, 2, 16); break; // BC5_UNORM/_SNORM  yet — skip-only)
        case 179: case 180: bcn(DataFormat::Bc6, 3, 16); break; // BC6_UFLOAT/_SFLOAT
        case 181: bcn(DataFormat::Bc7, 4, 16);        break;   // BC7_UNORM
        case 182: bcn(DataFormat::Bc7, 4, 16, true);  break;   // BC7_SRGB
        default: break;                                         // unmapped -> false
    }
    if (out) *out = fi;
    return fi.format != DataFormat::Unknown;
}

DecodedImageDescriptor decode_image_descriptor(const uint32_t t[8]) {
    DecodedImageDescriptor d;
    d.base      = (((uint64_t)t[0] | ((uint64_t)t[1] << 32)) & 0xFFFFFFFFFFull) << 8;             // Base40
    d.width     = (uint32_t)(((t[1] >> 30) & 0x3u) | (((t[2] >> 0) & 0xFFFu) << 2)) + 1;          // Width5
    d.height    = (uint32_t)((t[2] >> 14) & 0x3FFFu) + 1;                                          // Height5
    d.format    = (t[1] >> 20) & 0x1FFu;                                                           // Format
    d.tile_mode = (t[3] >> 20) & 0x1Fu;                                                            // TileMode (SW_MODE)
    d.type      = (uint8_t)((t[3] >> 28) & 0xFu);                                                  // Type
    d.dst_sel[0] = (uint8_t)((t[3] >> 0) & 0x7u);   // DST_SEL_X (WORD3 [2:0])
    d.dst_sel[1] = (uint8_t)((t[3] >> 3) & 0x7u);   // DST_SEL_Y ([5:3])
    d.dst_sel[2] = (uint8_t)((t[3] >> 6) & 0x7u);   // DST_SEL_Z ([8:6])
    d.dst_sel[3] = (uint8_t)((t[3] >> 9) & 0x7u);   // DST_SEL_W ([11:9])
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
                fprintf(stderr, "[t#] %ux%u base=0x%llx tile_mode=%u type=%u fmt=%u swz=%u,%u,%u,%u | raw: %08x %08x %08x %08x %08x %08x %08x %08x\n",
                        d.width, d.height, (unsigned long long)d.base, d.tile_mode, d.type, d.format,
                        d.dst_sel[0], d.dst_sel[1], d.dst_sel[2], d.dst_sel[3],
                        t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
            }
            // Decode the T#'s real Gen5 IMG_FMT (#65 — was hardcoded Unorm8 x4 / size w*h*4, which
            // mis-samples any non-RGBA8 texture and over-reads a BCn allocation up to 8x). Policy:
            //   * mapped uncompressed  -> emit truthfully (format/components/real byte size);
            //   * mapped BCn           -> recognized but no backend samples blocks yet: log once + SKIP
            //     (a wrong RGBA8 binding samples garbage AND reads w*h*4 past the real allocation);
            //   * unmapped value       -> log once, loudly, + SKIP (never silently assume RGBA8).
            // Skipping mirrors the degenerate-T# guard above: the sampling shader fails to recompile
            // and its draw is dropped, which is diagnosable — garbage pixels are not.
            Gen5ImageFormatInfo fi;
            static bool warned[512] = {};                        // once per 9-bit format value
            if (!gen5_image_format(d.format, &fi)) {
                // Unmapped IMG format. Under PROSPER_RTT, BIND it as RGBA8 anyway so the render-to-texture
                // path can inject the pixels we rendered into this address (the composite that samples a
                // scene color target uses an unmapped packed format, e.g. fmt=36 — skipping it means the
                // RTT cache is never consulted and the frame stays black). Without RTT, keep skipping
                // (a raw RGBA8 read of a real unmapped texture would sample garbage; #65).
                if (!getenv("PROSPER_RTT")) {
                    if (!warned[d.format & 511u]) { warned[d.format & 511u] = true;
                        fprintf(stderr, "[t#] UNMAPPED Gen5 IMG_FMT %u (%ux%u T#) -> skipping texture binding "
                                        "(extend gen5_image_format)\n", d.format, d.width, d.height); }
                    continue;
                }
                fi.format = DataFormat::Unorm8; fi.num_components = 4; fi.bytes_per_block = 4;
                fi.block_width = fi.block_height = 1;
            }
            // Block-compressed: BC1/BC2/BC3 are decoded to RGBA8 on upload (bc_decode). BC4/5/6/7 aren't
            // decoded yet, so keep skipping them (a raw RGBA8 binding would sample garbage + over-read).
            const bool is_bcn = fi.block_width > 1;
            if (is_bcn && fi.format != DataFormat::Bc1 && fi.format != DataFormat::Bc2 &&
                          fi.format != DataFormat::Bc3) {
                if (!warned[d.format & 511u]) { warned[d.format & 511u] = true;
                    fprintf(stderr, "[t#] Gen5 IMG_FMT %u is block-compressed BC4-7 (%ux%u T#) -> decode not "
                                    "wired; skipping texture binding\n", d.format, d.width, d.height); }
                continue;
            }
            ShaderResource r;
            r.cls           = ResourceClass::Texture;
            r.format        = fi.format;
            r.num_components = fi.num_components;
            r.binding       = binding++;
            r.gpu_addr      = d.base;
            r.width         = d.width;
            r.height        = d.height;
            r.tile_mode     = d.tile_mode;          // so the renderer can auto-detile a GPU-tiled surface
            r.swizzle[0] = d.dst_sel[0]; r.swizzle[1] = d.dst_sel[1];
            r.swizzle[2] = d.dst_sel[2]; r.swizzle[3] = d.dst_sel[3];   // T# DST_SEL channel remap (#261)
            r.srgb          = fi.srgb;              // gamma-encoded surface: sample with sRGB->linear (#263)
            if (fi.srgb && getenv("PROSPER_GFXLOG"))
                fprintf(stderr, "[t#] SRGB texture fmt=%u %ux%u (binding %u)\n", d.format, d.width, d.height, r.binding);
            // Backing byte size: block-compressed surfaces store one bytes_per_block unit per 4x4 block
            // (ceil dims); uncompressed store bytes_per_block per texel (fmt=56 -> *4).
            r.size          = is_bcn ? (((d.width + 3) / 4) * ((d.height + 3) / 4) * fi.bytes_per_block)
                                     : (d.width * d.height * fi.bytes_per_block);
            r.sgpr_base     = user_sgpr_base + off;  // DIRECT provenance key (image_sample SRSRC SGPR)
            r.srt_offset    = 0xFFFFFFFFu;
            // Paired sampler S# (sharp[2], same slot): decode its filter + wrap modes so the backend
            // samples the way the game asked (point vs bilinear, wrap vs clamp), instead of a hardcoded
            // LINEAR/clamp. 4-dword SQ_IMG_SAMP: WORD0 has CLAMP_X/Y/Z (bits [2:0]/[5:3]/[8:6]); WORD2
            // has XY_MAG_FILTER [21:20], XY_MIN_FILTER [23:22], MIP_FILTER [27:26] (0 = point/nearest).
            // Absent/garbage sampler -> keep the linear/clamp defaults. CONFIDENCE: HIGH (layout matches
            // GCN/RDNA2 SQ_IMG_SAMP; verified against decoded raw dwords under PROSPER_GFXLOG).
            if (const AgcShaderSharp* samps = ud->sharp_resource_offset[2]) {
                if (slot < ud->sharp_resource_count[2] && !samps[slot].empty()) {
                    uint32_t soff = samps[slot].offset_dw();
                    if ((uint64_t)soff + 4 <= num_user_sgprs) {
                        const uint32_t* sm = &user_sgprs[soff];
                        r.mag_filter  = ((sm[2] >> 20) & 0x3u) ? 1u : 0u;
                        r.min_filter  = ((sm[2] >> 22) & 0x3u) ? 1u : 0u;
                        r.mip_filter  = ((sm[2] >> 26) & 0x3u) ? 1u : 0u;
                        r.addr_uvw[0] = (sm[0] >> 0) & 0x7u;
                        r.addr_uvw[1] = (sm[0] >> 3) & 0x7u;
                        r.addr_uvw[2] = (sm[0] >> 6) & 0x7u;
                        if (getenv("PROSPER_GFXLOG"))
                            fprintf(stderr, "[s#] slot%u mag=%u min=%u mip=%u addr=%u,%u,%u | raw %08x %08x %08x %08x\n",
                                    slot, r.mag_filter, r.min_filter, r.mip_filter,
                                    r.addr_uvw[0], r.addr_uvw[1], r.addr_uvw[2], sm[0], sm[1], sm[2], sm[3]);
                    }
                }
            }
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
            // DIRECT provenance key in SHADER-SGPR space: user_sgpr_base + block index, exactly
            // like the cbuf/texture classes above. `reg` alone is the user-data BLOCK index; the
            // recompiler looks resources up by shader SGPR (by_sgpr_base_cls on the fetch's SRSRC),
            // and vertex buffers only matter in the VS where user_sgpr_base is 8 — the un-based key
            // could never match (or collided with an unrelated SGPR).
            r.sgpr_base      = user_sgpr_base + reg;
            r.srt_offset     = 0xFFFFFFFFu;       // not s_loaded
            table.resources.push_back(r);
        }
    }
    return table;
}

} // namespace prosper::gpu
