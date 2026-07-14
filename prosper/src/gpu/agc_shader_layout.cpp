// agc_shader_layout.cpp — see agc_shader_layout.hpp. V# decode + the front-half resource-table build.
#include "agc_shader_layout.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace prosper::gpu {

// Guest-memory readability probe (defined in gpu_executor.cpp, same lib) — used by the gated
// PROSPER_DUMP_TILERAW diagnostic to avoid a SIGSEGV on a mis-decoded texture base.
bool guest_readable(uint64_t addr, uint32_t bytes);

AgcPixelInputControls derive_agc_pixel_input_controls(const AgcShaderHeader* producer,
                                                      const AgcShaderHeader* pixel) {
    AgcPixelInputControls out;
    const uint32_t producer_count = producer && producer->output_semantics &&
                                    producer->num_output_semantics <= 64
        ? producer->num_output_semantics : 0u;

    // The no-PS form used by sceAgcCreateInterpolantMapping maps the producer's declared outputs
    // directly. Normal rendering always supplies a PS and takes the semantic-matching path below.
    if (!pixel) {
        const uint32_t n = std::min<uint32_t>(producer_count, 32u);
        for (uint32_t i = 0; i < n; ++i) {
            const AgcShaderSemantic& semantic = producer->output_semantics[i];
            const uint32_t slot = semantic.hardware_mapping() < 32u
                ? semantic.hardware_mapping() : i;
            out.controls[i] = slot | (semantic.is_flat_shaded() ? 0x400u : 0u);
            out.valid_mask |= 1u << i;
        }
        return out;
    }

    if (!pixel->input_semantics || pixel->num_input_semantics > 32u) return out;
    for (uint32_t i = 0; i < pixel->num_input_semantics; ++i) {
        const AgcShaderSemantic& input = pixel->input_semantics[i];
        uint32_t control = 0x20u | (input.default_value() << 8); // constant if no output matches
        for (uint32_t j = 0; j < producer_count; ++j) {
            const AgcShaderSemantic& output = producer->output_semantics[j];
            if (output.semantic() == input.semantic() && output.hardware_mapping() < 32u) {
                control = output.hardware_mapping();
                break;
            }
        }
        if (input.is_flat_shaded()) control |= 0x400u;
        out.controls[i] = control;
        out.valid_mask |= 1u << i;
    }
    return out;
}

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
    auto bcn = [&](DataFormat f, uint32_t n, uint32_t bpb, bool srgb = false, bool snorm = false) {
        fi.format = f; fi.num_components = n; fi.bytes_per_block = bpb;
        fi.block_width = fi.block_height = 4; fi.srgb = srgb; fi.snorm = snorm;
    };
    if (fmt == 36) {
        // GFX10_FORMAT_10_11_11_FLOAT — packed 32-bit R11G11B10F (R=bits[10:0], G=[21:11],
        // B=[31:22], unsigned small floats, no alpha). UE4's scene-color render-target format;
        // DOLL's final composite samples it at 3840x2160/1920x1080 (#294). Mapped HERE (not in
        // rdna2_buffer_format) on purpose: as a texture the upload path unpacks it to RGBA8, but
        // the vertex-fetch recompiler has no packed-component conversion, so a V# with this
        // format must keep falling back rather than mis-convert. CONFIDENCE: HIGH (value from
        // AMD's GFX10 register DB; live DOLL T#s confirmed fmt=36 at scene-color dimensions).
        fi.format = DataFormat::Float10_11_11; fi.num_components = 3; fi.bytes_per_block = 4;
    } else if (fmt == 50) {
        // GFX10_FORMAT_2_10_10_10_UNORM — Mesa maps a logical R10G10B10A2 format to this enum.
        // DOLL's final composite PS samples a live 32x32 T# in this format; leaving it unmapped drops
        // the binding and rejects the entire scene-color writer. It is one packed 4-byte texel and the
        // live renderer expands it to RGBA8 before upload. CONFIDENCE: HIGH (Mesa DB + live T#).
        fi.format = DataFormat::Unorm2_10_10_10; fi.num_components = 4; fi.bytes_per_block = 4;
    } else if (fmt >= 1 && fmt <= 77) {          // shared with the V# buffer-format numbering
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
        case 175: bcn(DataFormat::Bc4, 1,  8);                    break; // BC4_UNORM
        case 176: bcn(DataFormat::Bc4, 1,  8, false, true);       break; // BC4_SNORM (skip-only: signed)
        case 177: bcn(DataFormat::Bc5, 2, 16);                    break; // BC5_UNORM
        case 178: bcn(DataFormat::Bc5, 2, 16, false, true);       break; // BC5_SNORM (skip-only: signed)
        case 179: bcn(DataFormat::Bc6, 3, 16);              break; // BC6H_UFLOAT (decoded, #273)
        case 180: bcn(DataFormat::Bc6, 3, 16, false, true); break; // BC6H_SFLOAT (snorm flag = signed;
                                                                   // still skipped: no live example)
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

uint32_t image_type_to_dim(uint8_t type) {
    return type >= 8 && type <= 15 ? static_cast<uint32_t>(type - 8) : 1u;
}

ShaderResourceTable build_shader_resources(const AgcShaderHeader& shdr,
                                           const uint32_t* user_sgprs, uint32_t num_user_sgprs,
                                           uint32_t user_sgpr_base) {
    ShaderResourceTable table;
    const AgcShaderUserData* ud = shdr.user_data;
    if (!ud || !user_sgprs) return table;
    // The user_data pointer can be non-null yet point at UNMAPPED guest memory — e.g. an offline
    // capsule capture (#713) realizing a reconstructed per-draw state whose PS user_data block the
    // guest has since freed/reused. Dereferencing ud->... then SIGSEGVs and kills the capture of any
    // scene that contains such a draw. Verify the struct is actually mapped before reading it; if not,
    // skip this stage's resources (empty table) rather than fault. Live rendering (memory still mapped)
    // is byte-identical. CONFIDENCE: HIGH (pinpointed: build_shader_resources faults on the first
    // ud field read for PPSA02664 title draw 2's PS at capture time).
    if (!guest_readable((uint64_t)(uintptr_t)ud, sizeof(AgcShaderUserData))) return table;
    // The sharp/direct resource-offset arrays are further guest pointers in the same spill area and can
    // likewise be non-null-but-unmapped for a reconstructed capture state (#713). Verify each is mapped
    // for its declared entry count before iterating, else skip that category (don't fault).
    auto arr_ok = [](const void* p, size_t count, size_t elem) {
        if (!p || !count) return false;
        const size_t bytes = count * elem;
        if (bytes > (1u << 20)) return false;   // insane count -> treat as absent
        return guest_readable((uint64_t)(uintptr_t)p, (uint32_t)bytes);
    };

    uint32_t binding = 0;

    // Extended User Data (EUD): descriptors whose offset_dw is beyond the user-SGPR block live in a guest
    // memory spill area. Its base pointer sits in the user SGPR named by direct_resource_offset[5] (usage
    // type 5) — confirmed against the shader's own `s_load_dwordx4 sX, s[EUD:EUD+1], <off>`. A sharp at
    // offset_dw >= num_user_sgprs is at EUD_base + (offset_dw - num_user_sgprs)*4, and the shader loads it
    // with exactly that immediate — so its recompiler provenance key (srt_offset) is (off-nsgpr)*4, which
    // the s_load-tracked s_buffer_load resolves via by_srt_offset. This is why the game's TEXT was invisible:
    // its pixel shader's colour constant buffers are EUD-resident (offset_dw 40/44) and were being skipped,
    // so the text colour read as (0,0,0,0) = transparent black. #257.
    uint64_t eud_base = 0;
    if (const uint16_t* dro = ud->direct_resource_offset)
        if (arr_ok(dro, ud->direct_resource_count, sizeof(uint16_t)) &&
            ud->direct_resource_count > 5 && dro[5] != 0xffff && (uint64_t)dro[5] + 2 <= num_user_sgprs) {
            uint64_t p = (uint64_t)user_sgprs[dro[5]] | ((uint64_t)user_sgprs[dro[5] + 1] << 32);
            if (p > 0x10000 && p < 0x0000800000000000ull) eud_base = p;   // plausible guest pointer
        }
    // Fetch a sharp's N descriptor dwords from either the SGPR block or the EUD; returns the srt_offset
    // provenance key (or UINT32_MAX if unreadable). Keeps the in-SGPR path byte-identical.
    auto load_sharp = [&](uint32_t off, uint32_t n, uint32_t* buf) -> uint32_t {
        if ((uint64_t)off + n <= num_user_sgprs) {
            for (uint32_t i = 0; i < n; i++) buf[i] = user_sgprs[off + i];
            return off * 4;                                   // byte offset within user_data
        }
        if (!eud_base) return 0xFFFFFFFFu;
        // Bounds-check the EUD read (#375): offset_dw is a 15-bit field (up to ~128 KB past eud_base),
        // so an out-of-range cbuf slot would dereference unmapped guest memory and SIGSEGV the front-half
        // build on EVERY draw that binds this shader. Reject unless the read fits the guest-declared EUD
        // size AND is actually mapped — mirroring the guarded in-SGPR path above and the T# path below
        // (which already calls guest_readable). eud_size_dw was decoded but never used to bound a read.
        const uint64_t eoff = (uint64_t)(off - num_user_sgprs);
        if (ud->eud_size_dw && eoff + n > ud->eud_size_dw) return 0xFFFFFFFFu;   // beyond declared EUD
        if (!guest_readable(eud_base + eoff * 4, n * 4))    return 0xFFFFFFFFu;   // unmapped guest memory
        const uint32_t* src = (const uint32_t*)(uintptr_t)(eud_base + eoff * 4);
        for (uint32_t i = 0; i < n; i++) buf[i] = src[i];
        return (uint32_t)(eoff * 4);                          // matches the shader's s_load immediate
    };

    // Constant buffers: sharp_resource_offset[3] (storage-as-constant). Each slot's offset_dw points at
    // a 4-dword V# in the user-data SGPR block OR the EUD (see above).
    const AgcShaderSharp* cbufs = ud->sharp_resource_offset[3];
    if (arr_ok(cbufs, ud->sharp_resource_count[3], sizeof(AgcShaderSharp))) {
        for (uint16_t slot = 0; slot < ud->sharp_resource_count[3]; slot++) {
            const AgcShaderSharp& s = cbufs[slot];
            if (s.empty()) continue;
            uint32_t off = s.offset_dw();
            uint32_t vv[4]; uint32_t srt = load_sharp(off, 4, vv);
            if (srt == 0xFFFFFFFFu) continue;                 // unreadable (EUD absent / out of block)
            DecodedBufferDescriptor d = decode_buffer_descriptor(vv);
            ShaderResource r;
            r.cls            = ResourceClass::ConstantBuffer;
            r.format         = d.format;
            r.num_components  = d.num_components;
            r.binding        = binding++;
            r.gpu_addr       = d.base;
            r.size           = d.size_bytes;
            r.stride         = d.stride;
            // Provenance, mirroring the #382 texture fix (#453). An IN-SGPR cbuf resolves DIRECTLY —
            // the recompiler matches its s_buffer_load SBASE to sgpr_base (by_sgpr_base_cls); its V# is
            // never s_loaded, so SBASE carries no sreg_srt tag and by_srt_offset is never consulted for
            // it. Setting BOTH keys made the in-SGPR key (off*4) collide with an EUD cbuf at the same
            // spill offset (eoff*4) under by_srt_offset (first-match-wins) -> wrong uniform block bound.
            // So set ONLY the applicable key: EUD -> srt_offset (INDIRECT, == the s_load immediate),
            // in-SGPR -> sgpr_base (DIRECT); null the other so a stray lookup can't false-match.
            const bool cbuf_in_eud = (uint64_t)off + 4 > num_user_sgprs;
            r.srt_offset = cbuf_in_eud ? srt : 0xFFFFFFFFu;
            r.sgpr_base  = cbuf_in_eud ? 0xFFFFFFFFu : (user_sgpr_base + off);
            table.resources.push_back(r);
        }
    }

    // Textures: sharp_resource_offset[0] (textures2D). Each slot's offset_dw points at an 8-dword T#
    // in the user-data SGPR block. The PS reads it directly in SGPRs (image_sample SRSRC), so DIRECT
    // provenance: sgpr_base = offset_dw. The paired sampler (sharp[2]) is folded into the combined
    // image-sampler at the same binding by the backend, so we don't emit a separate Sampler resource.
    const AgcShaderSharp* texs = ud->sharp_resource_offset[0];
    if (arr_ok(texs, ud->sharp_resource_count[0], sizeof(AgcShaderSharp))) {
        for (uint16_t slot = 0; slot < ud->sharp_resource_count[0]; slot++) {
            const AgcShaderSharp& s = texs[slot];
            if (s.empty()) continue;
            uint32_t off = s.offset_dw();
            // A T# may live in the user-SGPR block OR spill into the EUD, exactly like the cbuf path
            // (#257/#382) — the old hard `continue` for off+8 > num_user_sgprs silently DROPPED any
            // EUD-resident texture, so its sampling draw rendered untextured. Fetch the 8 dwords via
            // load_sharp (bounds-checked; copies verbatim from the SGPR block for the in-block case, so
            // that path stays byte-identical) and decode from that buffer.
            uint32_t tv[8]; uint32_t tsrt = load_sharp(off, 8, tv);
            if (tsrt == 0xFFFFFFFFu) continue;                  // out of block / EUD absent / unreadable
            const bool tex_in_eud = (uint64_t)off + 8 > num_user_sgprs;
            DecodedImageDescriptor d = decode_image_descriptor(tv);
            if (d.base == 0 || d.width == 0 || d.height == 0 ||
                d.width > 16384 || d.height > 16384) continue;  // skip a garbage/degenerate T#
            // PROSPER_DUMP_TILERAW (issue #282 derivation): dump the raw guest texel bytes of a tiled
            // texture once per address, so its GPU tile swizzle can be reversed offline (coherence
            // scoring). Fires at T#-decode time (every draw's stage build), so it captures textures even
            // on runs whose intermittent content never reaches the render backend. Guest memory is
            // 1:1-mapped; the size is the tiled element footprint (BCn: (w/4)*(h/4)*block_bytes).
            if (getenv("PROSPER_DUMP_TILERAW") && d.base > 0x10000 && d.tile_mode != 0) {
                static bool tseen[1u << 12] = {};                  // dedupe by low base bits
                uint32_t key = (uint32_t)((d.base >> 12) & 0xfffu);
                if (!tseen[key]) {
                    Gen5ImageFormatInfo tf;
                    bool done = true;                              // default: mark seen (don't retry)
                    if (gen5_image_format(d.format, &tf)) {
                        uint32_t bpb = tf.bytes_per_block, bw = tf.block_width, bh = tf.block_height;
                        size_t nb = (size_t)((d.width + bw - 1) / bw) * ((d.height + bh - 1) / bh) * bpb;
                        if (nb && nb <= (32u << 20) && guest_readable(d.base, (uint32_t)nb)) {
                            // Only mark seen once the surface has CONTENT: a render target sampled
                            // before anything drew into it dumps all-zero and (deduped) would never
                            // be re-captured — keep retrying until nonzero bytes appear (#288).
                            const uint8_t* pb8 = (const uint8_t*)(uintptr_t)d.base;
                            size_t probe = nb < 65536 ? nb : 65536;
                            bool has_content = false;
                            for (size_t i = 0; i < probe; i += 64) if (pb8[i]) { has_content = true; break; }
                            done = has_content;
                            if (has_content) {
                                const char* dd = getenv("PROSPER_FRAME_DIR"); char fn[512];
                                snprintf(fn, sizeof fn, "%s/tileraw_%ux%u_bpb%u_tm%u_fmt%u_%llx.bin",
                                         dd ? dd : ".", d.width, d.height, bpb, d.tile_mode, d.format,
                                         (unsigned long long)d.base);
                                if (FILE* bf = fopen(fn, "wb")) {
                                    fwrite((const void*)(uintptr_t)d.base, 1, nb, bf); fclose(bf);
                                    fprintf(stderr, "[tileraw] %s (%zu)\n", fn, nb);
                                }
                            }
                        }
                    }
                    tseen[key] = done;
                }
            }
            if (getenv("PROSPER_GFXLOG")) {
                const uint32_t* t = tv;   // the fetched T# (SGPR block or EUD spill)
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
                // The normal per-target renderer can supply an unmapped image through RTT injection,
                // so bind it as RGBA8. The legacy single-target diagnostic keeps the old skip policy;
                // a raw RGBA8 read of an unknown guest format would sample garbage (#65/#294).
                static const bool rtt_bind = getenv("PROSPER_RTT") != nullptr ||
                                             getenv("PROSPER_RTT_PERTARGET") != nullptr;
                if (!rtt_bind) {
                    if (!warned[d.format & 511u]) { warned[d.format & 511u] = true;
                        fprintf(stderr, "[t#] UNMAPPED Gen5 IMG_FMT %u (%ux%u T#) -> skipping texture binding "
                                        "(extend gen5_image_format)\n", d.format, d.width, d.height); }
                    continue;
                }
                fi.format = DataFormat::Unorm8; fi.num_components = 4; fi.bytes_per_block = 4;
                fi.block_width = fi.block_height = 1;
            }
            // Block-compressed: BC1/2/3/4/5/7 AND BC6H_UF16 are decoded to RGBA8 on upload
            // (bc_decode, #290/#273 — BC6H clamps >1.0 HDR energy like the fp16 path). Still
            // skipped: SIGNED variants (BC4/BC5 SNORM, BC6H SF16 — the UNORM8 upload can't carry
            // signed samples; a remapped decode would be numerically wrong).
            const bool is_bcn = fi.block_width > 1;
            if (is_bcn && fi.snorm) {
                if (!warned[d.format & 511u]) { warned[d.format & 511u] = true;
                    fprintf(stderr, "[t#] Gen5 IMG_FMT %u is %s (%ux%u T#) -> decode not "
                                    "wired; skipping texture binding\n", d.format,
                            fi.format == DataFormat::Bc6 ? "BC6H SF16" : "SNORM BCn",
                            d.width, d.height); }
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
            // T# TYPE -> the MIMG dim convention (GFX10 SQ_RSRC_IMG: 8=1D, 9=2D, 10=3D, 11=CUBE,
            // 12=1D_ARRAY, 13=2D_ARRAY). A CUBE resource (img_dim 3) uploads as six faces stacked
            // vertically in one 2D image (#273); everything else keeps the 2D default.
            r.img_dim       = image_type_to_dim(d.type);
            r.srgb          = fi.srgb;              // gamma-encoded surface: sample with sRGB->linear (#263)
            if (fi.srgb && getenv("PROSPER_GFXLOG"))
                fprintf(stderr, "[t#] SRGB texture fmt=%u %ux%u (binding %u)\n", d.format, d.width, d.height, r.binding);
            // Backing byte size: block-compressed surfaces store one bytes_per_block unit per 4x4 block
            // (ceil dims); uncompressed store bytes_per_block per texel (fmt=56 -> *4).
            r.size          = is_bcn ? (((d.width + 3) / 4) * ((d.height + 3) / 4) * fi.bytes_per_block)
                                     : (d.width * d.height * fi.bytes_per_block);
            // SGPR-resident T#: DIRECT provenance (image_sample SRSRC SGPR). EUD-resident T#: INDIRECT
            // (the s_load immediate = tsrt), and sgpr_base must be invalid so a stray by_sgpr_base for an
            // unrelated op reading that (bogus, out-of-file) SGPR number can't spuriously match it (#382).
            r.sgpr_base     = tex_in_eud ? 0xFFFFFFFFu : (user_sgpr_base + off);
            r.srt_offset    = tex_in_eud ? tsrt : 0xFFFFFFFFu;
            // Paired sampler S# (sharp[2], same slot): decode its filter + wrap modes so the backend
            // samples the way the game asked (point vs bilinear, wrap vs clamp), instead of a hardcoded
            // LINEAR/clamp. 4-dword SQ_IMG_SAMP: WORD0 has CLAMP_X/Y/Z (bits [2:0]/[5:3]/[8:6]); WORD2
            // has XY_MAG_FILTER [21:20], XY_MIN_FILTER [23:22], MIP_FILTER [27:26] (0 = point/nearest).
            // Absent/garbage sampler -> keep the linear/clamp defaults. CONFIDENCE: HIGH (layout matches
            // GCN/RDNA2 SQ_IMG_SAMP; verified against decoded raw dwords under PROSPER_GFXLOG).
            const AgcShaderSharp* samps = ud->sharp_resource_offset[2];
            if (arr_ok(samps, ud->sharp_resource_count[2], sizeof(AgcShaderSharp))) {
                if (slot < ud->sharp_resource_count[2] && !samps[slot].empty()) {
                    uint32_t soff = samps[slot].offset_dw();
                    // The paired S# may live in the user-SGPR block OR spill to the EUD alongside its T#
                    // (#451) — fetch it via the same bounds-checked load_sharp used for the T# above,
                    // instead of indexing user_sgprs directly (which dropped EUD-resident samplers, so the
                    // texture silently reverted to the default LINEAR/clamp sampler for exactly the
                    // EUD-resident textures #257 added — a point/wrap sampler read as bilinear/clamped).
                    // load_sharp copies verbatim from the SGPR block for the in-block case, byte-identical.
                    uint32_t sm_buf[4];
                    if (load_sharp(soff, 4, sm_buf) != 0xFFFFFFFFu) {
                        const uint32_t* sm = sm_buf;
                        r.mag_filter  = ((sm[2] >> 20) & 0x3u) ? 1u : 0u;
                        r.min_filter  = ((sm[2] >> 22) & 0x3u) ? 1u : 0u;
                        r.mip_filter  = ((sm[2] >> 26) & 0x3u) ? 1u : 0u;
                        r.addr_uvw[0] = (sm[0] >> 0) & 0x7u;
                        r.addr_uvw[1] = (sm[0] >> 3) & 0x7u;
                        r.addr_uvw[2] = (sm[0] >> 6) & 0x7u;
                        // Remaining SQ_IMG_SAMP fields (#262). WORD0: MAX_ANISO_RATIO[11:9],
                        // DEPTH_COMPARE_FUNC[14:12], FORCE_UNNORMALIZED[15]. WORD1: MIN_LOD[11:0],
                        // MAX_LOD[23:12] (u4.8). WORD2: LOD_BIAS[13:0] (s5.8, signed). WORD3:
                        // BORDER_COLOR_TYPE[31:30]. CONFIDENCE: HIGH (standard GCN/RDNA2 SQ_IMG_SAMP layout).
                        r.max_aniso_ratio    = (sm[0] >> 9)  & 0x7u;
                        r.depth_compare_func = (sm[0] >> 12) & 0x7u;
                        r.unnormalized       = (sm[0] >> 15) & 0x1u;
                        r.min_lod            = (float)( sm[1]        & 0xFFFu) / 256.0f;
                        r.max_lod            = (float)((sm[1] >> 12) & 0xFFFu) / 256.0f;
                        int32_t bias14       = (int32_t)(sm[2] & 0x3FFFu);
                        if (bias14 & 0x2000) bias14 -= 0x4000;           // sign-extend the 14-bit s5.8
                        r.lod_bias           = (float)bias14 / 256.0f;
                        r.border_color_type  = (sm[3] >> 30) & 0x3u;
                        if (getenv("PROSPER_GFXLOG"))
                            fprintf(stderr, "[s#] slot%u mag=%u min=%u mip=%u addr=%u,%u,%u | aniso=%u cmp=%u unnorm=%u "
                                    "lod=[%.3f,%.3f] bias=%.3f border=%u | raw %08x %08x %08x %08x\n",
                                    slot, r.mag_filter, r.min_filter, r.mip_filter,
                                    r.addr_uvw[0], r.addr_uvw[1], r.addr_uvw[2],
                                    r.max_aniso_ratio, r.depth_compare_func, r.unnormalized,
                                    r.min_lod, r.max_lod, r.lod_bias, r.border_color_type,
                                    sm[0], sm[1], sm[2], sm[3]);
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
    // Dead Cells compute shaders use direct type 1 for an inline V# at COMPUTE_USER_DATA_0
    // (live descriptor: 16-byte stride, 0x1fe000 records, plausible Base48). Model it as a
    // ConstantBuffer-class resource: both that class and VertexBuffer lower to a Vulkan storage
    // buffer, while ConstantBuffer also lets scalar buffer loads resolve by their direct SBASE.
    // Restrict this Gen5 observation to compute shaders; other stages have no exercised contract.
    const uint16_t* dro = ud->direct_resource_offset;
    if (arr_ok(dro, ud->direct_resource_count, sizeof(uint16_t))) {
        for (uint16_t type = 0; type < ud->direct_resource_count; type++) {
            const bool compute_buffer = shdr.type == 0 && type == 1;
            if (!compute_buffer && type != 8 && type != 10) continue;
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
            // The metadata slot can name ordinary user data when the real V# is bindless. A mapped
            // buffer FORMAT is part of the direct-sharp contract and provides a stronger discriminator
            // than the size ceiling alone: DOLL has a live false positive just below 256 MiB whose four
            // pointer/data dwords decode with FORMAT=0. Dynamic descriptors whose format is patched by
            // shader ALU are recovered by resolve_dynamic_fetch instead of this metadata-only path.
            if (d.base == 0 || d.size_bytes == 0 || d.size_bytes > 0x10000000u ||
                d.format == DataFormat::Unknown || !d.num_components) continue;
            ShaderResource r;
            r.cls            = compute_buffer ? ResourceClass::ConstantBuffer : ResourceClass::VertexBuffer;
            r.format         = d.format;
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

    // Additional directly-placed COMPUTE buffers (#566). Gen5 compute shaders put buffer V#s straight
    // in the user-SGPR block and reference them by SRSRC, but the shader's resource-usage metadata only
    // declares the FIRST (direct_resource_offset type 1 -> reg 0, handled above). A format-copy compute
    // — Dead Cells: `buffer_load_format_xyzw v, vIDX, s[0:3]` then `buffer_store_format_xyzw v, vIDX,
    // s[4:7]` — therefore had its STORE destination V# (at s4) absent from the table, so the store's
    // SRSRC resolved to nothing and the whole shader was rejected (`[mubuf-unresolved] srsrc=s4`). Its
    // dispatch was then dropped, losing the scene-composition pass. Expose every remaining 4-dword-
    // aligned user-SGPR slot that decodes to a plausible buffer V#, keyed by sgpr_base, so the recompiler
    // resolves the SRSRC via by_sgpr_base. The SAME plausibility guard used for the type-1/vertex buffers
    // rejects leftover non-descriptor SGPRs: a real V# has a mapped base and a sane size, while stale
    // registers decode as base 0 or the ~0xFFFFFFFF max size. In the exercised Dead Cells frame this
    // emits ONLY the real copy destinations; every other compute shader's trailing slots are filtered.
    // An unreferenced extra resource is harmless — the recompiler binds only the SRSRCs it actually uses.
    // CONFIDENCE: HIGH (live-verified: dispatch 0x401aec800 realizes and the dropped pass returns).
    if (shdr.type == 0) {
        for (uint32_t reg = 0; reg + 4 <= num_user_sgprs; reg += 4) {
            const uint32_t sgpr = user_sgpr_base + reg;
            bool covered = false;
            for (const ShaderResource& e : table.resources)
                if (e.sgpr_base == sgpr) { covered = true; break; }
            if (covered) continue;
            DecodedBufferDescriptor d = decode_buffer_descriptor(&user_sgprs[reg]);
            if (d.base <= 0x10000 || d.size_bytes == 0 || d.size_bytes > 0x10000000u ||
                d.format == DataFormat::Unknown || !d.num_components) continue;
            ShaderResource r;
            r.cls            = ResourceClass::ConstantBuffer;   // storage buffer (readable + writable)
            r.format         = d.format;
            r.num_components  = d.num_components;
            r.binding        = binding++;
            r.gpu_addr       = d.base;
            r.size           = d.size_bytes;
            r.stride         = d.stride;
            r.sgpr_base      = sgpr;
            r.srt_offset     = 0xFFFFFFFFu;      // direct (not s_loaded)
            table.resources.push_back(r);
        }
    }
    return table;
}

} // namespace prosper::gpu
