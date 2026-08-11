// rdna2_decode.cpp — see rdna2_decode.hpp.
#include "rdna2_decode.hpp"

#include <algorithm>

namespace prosper::gpu {

namespace {
constexpr uint32_t LITERAL = 0xFFu;        // operand field value meaning "32-bit literal follows"
constexpr uint32_t S_ENDPGM = 0xBF810000u;

// A source operand == 255 selects an inline literal constant (one extra dword).
bool sop_has_literal(uint32_t w, int nsrc) {
    if ((w & 0xFFu) == LITERAL) return true;                       // ssrc0
    if (nsrc >= 2 && ((w >> 8) & 0xFFu) == LITERAL) return true;   // ssrc1
    return false;
}
bool vop_has_literal(uint32_t w) { return (w & 0x1FFu) == LITERAL; }   // src0 is 9 bits
}  // namespace

float inline_float_value(uint32_t code) {
    switch (code) {
        case 240: return 0.5f;  case 241: return -0.5f;
        case 242: return 1.0f;  case 243: return -1.0f;
        case 244: return 2.0f;  case 245: return -2.0f;
        case 246: return 4.0f;  case 247: return -4.0f;
        case 248: return 0.15915494f;   // 1/(2*pi)
        default:  return 0.0f;
    }
}

Operand decode_src_field(uint32_t f) {
    if (f >= 256) return {OperandKind::VGPR, (int32_t)(f - 256)};            // 256..511 = VGPR 0..255
    if (f <= 105) return {OperandKind::SGPR, (int32_t)f};                    // 0..105 = SGPR
    if (f == 128) return {OperandKind::InlineInt, 0};
    if (f >= 129 && f <= 192) return {OperandKind::InlineInt, (int32_t)f - 128};   // +1..+64
    if (f >= 193 && f <= 208) return {OperandKind::InlineInt, 192 - (int32_t)f};   // -1..-16
    if (f >= 240 && f <= 248) return {OperandKind::InlineFloat, (int32_t)f};       // inline floats
    if (f == 255) return {OperandKind::Literal, 0};
    return {OperandKind::Special, (int32_t)f};   // VCC_LO/HI, EXEC, M0, null, ttmp, ...
}

// VOPC lays out its 256 opcodes as alternating 16-wide cmp / cmpx blocks, so every `v_cmpx_*` sits
// at exactly its `v_cmp_*` counterpart + 0x10 and the base is recovered by subtracting it. The full
// gfx1030 map, derived by disassembling all 256 VOPC e32 encodings with
// `llvm-mc -arch=amdgcn -mcpu=gfx1030 -disassemble` (#2120):
//
//   0x00-0x0f v_cmp_*_f32      | 0x10-0x1f v_cmpx_*_f32
//   0x20-0x2f v_cmp_*_f64      | 0x30-0x3f v_cmpx_*_f64
//   0x40-0x7f  --- not compare opcodes at all: 64 invalid encodings ---
//   0x80-0x87 v_cmp_*_i32      | 0x90-0x97 v_cmpx_*_i32
//   0x88      v_cmp_class_f32  | 0x98      v_cmpx_class_f32
//   0x89-0x8e v_cmp_*_i16      | 0x99-0x9e v_cmpx_*_i16
//   0x8f      v_cmp_class_f16  | 0x9f      v_cmpx_class_f16
//   0xa0-0xa7 v_cmp_*_i64      | 0xb0-0xb7 v_cmpx_*_i64
//   0xa8      v_cmp_class_f64  | 0xb8      v_cmpx_class_f64
//   0xa9-0xae v_cmp_*_u16      | 0xb9-0xbe v_cmpx_*_u16
//   0xaf      INVALID          | 0xbf      INVALID
//   0xc0-0xc7 v_cmp_*_u32      | 0xd0-0xd7 v_cmpx_*_u32
//   0xc8-0xcf v_cmp_*_f16      | 0xd8-0xdf v_cmpx_*_f16
//   0xe0-0xe7 v_cmp_*_u64      | 0xf0-0xf7 v_cmpx_*_u64
//   0xe8-0xef v_cmp_*_f16      | 0xf8-0xff v_cmpx_*_f16   (the second half of the 16-op f16 set)
//
// The 0xaf / 0xbf holes are real and confirmed twice over: llvm-mc rejects both as invalid
// encodings, and the ISA guide's own numbered opcode table SKIPS 175 and 191 (it runs 174 -> 176
// and 190 -> 192). Note that the ISA's *summary* tables 58 and 60 disagree with all of this — they
// place v_cmpx_*_f32 at 0x50-0x5f and v_cmpx_*_i16 at 0xb0-0xb7, and they list V_CMPS/V_CMPSX
// opcodes that do not exist on RDNA2 at all. Those two tables are stale GCN-era boilerplate; the
// per-opcode table 61 is the one that matches hardware and llvm-mc. Do not "correct" this map back
// to them.
//
// This lives here, next to the decode tables, because it is a property of the ENCODING rather than
// of the translation: both the decoder's SDWA admission and the recompiler's EXEC/mask bookkeeping
// must agree on it, and they did not (#2120 — the decoder listed three of the six windows, so a
// `v_cmpx_*_u16` SDWA packet rejected while its plain e32 form worked).
bool vopc_is_cmpx(uint32_t opcode) {
    return (opcode >= 0x10u && opcode <= 0x1Fu) ||   // f32
           (opcode >= 0x30u && opcode <= 0x3Fu) ||   // f64
           (opcode >= 0x90u && opcode <= 0x9Fu) ||   // i32 / class_f32 / i16 / class_f16
           (opcode >= 0xB0u && opcode <= 0xBEu) ||   // i64 / class_f64 / u16   (0xbf is INVALID)
           (opcode >= 0xD0u && opcode <= 0xDFu) ||   // u32 / f16
           (opcode >= 0xF0u && opcode <= 0xFFu);     // u64 / f16
}

bool rdna2_instruction_may_change_exec(const Rdna2Inst& in) {
    if (in.fmt == Rdna2Format::VOPC && vopc_is_cmpx(in.opcode)) return true;
    // saveexec writes EXEC implicitly while its explicit destination receives the previous mask.
    if (in.fmt == Rdna2Format::SOP1 &&
        ((in.opcode >= 0x24u && in.opcode <= 0x2bu) ||
         in.opcode == 0x37u || in.opcode == 0x38u ||
         in.opcode == 0x3cu || in.opcode == 0x40u || in.opcode == 0x44u))
        return true;
    auto is_exec = [](const Operand& operand) {
        return (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
               (operand.value == 126 || operand.value == 127);
    };
    return is_exec(in.dst) || is_exec(in.sdst);
}

static uint32_t mtbuf_vdata_dwords(const Rdna2Inst& in) {
    if (in.opcode > 15u) return 0;
    const uint32_t components = (in.opcode & 3u) + 1u;
    return in.opcode >= 8u ? (components + 1u) / 2u : components;
}

static uint32_t mimg_vdata_dwords(const Rdna2Inst& in) {
    uint32_t components = 0;
    if (in.opcode == 0x47u || in.opcode == 0x57u) {
        components = 4u; // gather4
    } else {
        for (uint32_t component = 0; component < 4; ++component)
            components += (in.mimg_dmask >> component) & 1u;
        if (!components) components = 4u; // unknown/empty mask: account conservatively.
    }
    return in.mimg_d16 ? (components + 1u) / 2u : components;
}

uint32_t rdna2_vgpr_write_count(const Rdna2Inst& in) {
    if (in.dst.kind != OperandKind::VGPR) return 0;
    switch (in.fmt) {
        case Rdna2Format::VOP1:
            return in.opcode == 0x02u ? 0u : 1u; // v_readfirstlane writes the decoded SGPR.
        case Rdna2Format::VOP2:
        case Rdna2Format::VOP3P:
            return 1;
        case Rdna2Format::VOP3:
            if (in.opcode == 0x360u) return 0; // v_readlane_b32 writes an SGPR.
            // v_div_scale_f64 and the two integer MAD64 forms write a VGPR pair.
            return in.opcode == 0x16eu || in.opcode == 0x176u || in.opcode == 0x177u ? 2u : 1u;
        case Rdna2Format::DS:
            // Keep this aligned with the DS result opcodes admitted by the emitter. Other DS
            // encodings in that subset are stores or no-return atomics whose VDST field is a source.
            if (in.opcode == 0x35u || in.opcode == 0x36u || in.opcode == 0x3du ||
                in.opcode == 0x3eu || in.opcode == 0xb1u || in.opcode == 0x20u ||
                in.opcode == 0x2du)
                return 1;
            if (in.opcode == 0x37u || in.opcode == 0x76u) return 2;
            if (in.opcode == 0xfeu) return 3;
            if (in.opcode == 0x77u || in.opcode == 0xffu) return 4;
            return 0;
        case Rdna2Format::MUBUF:
            // Format and raw stores read VDATA. Loads use the gfx10 ordering where x3 follows x4.
            if ((in.opcode >= 0x04u && in.opcode <= 0x07u) ||
                (in.opcode >= 0x1cu && in.opcode <= 0x1fu))
                return 0;
            if (in.opcode <= 0x03u) return in.opcode + 1u;
            if (in.opcode >= 0x08u && in.opcode <= 0x0cu) return 1;
            if (in.opcode == 0x0du) return 2;
            if (in.opcode == 0x0eu) return 4;
            if (in.opcode == 0x0fu) return 3;
            // Supported return-value atomics are one dword. Unknown buffer operations remain a
            // conservative one-register clobber, matching the prior analysis contract.
            return 1;
        case Rdna2Format::MTBUF:
            if (in.opcode <= 3u || (in.opcode >= 8u && in.opcode <= 11u))
                return mtbuf_vdata_dwords(in); // ordinary and packed-D16 typed loads
            return 0; // typed stores read VDATA; TFE's trailing status is handled separately.
        case Rdna2Format::FLAT:
            switch (in.opcode) {
                case 0x08u: case 0x09u: case 0x0au: case 0x0bu: case 0x0cu: return 1;
                case 0x0du: return 2;
                case 0x0eu: return 4;
                case 0x0fu: return 3;
                default: return 0; // audited store/deferred forms do not produce a VGPR result.
            }
        case Rdna2Format::MIMG: {
            if (in.opcode == 0x08u || in.opcode == 0x09u) return 0; // IMAGE_STORE[_MIP]
            return mimg_vdata_dwords(in);
        }
        case Rdna2Format::VINTRP:
            return 1;
        default:
            return 0;
    }
}

int rdna2_tfe_status_vgpr(const Rdna2Inst& in) {
    if (in.dst.kind != OperandKind::VGPR || in.dst.value < 0) return -1;
    if (in.fmt == Rdna2Format::MIMG && in.mimg_tfe)
        return in.dst.value + static_cast<int>(mimg_vdata_dwords(in));
    if (in.fmt == Rdna2Format::MTBUF && in.mtbuf_tfe) {
        const uint32_t data_dwords = mtbuf_vdata_dwords(in);
        if (data_dwords) return in.dst.value + static_cast<int>(data_dwords);
    }
    return -1;
}

uint32_t rdna2_vgpr_destination_span(const Rdna2Inst& in) {
    const uint32_t writes = rdna2_vgpr_write_count(in);
    if (in.dst.kind != OperandKind::VGPR || in.dst.value < 0) return 0;
    uint32_t span = writes;

    if (!span) switch (in.fmt) {
        case Rdna2Format::MUBUF:
            if (in.opcode >= 0x04u && in.opcode <= 0x07u) span = in.opcode - 3u;
            else if (in.opcode == 0x1cu) span = 1;
            else if (in.opcode == 0x1du) span = 2;
            else if (in.opcode == 0x1eu) span = 4;
            else if (in.opcode == 0x1fu) span = 3;
            break;
        case Rdna2Format::MTBUF:
            span = mtbuf_vdata_dwords(in);
            break;
        case Rdna2Format::FLAT:
            switch (in.opcode) {
                case 0x18u: case 0x1au: case 0x1cu: span = 1; break;
                case 0x1du: span = 2; break;
                case 0x1eu: span = 4; break;
                case 0x1fu: span = 3; break;
                default: break;
            }
            break;
        case Rdna2Format::MIMG: {
            if (in.opcode == 0x08u || in.opcode == 0x09u)
                span = mimg_vdata_dwords(in);
            break;
        }
        default:
            break;
    }
    const int status_vgpr = rdna2_tfe_status_vgpr(in);
    if (status_vgpr >= in.dst.value)
        span = std::max(span, static_cast<uint32_t>(status_vgpr - in.dst.value + 1));
    return span;
}

// VERIFIED(round-trip llvm-mc gfx1030, VOP1): 0x54 v_rcp_f16, 0x55 v_sqrt_f16, 0x56 v_rsq_f16,
// 0x57 v_log_f16, 0x58 v_exp_f16, 0x59 v_frexp_mant_f16, 0x5A v_frexp_exp_i16_f16, 0x5B v_floor_f16,
// 0x5C v_ceil_f16, 0x5D v_trunc_f16, 0x5E v_rndne_f16, 0x5F v_fract_f16, 0x60 v_sin_f16,
// 0x61 v_cos_f16. The two holes are the FREXP pair — see the header for why they are excluded.
bool vop1_is_f16_unary(uint32_t opcode) {
    return (opcode >= 0x54u && opcode <= 0x58u) || (opcode >= 0x5Bu && opcode <= 0x61u);
}

namespace {
Operand vgpr(uint32_t n) { return {OperandKind::VGPR, (int32_t)(n & 0xFFu)}; }
Operand sgpr(uint32_t n) { return {OperandKind::SGPR, (int32_t)(n & 0x7Fu)}; }
int32_t sext16(uint32_t w) { return (int32_t)(int16_t)(w & 0xFFFFu); }

// Fill opcode + operands for the ALU formats. `d1` is the second dword (for VOP3).
void decode_operands(Rdna2Inst& i) {
    const uint32_t w = i.words[0];
    switch (i.fmt) {
        case Rdna2Format::VOP1:
            i.opcode = (w >> 9) & 0xFFu;  i.dst = vgpr(w >> 17);
            if ((w & 0x1FFu) == 0xF9u) {                          // SDWA (src0 only)
                const uint32_t sd = i.words[1];
                i.src[0] = ((sd >> 23) & 1u) ? decode_src_field(sd & 0xFFu) : vgpr(sd & 0xFFu);  // S0 -> scalar
                i.n_src = 1;
                // "Trivial-or-modifier-only" SDWA: DWORD(6) sels, no sext/reserved — src0 neg@20/abs@21
                // AND the output modifiers CLAMP@13 / OMOD@[15:14] are read + applied by the recompiler
                // (like VOP2/VOP3; DOLL VS: `v_exp_f32_sdwa v0, v0 clamp`). Any real sub-dword select
                // or sext keeps has_modifier and is rejected.
                i.src_neg[0] = ((sd >> 20) & 1u) != 0; i.src_abs[0] = ((sd >> 21) & 1u) != 0;
                i.clamp = ((sd >> 13) & 1u) != 0; i.omod = (uint8_t)((sd >> 14) & 3u);
                if (((w >> 9) & 0xFFu) != 0x38u &&
                    ((sd >> 8) & 7u) == 6u && ((sd >> 16) & 7u) == 6u &&
                    !((sd >> 19) & 0x9u)) i.has_modifier = false;
                // WORD-select v_mov_b32_sdwa (#273 — the f16 half-move idiom): dst WORD_0/1 with
                // UNUSED_PRESERVE, src0 WORD_0/WORD_1/DWORD, no sext/neg/abs/clamp/omod. The
                // recompiler inserts/extracts the 16-bit halves explicitly. (llvm-mc gfx1010:
                // 0x7e0002f9 0x00051400 -> v_mov_b32_sdwa v0, v0 dst_sel:WORD_0
                // dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 — DOLL's box-blur tail.)
                else if (((w >> 9) & 0xFFu) == 0x01u) {
                    uint32_t dsel = (sd >> 8) & 7u, dun = (sd >> 11) & 3u, s0sel = (sd >> 16) & 7u;
                    if ((dsel == 4u || dsel == 5u) && dun == 2u && (s0sel >= 4u && s0sel <= 6u) &&
                        !((sd >> 19) & 0x9u) && !i.clamp && !i.omod && !i.src_neg[0] && !i.src_abs[0]) {
                        i.sdwa_dst_sel = (uint8_t)dsel; i.sdwa_dst_unused = (uint8_t)dun;
                        i.sdwa_src0_sel = (uint8_t)s0sel;
                        i.has_modifier = false;
                    }
                    // BYTE- or WORD-select move into a full dword (#273/#2013 — sub-dword unpack):
                    // dst DWORD + UNUSED_PAD, src0 BYTE_0..3 or WORD_0/1, no neg/abs/clamp/omod.
                    // The recompiler zero- or sign-extends the selected field as S0_SEXT (bit 19)
                    // says; the two are different operations and `sdwa_src0_sext` carries which.
                    // DST_UNUSED is immaterial here because a DWORD destination select writes all
                    // 32 bits, so UNUSED_SEXT (dun==1) is still refused rather than treated as PAD.
                    // (llvm-mc gfx1030: 0x7e0c02f9 0x0000060f -> v_mov_b32_sdwa v6, v15
                    // dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0; Sonic Racing:
                    // CrossWorlds' 0x7e0002f9 0x00040600 -> the same form with src0_sel:WORD_0, and
                    // 0x7e1c02f9 0x000c0608 -> `v_mov_b32_sdwa v14, sext(v8) src0_sel:WORD_0`,
                    // 0x7e3202f9 0x00090619 -> `v_mov_b32_sdwa v25, sext(v25) src0_sel:BYTE_1`.)
                    else if (dsel == 6u && dun == 0u && s0sel <= 5u &&
                             !((sd >> 22) & 1u) && !i.clamp && !i.omod && !i.src_neg[0] && !i.src_abs[0]) {
                        i.sdwa_dst_sel = (uint8_t)dsel; i.sdwa_dst_unused = (uint8_t)dun;
                        i.sdwa_src0_sel = (uint8_t)s0sel;
                        i.sdwa_src0_sext = ((sd >> 19) & 1u) != 0;
                        i.has_modifier = false;
                    }
                }
                // GTA V's compute compaction kernels reverse a selected source word into a full
                // dword: `v_bfrev_b32_sdwa v6, v6 dst_sel:DWORD dst_unused:UNUSED_PAD
                // src0_sel:WORD_0` (7e0c70f9 00040606). SDWA selects and zero-extends the word
                // before BREV, so its bits land in D[31:16] in reversed order. Admit byte/word
                // selects only for the exact full-dword, zero-extending, unmodified shape; SEXT,
                // source/output modifiers, partial destinations, and reserved fields remain
                // fail-visible rather than being interpreted as this narrower operation.
                else if (((w >> 9) & 0xFFu) == 0x38u) {
                    const uint32_t dsel = (sd >> 8) & 7u, dun = (sd >> 11) & 3u;
                    const uint32_t s0sel = (sd >> 16) & 7u;
                    if (dsel == 6u && dun == 0u && s0sel <= 5u &&
                        !((sd >> 19) & 0x9u) && !i.clamp && !i.omod &&
                        !i.src_neg[0] && !i.src_abs[0] && !(sd & 0xff000000u)) {
                        i.sdwa_dst_sel = static_cast<uint8_t>(dsel);
                        i.sdwa_dst_unused = static_cast<uint8_t>(dun);
                        i.sdwa_src0_sel = static_cast<uint8_t>(s0sel);
                        i.has_modifier = false;
                    }
                }
                // v_cvt_f32_f16_sdwa may select either packed half before conversion. The source
                // abs/neg modifiers apply to that f16 value (the recompiler applies them after
                // unpacking); destination remains a full f32 dword. UE4's volume-lighting kernel
                // uses WORD_1 heavily for packed half intermediates.
                else if (((w >> 9) & 0xFFu) == 0x0Bu) {
                    const uint32_t dsel = (sd >> 8) & 7u, dun = (sd >> 11) & 3u;
                    const uint32_t s0sel = (sd >> 16) & 7u;
                    if (dsel == 6u && dun == 0u && s0sel >= 4u && s0sel <= 6u &&
                        !((sd >> 19) & 0x9u) && !i.clamp && !i.omod) {
                        i.sdwa_src0_sel = static_cast<uint8_t>(s0sel);
                        i.has_modifier = false;
                    }
                }
                // v_cvt_f16_f32_sdwa packs a full f32 source into the selected destination half.
                // The live producer writes WORD_1 with UNUSED_PRESERVE to assemble packed values.
                else if (((w >> 9) & 0xFFu) == 0x0Au) {
                    const uint32_t dsel = (sd >> 8) & 7u, dun = (sd >> 11) & 3u;
                    const uint32_t s0sel = (sd >> 16) & 7u;
                    if ((dsel == 4u || dsel == 5u) && dun == 2u && s0sel == 6u &&
                        !((sd >> 19) & 0x9u) && !i.clamp && !i.omod &&
                        !i.src_neg[0] && !i.src_abs[0]) {
                        i.sdwa_dst_sel = static_cast<uint8_t>(dsel);
                        i.sdwa_dst_unused = static_cast<uint8_t>(dun);
                        i.has_modifier = false;
                    }
                }
                // v_cvt_i32_f32_sdwa may write only one destination word while preserving the
                // other. Astro's world-map material uses WORD_0 + UNUSED_PRESERVE from a full
                // DWORD f32 source before packing two integer results into one VGPR.
                else if (((w >> 9) & 0xFFu) == 0x08u) {
                    const uint32_t dsel = (sd >> 8) & 7u, dun = (sd >> 11) & 3u;
                    const uint32_t s0sel = (sd >> 16) & 7u;
                    if ((dsel == 4u || dsel == 5u) && dun == 2u && s0sel == 6u &&
                        !((sd >> 19) & 0x9u) && !i.clamp && !i.omod &&
                        !i.src_neg[0] && !i.src_abs[0]) {
                        i.sdwa_dst_sel = static_cast<uint8_t>(dsel);
                        i.sdwa_dst_unused = static_cast<uint8_t>(dun);
                        i.sdwa_src0_sel = static_cast<uint8_t>(s0sel);
                        i.has_modifier = false;
                    }
                }
                // Integer-to-f32 conversion may select a byte/word before conversion. Unsigned
                // conversion zero-extends it; signed conversion honors SDWA SEXT (bit 19). Astro's
                // title post PS uses WORD_1+SEXT for v_cvt_f32_i32.
                // Astro title-ship PS uses the exact packet 7e0a0cf9 0000160b:
                // v_cvt_f32_u32_sdwa v5, v11 src0_sel:BYTE_0. Destination is a full dword; the
                // canonical UNUSED_PRESERVE value is immaterial because all 32 bits are written.
                else if (((w >> 9) & 0xFFu) == 0x05u ||
                         ((w >> 9) & 0xFFu) == 0x06u) {
                    const uint32_t dsel = (sd >> 8) & 7u, dun = (sd >> 11) & 3u;
                    const uint32_t s0sel = (sd >> 16) & 7u;
                    const bool signed_cvt = ((w >> 9) & 0xFFu) == 0x05u;
                    const bool sext = ((sd >> 19) & 1u) != 0;
                    if (dsel == 6u && (dun == 0u || dun == 2u) && s0sel <= 5u &&
                        !((sd >> 20) & 0x7u) && (signed_cvt || !sext) &&
                        !i.clamp && !i.omod &&
                        !i.src_neg[0] && !i.src_abs[0]) {
                        i.sdwa_dst_sel = static_cast<uint8_t>(dsel);
                        i.sdwa_dst_unused = static_cast<uint8_t>(dun);
                        i.sdwa_src0_sel = static_cast<uint8_t>(s0sel);
                        i.has_modifier = false;
                    }
                }
                // The 16-bit CONVERT family (0x50-0x53) in SDWA form: the live producer writes one
                // destination word with UNUSED_PRESERVE and reads either source word (Sonic Racing:
                // CrossWorlds emits both `7e12a6f9,00061509` = `v_cvt_i16_f16_sdwa v9, v9
                // dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:DWORD` and
                // `7e20a2f9,0005140b` = `v_cvt_f16_i16_sdwa v16, v11 dst_sel:WORD_0
                // dst_unused:UNUSED_PRESERVE src0_sel:WORD_1`, #2013). DWORD and WORD_0 name the
                // same operand bits [15:0] for a 16-bit op; WORD_1 names [31:16]. A BYTE source
                // select or any neg/abs/clamp/omod stays fail-visible: the generic VOP1 modifier
                // path would apply the modifier in the f32 domain to the packed dword, which is the
                // wrong half for an op that reads a 16-bit operand.
                else if (((w >> 9) & 0xFFu) >= 0x50u && ((w >> 9) & 0xFFu) <= 0x53u) {
                    const uint32_t dsel = (sd >> 8) & 7u, dun = (sd >> 11) & 3u;
                    const uint32_t s0sel = (sd >> 16) & 7u;
                    if ((dsel == 4u || dsel == 5u) && dun == 2u &&
                        (s0sel >= 4u && s0sel <= 6u) &&
                        !((sd >> 19) & 0x9u) && !i.clamp && !i.omod &&
                        !i.src_neg[0] && !i.src_abs[0]) {
                        i.sdwa_dst_sel = static_cast<uint8_t>(dsel);
                        i.sdwa_dst_unused = static_cast<uint8_t>(dun);
                        i.sdwa_src0_sel = static_cast<uint8_t>(s0sel);
                        i.has_modifier = false;
                    }
                }
                // Packed unary f16 SDWA selects one source half, writes one destination half, and
                // preserves the other. The admitted set is exactly the VOP1 f16 unary family the
                // recompiler's plain-e32 lowering already covers, named once by `vop1_is_f16_unary`
                // so this admission and `emit_f16_unary` cannot disagree about what the family is.
                // (Sonic Racing: CrossWorlds' compute kernels, #2013: 0x7e0cbcf9 0x00051406 ->
                // `v_rndne_f16_sdwa v6, v6 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE
                // src0_sel:WORD_1` and 0x7e14b8f9 0x0005150a -> `v_ceil_f16_sdwa v10, v10
                // dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1`.)
                else if (vop1_is_f16_unary((w >> 9) & 0xFFu)) {
                    const uint32_t dsel = (sd >> 8) & 7u, dun = (sd >> 11) & 3u;
                    const uint32_t s0sel = (sd >> 16) & 7u;
                    if ((dsel == 4u || dsel == 5u) && dun == 2u &&
                        (s0sel >= 4u && s0sel <= 6u) && !((sd >> 19) & 0x9u) &&
                        !i.clamp && !i.omod && !i.src_neg[0] && !i.src_abs[0]) {
                        i.sdwa_dst_sel = static_cast<uint8_t>(dsel);
                        i.sdwa_dst_unused = static_cast<uint8_t>(dun);
                        i.sdwa_src0_sel = static_cast<uint8_t>(s0sel);
                        i.has_modifier = false;
                    }
                }
            } else if (i.has_dpp) { i.src[0] = vgpr(i.words[1]); i.n_src = 1; }   // DPP16: real SRC0 in dw1[7:0]
            else { i.src[0] = decode_src_field(w & 0x1FFu); i.n_src = 1; }
            break;
        case Rdna2Format::VOP2:
            i.opcode = (w >> 25) & 0x3Fu; i.dst = vgpr(w >> 17);
            if ((w & 0x1FFu) == 0xF9u) {                          // SDWA (src0 in ctrl dword; src1 in dword0)
                const uint32_t sd = i.words[1];
                i.src[0] = ((sd >> 23) & 1u) ? decode_src_field(sd & 0xFFu)     : vgpr(sd & 0xFFu);       // S0
                i.src[1] = ((sd >> 31) & 1u) ? decode_src_field((w >> 9) & 0xFFu) : vgpr((w >> 9) & 0xFFu); // S1
                i.n_src = 2;
                // SDWA source float modifiers: src0 neg@20/abs@21, src1 neg@28/abs@29 — the recompiler
                // applies these (like VOP3). Read them, then treat the packet as handleable ("trivial")
                // when the only sub-dword controls present are neg/abs: DWORD(6) sels, no clamp/omod, and
                // no SEXT(bit19/27) or reserved(bit22/30) — i.e. mask out neg/abs (0x6) from the nibble check.
                i.src_neg[0] = ((sd >> 20) & 1u) != 0; i.src_abs[0] = ((sd >> 21) & 1u) != 0;
                i.src_neg[1] = ((sd >> 28) & 1u) != 0; i.src_abs[1] = ((sd >> 29) & 1u) != 0;
                // Output modifiers CLAMP@13 / OMOD@[15:14] (×2/×4/×0.5) — the recompiler applies these to the
                // float result (like VOP3), so they no longer force rejection. Only real sub-dword selects
                // (dst/src != DWORD=6) or SEXT/reserved keep has_modifier. (PPSA02664 PS: v_mul_f32 ×2, #121.)
                i.clamp = ((sd >> 13) & 1u) != 0; i.omod = (uint8_t)((sd >> 14) & 3u);
                if (((sd >> 8) & 7u) == 6u && ((sd >> 16) & 7u) == 6u && ((sd >> 24) & 7u) == 6u &&
                    !((sd >> 19) & 0x9u) && !((sd >> 27) & 0x9u))
                    i.has_modifier = false;
                // Integer VOP2 SDWA may select a byte/word from either source before the ALU
                // operation, and may write the result into a byte/word of the destination.  Plucky
                // Squire's first gameplay scene uses the four byte selectors with v_mul_u32_u24 to
                // unpack an RGBA8 value (three times BYTE_0..3); Sonic Racing: CrossWorlds emits
                // `360400f9,04861486` = `v_and_b32_sdwa v2, 6, v0 dst_sel:WORD_0
                // dst_unused:UNUSED_PRESERVE src1_sel:WORD_0`, a sub-dword DESTINATION (#2013).
                // Admit the general no-saturation integer subset: each source field is zero- or
                // sign-extended as its SEXT bit says (19 for S0, 27 for S1), the destination field
                // is written with the low bits of the result and the rest of the register is
                // zero-filled (UNUSED_PAD) or kept (UNUSED_PRESERVE) as DST_UNUSED says.
                // (Sonic Racing: CrossWorlds' compute kernels, #2013: 0x4a080af9 0x0c860688 ->
                // `v_add_nc_u32_sdwa v4, 8, sext(v5) dst_sel:DWORD dst_unused:UNUSED_PAD
                // src0_sel:DWORD src1_sel:WORD_0`.) SEXT is admitted only together with a real
                // sub-dword select: no live encoding here sets it on a DWORD source, so that
                // combination stays fail-visible rather than assumed to be a no-op. UNUSED_SEXT and
                // integer clamp likewise remain fail-visible until their semantics are modeled.
                else {
                    const uint32_t op = (w >> 25) & 0x3Fu;
                    const bool integer_op =
                        op == 0x0Bu || (op >= 0x11u && op <= 0x14u) ||
                        op == 0x16u || op == 0x18u || (op >= 0x1Au && op <= 0x1Eu) ||
                        (op >= 0x25u && op <= 0x2Au);
                    const uint32_t dsel = (sd >> 8) & 7u, dun = (sd >> 11) & 3u;
                    const uint32_t s0sel = (sd >> 16) & 7u, s1sel = (sd >> 24) & 7u;
                    const bool s0_sext = ((sd >> 19) & 1u) != 0, s1_sext = ((sd >> 27) & 1u) != 0;
                    if (integer_op && dsel <= 6u && (dun == 0u || dun == 2u) &&
                        s0sel <= 6u && s1sel <= 6u &&
                        !(s0_sext && s0sel == 6u) && !(s1_sext && s1sel == 6u) &&
                        !((sd >> 20) & 0x7u) && !((sd >> 28) & 0x7u) &&
                        !i.clamp && !i.omod) {
                        i.sdwa_dst_sel = static_cast<uint8_t>(dsel);
                        i.sdwa_dst_unused = static_cast<uint8_t>(dun);
                        i.sdwa_src0_sel = static_cast<uint8_t>(s0sel);
                        i.sdwa_src1_sel = static_cast<uint8_t>(s1sel);
                        i.sdwa_src0_sext = s0_sext;
                        i.sdwa_src1_sext = s1_sext;
                        i.has_modifier = false;
                    }
                }
                // WORD-dst v_mul_f16_sdwa (#273 — the f16 half-packing idiom): dst WORD_0/1 with
                // UNUSED_PRESERVE, src sels WORD/DWORD, no sext/neg/abs/clamp/omod. (llvm-mc gfx1010:
                // 0x6a0000f9 0x0686156a -> v_mul_f16_sdwa v0, vcc_lo, v0 dst_sel:WORD_1
                // dst_unused:UNUSED_PRESERVE — DOLL's box-blur tail.)
                if (i.has_modifier && (((w >> 25) & 0x3Fu) == 0x32u ||
                         ((w >> 25) & 0x3Fu) == 0x33u ||
                         ((w >> 25) & 0x3Fu) == 0x35u ||
                         ((w >> 25) & 0x3Fu) == 0x39u ||
                         ((w >> 25) & 0x3Fu) == 0x3Au)) {
                    uint32_t dsel = (sd >> 8) & 7u, dun = (sd >> 11) & 3u;
                    uint32_t s0sel = (sd >> 16) & 7u, s1sel = (sd >> 24) & 7u;
                    const bool preserve_word = (dsel == 4u || dsel == 5u) && dun == 2u;
                    const bool padded_dword = dsel == 6u && dun == 0u;
                    if ((preserve_word || padded_dword) &&
                        (s0sel >= 4u && s0sel <= 6u) && (s1sel >= 4u && s1sel <= 6u) &&
                        !((sd >> 19) & 0x9u) && !((sd >> 27) & 0x9u) && !i.omod) {
                        i.sdwa_dst_sel = (uint8_t)dsel; i.sdwa_dst_unused = (uint8_t)dun;
                        i.sdwa_src0_sel = (uint8_t)s0sel; i.sdwa_src1_sel = (uint8_t)s1sel;
                        i.has_modifier = false;
                    }
                }
                // WORD-destination v_cndmask_b32_sdwa selects one 16-bit source through VCC and
                // preserves the opposite destination half (the producer's packed conditional).
                else if (i.has_modifier && ((w >> 25) & 0x3Fu) == 0x01u) {
                    uint32_t dsel = (sd >> 8) & 7u, dun = (sd >> 11) & 3u;
                    uint32_t s0sel = (sd >> 16) & 7u, s1sel = (sd >> 24) & 7u;
                    if ((dsel == 4u || dsel == 5u) && dun == 2u &&
                        s0sel >= 4u && s0sel <= 6u && s1sel >= 4u && s1sel <= 6u &&
                        !((sd >> 19) & 0x9u) && !((sd >> 27) & 0x9u) && !i.clamp && !i.omod &&
                        !i.src_neg[0] && !i.src_abs[0] && !i.src_neg[1] && !i.src_abs[1]) {
                        i.sdwa_dst_sel = static_cast<uint8_t>(dsel);
                        i.sdwa_dst_unused = static_cast<uint8_t>(dun);
                        i.sdwa_src0_sel = static_cast<uint8_t>(s0sel);
                        i.sdwa_src1_sel = static_cast<uint8_t>(s1sel);
                        i.has_modifier = false;
                    }
                }
                // NGG culling extracts three packed byte indices and converts them to LDS byte
                // addresses with `v_lshlrev_b32_sdwa ..., 2, vN src1_sel:BYTE_k`. Admit the exact
                // zero-extending form: full-dword destination/shift amount, one byte from src1, and
                // no sign/float/output modifiers.
                else if (((w >> 25) & 0x3Fu) == 0x1Au) {
                    const uint32_t dsel = (sd >> 8) & 7u, dun = (sd >> 11) & 3u;
                    const uint32_t s0sel = (sd >> 16) & 7u, s1sel = (sd >> 24) & 7u;
                    if (dsel == 6u && dun == 0u && s0sel == 6u && s1sel <= 5u &&
                        !((sd >> 19) & 0xFu) && !((sd >> 27) & 0xFu) &&
                        !i.clamp && !i.omod) {
                        i.sdwa_src0_sel = static_cast<uint8_t>(s0sel);
                        i.sdwa_src1_sel = static_cast<uint8_t>(s1sel);
                        i.has_modifier = false;
                    }
                }
            } else if (i.has_dpp) {   // DPP16: real SRC0 in dw1[7:0]; SRC1 stays the dword0 VGPR field
                i.src[0] = vgpr(i.words[1]); i.src[1] = vgpr(w >> 9); i.n_src = 2;
            } else { i.src[0] = decode_src_field(w & 0x1FFu); i.src[1] = vgpr(w >> 9); i.n_src = 2; }
            break;
        case Rdna2Format::VOPC:
            i.opcode = (w >> 17) & 0xFFu; i.dst = {OperandKind::Special, 106 /*VCC_LO*/};
            if ((w & 0x1FFu) == 0xF9u) {                          // SDWA (same layout as VOP2)
                const uint32_t sd = i.words[1];
                i.src[0] = ((sd >> 23) & 1u) ? decode_src_field(sd & 0xFFu)     : vgpr(sd & 0xFFu);
                i.src[1] = ((sd >> 31) & 1u) ? decode_src_field((w >> 9) & 0xFFu) : vgpr((w >> 9) & 0xFFu);
                i.n_src = 2;
                // SDWAB: bit15 = SD (write the SDST SGPR pair instead of VCC), bits[14:8] = SDST index.
                if ((sd >> 15) & 1u) i.dst = sgpr((sd >> 8) & 0x7Fu);
                // VOPC SDWA has NO dst_sel/clamp/omod (byte1 holds SDST/SD, not a VGPR dst_sel). Source
                // float modifiers src0 neg@20/abs@21, src1 neg@28/abs@29 are read + applied by the
                // recompiler (like VOP2/VOP3 — DOLL: `v_cmp_gt_f32_sdwa vcc_lo, |v5|, s4`); trivial iff
                // both source selects are DWORD and no SEXT(19/27)/reserved(22/30) bits.
                i.src_neg[0] = ((sd >> 20) & 1u) != 0; i.src_abs[0] = ((sd >> 21) & 1u) != 0;
                i.src_neg[1] = ((sd >> 28) & 1u) != 0; i.src_abs[1] = ((sd >> 29) & 1u) != 0;
                if (((sd >> 16) & 7u) == 6u && ((sd >> 24) & 7u) == 6u &&
                    !((sd >> 19) & 0x9u) && !((sd >> 27) & 0x9u))
                    i.has_modifier = false;
                // f16 VOPC may select either 16-bit half. UE4 uses WORD_1 for packed visibility
                // values (`v_cmpx_gt_f16_sdwa ..., v7, 0 src0_sel:WORD_1`). Preserve the selects so
                // the recompiler can unpack the chosen half; other sub-dword compare forms reject.
                else {
                    // Map a cmpx opcode onto the base compare whose windows are tested below. This
                    // used to re-list the cmpx ranges here and got three of the six, so every
                    // `v_cmpx_*_u16` SDWA packet rejected while its plain e32 form worked (#2120).
                    const uint32_t eff =
                        vopc_is_cmpx(i.opcode) ? i.opcode - 0x10u : i.opcode;
                    const uint32_t s0sel = (sd >> 16) & 7u, s1sel = (sd >> 24) & 7u;
                    if (eff >= 0xC9u && eff <= 0xCEu &&
                        s0sel >= 4u && s0sel <= 6u && s1sel >= 4u && s1sel <= 6u &&
                        !((sd >> 19) & 0x9u) && !((sd >> 27) & 0x9u)) {
                        i.sdwa_src0_sel = static_cast<uint8_t>(s0sel);
                        i.sdwa_src1_sel = static_cast<uint8_t>(s1sel);
                        i.has_modifier = false;
                    }
                    // Integer VOPC SDWA may compare a selected byte/word after zero extension.
                    // Astro Bot uses `v_cmp_ne_u32 1, v63 src1_sel:BYTE_0` in its visibility
                    // kernel. Admit the exact no-SEXT/no-neg/abs subset; signed extension and
                    // integer source modifiers remain fail-visible until modeled.
                    // The 32-bit windows (i32 0x80-0x87, u32 0xc0-0xc7) and the 16-bit ones
                    // (i16 0x88-0x8f, u16 0xa8-0xaf) — VERIFIED(round-trip llvm-mc gfx1030:
                    // `7d5700f9,86060003` = `v_cmp_le_u16_sdwa vcc_lo, v3, 0`, opcode 0xab, #2013).
                    const bool integer_compare =
                        (eff >= 0x81u && eff <= 0x86u) ||
                        (eff >= 0x89u && eff <= 0x8Eu) ||
                        (eff >= 0xA9u && eff <= 0xAEu) ||
                        (eff >= 0xC1u && eff <= 0xC6u);
                    if (integer_compare && s0sel <= 6u && s1sel <= 6u &&
                        !((sd >> 19) & 0xFu) && !((sd >> 27) & 0xFu)) {
                        i.sdwa_src0_sel = static_cast<uint8_t>(s0sel);
                        i.sdwa_src1_sel = static_cast<uint8_t>(s1sel);
                        i.has_modifier = false;
                    }
                }
            } else { i.src[0] = decode_src_field(w & 0x1FFu); i.src[1] = vgpr(w >> 9); i.n_src = 2; }
            break;
        case Rdna2Format::VOP3: {
            const uint32_t d1 = i.words[1];
            i.opcode = (w >> 16) & 0x3FFu; i.dst = vgpr(w);   // VOP3A: vdst in [7:0] of dword0
            i.src[0] = decode_src_field(d1 & 0x1FFu);
            i.src[1] = decode_src_field((d1 >> 9) & 0x1FFu);
            i.src[2] = decode_src_field((d1 >> 18) & 0x1FFu); i.n_src = 3;
            // Source float modifiers: ABS = dword0[10:8], NEG = dword1[31:29] (one bit per source). These
            // are correctness-critical (a-b, abs(), -x are ubiquitous) — previously ignored → silent wrong.
            for (int k = 0; k < 3; k++) { i.src_abs[k] = ((w >> (8 + k)) & 1u) != 0; i.src_neg[k] = ((d1 >> (29 + k)) & 1u) != 0; }
            // CLAMP (dword0[15]) = saturate result to [0,1]; OMOD (dword1[28:27]) = result ×2/×4/×0.5.
            // Applied to the float result in the recompiler (rdna2_to_spirv VOP3 float ops).
            i.clamp = ((w >> 15) & 1u) != 0;
            i.omod  = (uint8_t)((d1 >> 27) & 3u);
            // VOPC-as-VOP3 (the e64 compare encoding) occupies opcodes 0x000..0x0ff. Its
            // dword0 low field is an explicit scalar-mask destination, not VDST, and it has only
            // two data sources. Classify it as VOPC after retaining the e64 source modifiers so
            // every mask/provenance consumer sees the architectural destination. Treating Astro's
            // `v_cmp_gt_u64_e64 vcc_lo,vcc,0` as ordinary VOP3 invented a VGPR106 write and made the
            // compute CFG dispatcher reject opcode 0xe4.
            if (i.opcode < 0x100u) {
                i.fmt = Rdna2Format::VOPC;
                i.dst = sgpr(w & 0x7Fu);
                i.src[2] = {};
                i.n_src = 2;
            }
            // VOP3B (sec 13.3.4 — the complete opcode list: v_add/sub/subrev_co_ci_u32
            // 0x128/0x129/0x12A, v_div_scale_f32/f64 0x16D/0x16E, v_mad_u64_u32/v_mad_i64_i32
            // 0x176/0x177, v_add/sub/subrev_co_u32 0x30F/0x310/0x319): dword0[14:8] is a scalar
            // carry/flag dst (SDST), not abs modifiers — decode it and clear the mis-read abs
            // bits. CLMP (bit 15) IS architecturally valid in VOP3B and is kept so stage 2 can
            // reject unmodeled integer saturation instead of silently dropping it.
            switch (i.opcode) {
                case 0x128u: case 0x129u: case 0x12Au: case 0x16Du: case 0x16Eu:
                case 0x176u: case 0x177u: case 0x30Fu: case 0x310u: case 0x319u:
                    i.sdst = sgpr((w >> 8) & 0x7Fu);
                    i.src_abs[0] = i.src_abs[1] = i.src_abs[2] = false;
                    break;
                default: break;
            }
            // The 0x30F/0x310/0x319 _co_ forms produce carry/borrow in SDST but do not consume a
            // carry-in. Their reserved SRC2 field commonly decodes as s0; exposing that phantom
            // operand makes CFG provenance reject a valid instruction when s0 differs across a
            // join, before the instruction can replace it with its fresh carry result.
            if (i.opcode == 0x30Fu || i.opcode == 0x310u || i.opcode == 0x319u) {
                i.src[2] = {};
                i.n_src = 2;
            }
            // V_LDEXP_F32 is likewise a two-source VOP3A instruction. Its reserved SRC2 bits are
            // zero in GTA V's exact packet and therefore decode as s0 unless cleared here. Exposing
            // that phantom scalar read can make CFG/provenance analysis reject an otherwise valid
            // instruction when s0 differs across a merge, before the opcode emitter is reached.
            if (i.opcode == 0x362u) {
                i.src[2] = {};
                i.n_src = 2;
            }
            // Scalar 16-bit VOP3 operations: OPSEL[2:0] selects each packed source half and OPSEL[3]
            // selects the destination half. Reuse the packed-op selector field for this family.
            // VERIFIED(llvm-mc gfx1030): 0x351/0x354/0x357 are v_min3_f16/v_max3_f16/v_med3_f16
            // (their i16 and u16 siblings sit at 0x352/0x355/0x358 and 0x353/0x356/0x359, so the
            // three signedness variants interleave rather than forming one contiguous run —
            // positional inference across this window is exactly how a wrong lowering gets
            // written); 0x303-0x30e (0x306 is not an
            // instruction), 0x314, 0x340/0x35e and 0x352-0x359 are the 16-bit integer VALU family —
            // add/sub/mul, the reversed shifts, min/max, mad and min3/max3/med3 (#2013 — Sonic
            // Racing: CrossWorlds emits
            // `v_lshrrev_b16 v1, 1, v0 op_sel:[0,0,1]`, which writes the HIGH destination half;
            // dropping OPSEL there would silently write the wrong half).
            if (i.opcode == 0x311u || i.opcode == 0x34Bu || i.opcode == 0x351u ||
                i.opcode == 0x354u || i.opcode == 0x357u ||
                (i.opcode >= 0x303u && i.opcode <= 0x30Eu && i.opcode != 0x306u) ||
                i.opcode == 0x314u || i.opcode == 0x340u || i.opcode == 0x35Eu ||
                i.opcode == 0x352u || i.opcode == 0x353u || i.opcode == 0x355u ||
                i.opcode == 0x356u || i.opcode == 0x358u || i.opcode == 0x359u)
                i.vop3p_opsel = static_cast<uint8_t>((w >> 11) & 0xFu);
            // V_PERMLANE16_B32 / V_PERMLANEX16_B32 overload OPSEL[0] as FI and OPSEL[1] as
            // BOUND_CTRL. OPSEL[2:3], ABS, NEG, CLAMP and OMOD remain reserved/unsupported and are
            // rejected by the emitter so malformed encodings cannot silently lose modifiers.
            if (i.opcode == 0x377u || i.opcode == 0x378u) {
                i.permlane_fetch_inactive = ((w >> 11) & 1u) != 0;
                i.permlane_bound_ctrl = ((w >> 12) & 1u) != 0;
            }
            break;
        }
        case Rdna2Format::VOP3P: {
            // dword0: VDST[7:0], NEG_HI[10:8], OPSEL[13:11], OPSEL_HI2[14], CLAMP[15], OP[22:16].
            // dword1: SRC0[8:0], SRC1[17:9], SRC2[26:18], OPSEL_HI[28:27], NEG[31:29]. Any modifier
            // bit set (neg/neg_hi/opsel/clamp) keeps has_modifier so the recompiler rejects rather
            // than miscomputes; the trivial form reads all three sources as full f32 (fma_mix).
            const uint32_t d1 = i.words[1];
            i.opcode = (w >> 16) & 0x7Fu; i.dst = vgpr(w);
            i.src[0] = decode_src_field(d1 & 0x1FFu);
            i.src[1] = decode_src_field((d1 >> 9) & 0x1FFu);
            i.src[2] = decode_src_field((d1 >> 18) & 0x1FFu); i.n_src = 3;
            // Packed f16 fma/add/mul/min/max (0x0e-0x12), the packed 16-bit integer family
            // (0x00-0x0d) and v_fma_mix_f32/mixlo/mixhi (0x20-0x22): every modifier bit is MODELED
            // (#273/#2013). For packed ops OPSEL/NEG select and negate each source independently
            // for the low result; OPSEL_HI/NEG_HI do the same for the high result.
            // For the mix family:
            // OPSEL_HI[k] selects an f16-half read (which half via OPSEL[k]), NEG negates, NEG_HI
            // is ABS for the mix family, CLAMP saturates. (llvm-mc gfx1030 round-trip on the live
            // DOLL bytes: 0xcc200044 0x9a02170b = v_fma_mix_f32 v68, v11, v11, neg(0)
            // op_sel_hi:[1,1,0].) VOP3P opcodes outside those ranges still reject on any modifier
            // bit.
            if (i.opcode == 0x0E || i.opcode == 0x0F || i.opcode == 0x10 ||
                i.opcode == 0x11 || i.opcode == 0x12) {
                // Packed f16 fma/add/mul/min/max. VERIFIED(round-trip llvm-mc gfx1030):
                // 0x0e v_pk_fma_f16, 0x0f v_pk_add_f16, 0x10 v_pk_mul_f16, 0x11 v_pk_min_f16,
                // 0x12 v_pk_max_f16.
                const uint32_t neg = (d1 >> 29) & 7u;
                for (int k = 0; k < 3; ++k) i.src_neg[k] = ((neg >> k) & 1u) != 0;
                i.vop3p_neg_hi  = static_cast<uint8_t>((w >> 8) & 7u);
                i.vop3p_opsel   = static_cast<uint8_t>((w >> 11) & 7u);
                i.vop3p_opsel_hi = static_cast<uint8_t>(((d1 >> 27) & 3u) | (((w >> 14) & 1u) << 2));
                i.clamp = ((w >> 15) & 1u) != 0;
            } else if (i.opcode <= 0x0Du) {
                // Packed 16-bit INTEGER family (0x00..0x0d — mad/mul/add/sub/shift/min/max, both
                // signednesses). VERIFIED(round-trip llvm-mc gfx1030 for every opcode in the range;
                // Sonic Racing: CrossWorlds emits `cc0a0002,18020504` = v_pk_add_u16 v2, v4, v2,
                // whose DEFAULT op_sel_hi:[1,1] sets dword1[28:27] — which is why the blanket
                // "any modifier bit rejects" rule below refused the plainest possible encoding).
                // OPSEL/OPSEL_HI select each source's half for the low/high result exactly as they
                // do for the packed f16 ops, and NEG/NEG_HI occupy the same fields; the emitter
                // applies them as the sign-bit flip they physically are. CLAMP requests integer
                // saturation the emitter does not model, so it keeps has_modifier and rejects.
                const uint32_t neg = (d1 >> 29) & 7u;
                for (int k = 0; k < 3; ++k) i.src_neg[k] = ((neg >> k) & 1u) != 0;
                i.vop3p_neg_hi   = static_cast<uint8_t>((w >> 8) & 7u);
                i.vop3p_opsel    = static_cast<uint8_t>((w >> 11) & 7u);
                i.vop3p_opsel_hi = static_cast<uint8_t>(((d1 >> 27) & 3u) | (((w >> 14) & 1u) << 2));
                if (((w >> 15) & 1u) != 0u) i.has_modifier = true;
            } else if (i.opcode >= 0x20 && i.opcode <= 0x22) {
                const uint32_t neg = (d1 >> 29) & 7u, neg_hi = (w >> 8) & 7u;
                for (int k = 0; k < 3; k++) {
                    i.src_neg[k] = ((neg >> k) & 1u) != 0;
                    i.src_abs[k] = ((neg_hi >> k) & 1u) != 0;
                }
                i.vop3p_opsel    = (uint8_t)((w >> 11) & 7u);
                i.vop3p_opsel_hi = (uint8_t)((((d1 >> 27) & 3u)) | (((w >> 14) & 1u) << 2));
                i.clamp = ((w >> 15) & 1u) != 0;
            } else if (((w >> 8) & 0xFFu) != 0u || ((d1 >> 27) & 0x1Fu) != 0u) i.has_modifier = true;
            break;
        }
        case Rdna2Format::SOP1:
            i.opcode = (w >> 8) & 0xFFu;  i.dst = sgpr(w >> 16);
            i.src[0] = decode_src_field(w & 0xFFu); i.n_src = 1; break;
        case Rdna2Format::SOP2:
            i.opcode = (w >> 23) & 0x7Fu; i.dst = sgpr(w >> 16);
            i.src[0] = decode_src_field(w & 0xFFu); i.src[1] = decode_src_field((w >> 8) & 0xFFu);
            i.n_src = 2; break;
        case Rdna2Format::SOPK:
            i.opcode = (w >> 23) & 0x1Fu; i.dst = sgpr(w >> 16); i.simm16 = sext16(w); break;
        case Rdna2Format::SOPC:
            i.opcode = (w >> 16) & 0x7Fu;
            i.src[0] = decode_src_field(w & 0xFFu); i.src[1] = decode_src_field((w >> 8) & 0xFFu);
            i.n_src = 2; break;
        case Rdna2Format::SOPP:
            i.opcode = (w >> 16) & 0x7Fu; i.simm16 = sext16(w); break;
        case Rdna2Format::EXP: {
            const uint32_t d1 = i.words[1];
            i.exp_en = w & 0xFu; i.exp_target = (w >> 4) & 0x3Fu;
            i.exp_compr = ((w >> 10) & 1u) != 0;    // COMPR (bit 10): src[0]/src[1] hold packed f16x2
            for (int k = 0; k < 4; k++) i.src[k] = vgpr((d1 >> (8 * k)) & 0xFFu);
            i.n_src = 4; break;
        }
        case Rdna2Format::SMEM: {
            // Scalar memory load. opcode[25:18]; SDATA (dest SGPR) [12:6]; SBASE (SGPR *pair*, field
            // is the pair index so ×2) [5:0]. Address = SBASE + SOFFSET-reg + OFFSET, where dword1[20:0]
            // is a SIGNED 21-bit immediate byte OFFSET (stored sign-extended in `literal`) and
            // dword1[31:25] is the 7-bit SOFFSET register field (125 = NULL = immediate-only). Both are
            // decoded now (#149): SOFFSET into src[1] so a register-offset load can be recognized (and
            // rejected until supported) instead of silently translating with the immediate alone.
            i.opcode = (w >> 18) & 0xFFu;
            i.dst    = sgpr(w >> 6);                 // SDATA
            i.src[0] = sgpr((w & 0x3Fu) << 1);       // SBASE (pair base)
            uint32_t off21 = i.words[1] & 0x1FFFFFu;
            i.literal = (off21 & 0x100000u) ? (off21 | 0xFFE00000u) : off21;   // sign-extend bit 20
            i.src[1] = decode_src_field((i.words[1] >> 25) & 0x7Fu);           // SOFFSET (125 = NULL)
            i.n_src = 2; break;
        }
        case Rdna2Format::MUBUF: {
            // Untyped buffer op. opcode[25:18] (8 bits — Table 98; a 7-bit mask aliased the D16
            // format ops 128-135 onto the 32-bit format ops 0-7, silently translating the wrong
            // data layout); VDATA (dest/src VGPR) d1[15:8]; VADDR d1[7:0];
            // SRSRC (V# descriptor base SGPR, field ×4) d1[20:16]; SOFFSET d1[31:24]. 12-bit inst
            // offset d0[11:0] + offen d0[12] + idxen d0[13] packed into `literal`.
            const uint32_t d1 = i.words[1];
            i.opcode = (w >> 18) & 0xFFu;
            i.mubuf_glc = ((w >> 14) & 1u) != 0;   // atomics: return pre-op value to VGPR
            i.mubuf_dlc = ((w >> 15) & 1u) != 0;   // ordinary loads: bypass device-level cache
            i.mubuf_lds = ((w >> 16) & 1u) != 0;   // buffer<->LDS transfer (rejected in stage 2)
            i.dst    = vgpr(d1 >> 8);                          // VDATA
            i.src[0] = vgpr(d1);                              // VADDR
            i.src[1] = sgpr(((d1 >> 16) & 0x1Fu) << 2);       // SRSRC (descriptor base)
            // SOFFSET is a full 8-bit src field: 0x80 = inline 0 (the standard "no soffset" encoding).
            // A 7-bit mask folded 0x80 onto SGPR s0, silently adding a live descriptor dword to every
            // buffer address whenever a shader wrote s0 before the fetch.
            i.src[2] = decode_src_field((d1 >> 24) & 0xFFu);  // SOFFSET
            i.literal = (w & 0xFFFu) | (((w >> 12) & 1u) << 12) | (((w >> 13) & 1u) << 13);
            i.n_src = 3; break;
        }
        case Rdna2Format::MTBUF: {
            // Typed buffer op. gfx1030 keeps the eight load/store opcodes in d0[18:16] and uses
            // the Gen5 combined seven-bit BUF_FMT in d0[25:19]. Operand/address fields match
            // MUBUF: VDATA, VADDR, SRSRC*4, SOFFSET plus OFFSET/OFFEN/IDXEN.
            const uint32_t d1 = i.words[1];
            // gfx10 moves OP[3] to instruction bit 53 (dword1 bit 21) to make room for the
            // combined seven-bit format. Opcodes 8..15 are the packed-D16 variants.
            i.opcode = ((w >> 16) & 0x7u) | (((d1 >> 21) & 1u) << 3);
            i.mtbuf_format = (w >> 19) & 0x7Fu;
            i.mtbuf_tfe = ((d1 >> 23) & 1u) != 0;
            i.mubuf_glc = ((w >> 14) & 1u) != 0;
            i.mubuf_dlc = ((w >> 15) & 1u) != 0;
            i.dst    = vgpr(d1 >> 8);
            i.src[0] = vgpr(d1);
            i.src[1] = sgpr(((d1 >> 16) & 0x1Fu) << 2);
            i.src[2] = decode_src_field((d1 >> 24) & 0xFFu);
            i.literal = (w & 0xFFFu) | (((w >> 12) & 1u) << 12) |
                        (((w >> 13) & 1u) << 13);
            i.n_src = 3; break;
        }
        case Rdna2Format::FLAT: {
            // gfx10 FLAT/GLOBAL/SCRATCH. OP d0[24:18], SEG d0[15:14], LDS d0[13],
            // signed byte OFFSET d0[11:0];
            // d1 holds VADDR[7:0], VDATA[15:8], SADDR[22:16], and VDST[31:24]. A scratch instruction
            // uses either `off, sN` (SADDR != NULL; canonical VADDR field 0) or `vN, off`
            // (SADDR=NULL=125). Global/flat address forms retain their operands for diagnostics but
            // remain fail-closed in the recompiler until arbitrary guest pointers are modeled.
            const uint32_t d1 = i.words[1];
            i.opcode = (w >> 18) & 0x7Fu;
            i.flat_segment = (w >> 14) & 0x3u;
            i.flat_glc = ((w >> 16) & 1u) != 0;
            i.flat_slc = ((w >> 17) & 1u) != 0;
            i.flat_dlc = ((w >> 12) & 1u) != 0;
            i.flat_lds = ((w >> 13) & 1u) != 0;
            const uint32_t off12 = w & 0xFFFu;
            i.literal = (off12 & 0x800u) ? (off12 | 0xFFFFF000u) : off12;
            const uint32_t vaddr = d1 & 0xFFu;
            const uint32_t vdata = (d1 >> 8) & 0xFFu;
            const uint32_t saddr = (d1 >> 16) & 0x7Fu;
            const uint32_t vdst = (d1 >> 24) & 0xFFu;
            const bool store = i.opcode >= 0x18u && i.opcode <= 0x1Fu;
            i.dst = vgpr(store ? vdata : vdst);
            if (i.flat_segment == 1u && saddr != 125u && vaddr == 0u)
                i.src[0] = {};                         // canonical scratch `off, sN`
            else
                i.src[0] = vgpr(vaddr);
            i.src[1] = decode_src_field(saddr);
            i.n_src = 2;
            break;
        }
        case Rdna2Format::MIMG: {
            // Image op. opcode is 8 bits: MSB in dword0 bit 0, low 7 bits in [24:18] (Table 100:
            // "combine bits zero and 18-24" — dropping bit 0 aliased IMAGE_MSAA_LOAD (128) onto
            // IMAGE_LOAD (0) and the _G16/BVH families onto wrong identities); dmask[11:8];
            // dim[5:3]; DLC[7]; unorm[12]; GLC[13]; R128[15]; TFE[16]; LWE[17]; SLC[25].
            // dword1: VADDR base[7:0]; VDATA base[15:8]; SRSRC (T# base SGPR, ×4)[20:16];
            // SSAMP (S# base SGPR, ×4)[25:21]; A16[30]; D16[31]. Retain the reserved holes too,
            // so an unsupported raw packet cannot be mistaken for the ordinary form.
            // image_sample = opcode 0x20, image_load = 0x00. (Bit layout verified via llvm-mc gfx1030.)
            const uint32_t d1 = i.words[1];
            i.opcode     = ((w & 1u) << 7) | ((w >> 18) & 0x7Fu);
            i.mimg_nsa   = (w >> 1)  & 0x3u;
            i.mimg_dmask = (w >> 8)  & 0xFu;
            i.mimg_unorm = (w >> 12) & 0x1u;
            i.mimg_glc   = (w >> 13) & 0x1u;
            i.mimg_dim   = (w >> 3)  & 0x7u;
            i.mimg_dlc   = (w >> 7)  & 0x1u;
            i.mimg_r128  = (w >> 15) & 0x1u;
            i.mimg_tfe   = (w >> 16) & 0x1u;
            i.mimg_lwe   = (w >> 17) & 0x1u;
            i.mimg_slc   = (w >> 25) & 0x1u;
            i.mimg_a16   = (d1 >> 30) & 0x1u;
            i.mimg_d16   = (d1 >> 31) & 0x1u;
            i.mimg_reserved = (w & ((1u << 6) | (1u << 14))) != 0u ||
                              (d1 & (0xFu << 26)) != 0u;
            i.dst    = vgpr(d1 >> 8);                         // VDATA (dest base)
            i.src[0] = vgpr(d1 & 0xFFu);                      // VADDR (coord base VGPR)
            i.src[1] = sgpr(((d1 >> 16) & 0x1Fu) << 2);       // SRSRC (T# base, 8 SGPRs)
            i.src[2] = sgpr(((d1 >> 21) & 0x1Fu) << 2);       // SSAMP (S# base, 4 SGPRs)
            i.n_src  = 3; break;
        }
        case Rdna2Format::DS: {
            // LDS/GDS op. opcode[25:18]; 16-bit byte offset[15:0]; dword1: ADDR[7:0], DATA0[15:8],
            // DATA1[23:16], VDST[31:24]. (Verified via llvm-mc gfx1030: ds_write_b32=0x0d,
            // ds_read_b32=0x36, ds_write_b64=0x4d, ds_write2_b64=0x4e.)
            const uint32_t d1 = i.words[1];
            i.opcode  = (w >> 18) & 0xFFu;
            // GDS flag: llvm-mc gfx1030 places it at bit 17 (ds_add_u32 gds = 0xd8020000 vs
            // 0xd8000000; ds_append gds = 0xd8fa0000 vs plain 0xd8f80000 — Table 94's "GDS [16]"
            // is a GFX9 carryover). Capture bit 16 too so an unknown flag rejects fail-visibly
            // instead of running a device-global op against workgroup LDS.
            i.ds_gds  = ((w >> 16) & 0x3u) != 0;
            i.literal = w & 0xFFFFu;                 // byte offset (offset0:offset1)
            i.src[0]  = vgpr(d1 & 0xFFu);            // ADDR (byte offset into LDS)
            i.src[1]  = vgpr((d1 >> 8) & 0xFFu);     // DATA0 (store source)
            i.src[2]  = vgpr((d1 >> 16) & 0xFFu);    // DATA1 (write2 store source)
            i.dst     = vgpr((d1 >> 24) & 0xFFu);    // VDST (load dest)
            i.n_src = 3; break;
        }
        case Rdna2Format::VINTRP: {
            // Pixel-shader interpolation. opcode[17:16] (p1=0,p2=1,mov=2); vdst[25:18]; attr[15:10];
            // chan[9:8]; vsrc[7:0]. (Bit layout verified via llvm-mc gfx1030.)
            i.opcode      = (w >> 16) & 0x3u;
            i.dst         = vgpr((w >> 18) & 0xFFu);
            i.vintrp_attr = (w >> 10) & 0x3Fu;
            i.vintrp_chan = (w >> 8)  & 0x3u;
            i.src[0]      = vgpr(w & 0xFFu);
            i.n_src = 1; break;
        }
        default: break;   // FLAT operands are not decoded at this stage
    }
}
}  // namespace

Rdna2Inst rdna2_decode_one(const uint32_t* code, size_t max_dwords) {
    Rdna2Inst i;
    if (max_dwords == 0) { i.fmt = Rdna2Format::Unknown; i.len_dwords = 0; return i; }
    const uint32_t w = code[0];
    i.words[0] = w;

    auto two_dword = [&](Rdna2Format f) {
        i.fmt = f;
        i.len_dwords = (max_dwords >= 2) ? 2 : 1;
        if (max_dwords >= 2) i.words[1] = code[1];
    };
    auto one_plus_lit = [&](Rdna2Format f, bool lit) {
        i.fmt = f;
        if (lit && max_dwords >= 2) { i.has_literal = true; i.literal = code[1]; i.len_dwords = 2; }
        else i.len_dwords = 1;
    };

    if ((w & 0x80000000u) == 0u) {
        // Vector ALU group (bit31 == 0): VOP1 (0x7E prefix), VOPC (0x7C prefix), else VOP2.
        Rdna2Format vf = ((w & 0xFE000000u) == 0x7E000000u) ? Rdna2Format::VOP1
                       : ((w & 0xFE000000u) == 0x7C000000u) ? Rdna2Format::VOPC
                                                            : Rdna2Format::VOP2;
        // A VOP src0 field selects an extra dword in four ways: 0xFF = trailing 32-bit literal;
        // 0xF9 = SDWA, 0xFA = DPP16, 0xE9/0xEA = DPP8 = a control word. All make the instruction 2
        // dwords — miss one and the extra dword is mis-decoded as a phantom instruction, derailing the
        // rest of the stream. The SDWA/DPP forms use sub-dword select / cross-lane semantics we don't
        // model, so they are flagged has_modifier and later rejected (vs. silently miscomputed).
        const uint32_t src0 = w & 0x1FFu;
        const bool modifier = (src0 == 0xF9u || src0 == 0xFAu || src0 == 0xE9u || src0 == 0xEAu);
        if (modifier) {
            i.fmt = vf; i.has_modifier = true;
            i.has_sdwa = (src0 == 0xF9u);
            i.len_dwords = (max_dwords >= 2) ? 2 : 1;
            if (max_dwords >= 2) i.words[1] = code[1];
            // DPP16 (src0 == 0xFA) modeled subsets (#273/#1390): dword1 = SRC0[7:0],
            // DPP_CTRL[16:8] (< 0x100 = quad_perm; 0x111..0x11f = row_shr:1..15), FI[18],
            // BC[19], src neg/abs [23:20], bank_mask[27:24], row_mask[31:28].  Full-mask operations
            // without source modifiers/FI can be lowered by a supported shader-stage model,
            // including Astro's fragment row reduction and Plucky's bounded compute shift. Other
            // row operations, partial masks, and modifiers keep has_modifier -> rejected.
            // (Field layout verified against llvm-mc gfx1010 round-trips of DOLL's live words,
            // e.g. 0x7e0802fa 0xff08a002 -> v_mov_b32_dpp v4, v2 quad_perm:[0,0,2,2]
            // row_mask:0xf bank_mask:0xf bound_ctrl:1.)
            if (src0 == 0xFAu && max_dwords >= 2 && vf != Rdna2Format::VOPC) {
                const uint32_t d1 = code[1];
                const uint32_t ctrl = (d1 >> 8) & 0x1FFu;
                const bool modeled_ctrl = ctrl < 0x100u || (ctrl >= 0x111u && ctrl <= 0x11Fu);
                if (modeled_ctrl && ((d1 >> 28) & 0xFu) == 0xFu && ((d1 >> 24) & 0xFu) == 0xFu &&
                    ((d1 >> 20) & 0xFu) == 0u && ((d1 >> 18) & 1u) == 0u) {
                    i.has_modifier = false; i.has_dpp = true; i.dpp_ctrl = (uint16_t)ctrl;
                    i.dpp_bound_ctrl = ((d1 >> 19) & 1u) != 0u;
                }
                // Astro's world-map material PS OR-reduces a lane value across each 16-lane row
                // with row_shr:{1,2,4,8}. BOUND_CTRL=0 retains src0 for an out-of-row fetch, which
                // is exact in the subgroup-shuffle lowering and is also the OR identity here.
                else if (vf == Rdna2Format::VOP2 && ((w >> 25) & 0x3Fu) == 0x1Cu &&
                         ctrl >= 0x111u && ctrl <= 0x11Fu &&
                         ((d1 >> 28) & 0xFu) == 0xFu && ((d1 >> 24) & 0xFu) == 0xFu &&
                         ((d1 >> 20) & 0xFu) == 0u && ((d1 >> 19) & 1u) == 0u &&
                         ((d1 >> 18) & 1u) == 0u) {
                    i.has_modifier = false; i.has_dpp = true; i.dpp_ctrl = (uint16_t)ctrl;
                }
                // Astro Bot's NGG primitive packer performs two bounded row-right shifts before
                // v_add_nc_u32. The one-lane vertex model lowers the absent neighbor to zero; admit
                // only that exact integer-add form with full row/bank masks and BOUND_CTRL=1.
                else if (vf == Rdna2Format::VOP2 && ((w >> 25) & 0x3Fu) == 0x25u &&
                         ctrl >= 0x111u && ctrl <= 0x11Fu &&
                         ((d1 >> 28) & 0xFu) == 0xFu && ((d1 >> 24) & 0xFu) == 0xFu &&
                         ((d1 >> 20) & 0xFu) == 0u && ((d1 >> 19) & 1u) == 1u &&
                         ((d1 >> 18) & 1u) == 0u) {
                    i.has_modifier = false; i.has_dpp = true; i.dpp_ctrl = (uint16_t)ctrl;
                }
                // GTA V selects alternate 16-lane rows after its in-row reduction. QUAD_PERM
                // identity keeps SRC0 in the current lane; ROW_MASK=0xa executes rows 1/3 while
                // rows 0/2 preserve VDST. Admit only the exact live integer-add control form:
                // BC=0, BANK_MASK=0xf, no FI/source modifiers, and retain both masks explicitly.
                else if (vf == Rdna2Format::VOP2 && ((w >> 25) & 0x3Fu) == 0x25u &&
                         ctrl == 0xe4u &&
                         ((d1 >> 28) & 0xFu) == 0xau &&
                         ((d1 >> 24) & 0xFu) == 0xfu &&
                         ((d1 >> 20) & 0xFu) == 0u &&
                         ((d1 >> 19) & 1u) == 0u &&
                         ((d1 >> 18) & 1u) == 0u) {
                    i.has_modifier = false; i.has_dpp = true; i.dpp_ctrl = (uint16_t)ctrl;
                }
                if (i.has_dpp) {
                    i.dpp_bound_ctrl = ((d1 >> 19) & 1u) != 0u;
                    i.dpp_bank_mask = static_cast<uint8_t>((d1 >> 24) & 0xfu);
                    i.dpp_row_mask = static_cast<uint8_t>((d1 >> 28) & 0xfu);
                }
            }
        } else {
            // The six K-carrying VOP2 mul-adds embed a mandatory 32-bit literal K: v_madmk_f32 (0x20),
            // v_madak_f32 (0x21), v_fmamk_f32 (0x2C), v_fmaak_f32 (0x2D), v_fmamk_f16 (0x37),
            // v_fmaak_f16 (0x38 — Table 75 ops 55/56). Miss one and K mis-decodes as a phantom
            // instruction, desyncing the stream. (f32 opcodes verified via round-trip llvm-mc gfx1010.)
            bool lit = (src0 == 0xFFu);
            if (vf == Rdna2Format::VOP2) { uint32_t op = (w >> 25) & 0x3Fu;
                if (op == 0x20u || op == 0x21u || op == 0x2Cu || op == 0x2Du ||
                    op == 0x37u || op == 0x38u) lit = true; }
            one_plus_lit(vf, lit);
        }
    } else if ((w & 0xC0000000u) == 0x80000000u) {
        // Scalar group (bits[31:30] == 10). Carve out SOPP/SOPC/SOP1/SOPK before the SOP2 default.
        if      ((w & 0xFF800000u) == 0xBF800000u) { i.fmt = Rdna2Format::SOPP; i.len_dwords = 1;
                                                     i.is_end = (w == S_ENDPGM); }
        else if ((w & 0xFF800000u) == 0xBF000000u) one_plus_lit(Rdna2Format::SOPC, sop_has_literal(w, 2));
        else if ((w & 0xFF800000u) == 0xBE800000u) one_plus_lit(Rdna2Format::SOP1, sop_has_literal(w, 1));
        else if ((w & 0xF0000000u) == 0xB0000000u)
            // SOPK is 1 dword except S_SETREG_IMM32_B32 (opcode 21), whose 32-bit register data
            // trails as a mandatory literal (Table 66 "SOPK*") — miss it and the data dword
            // re-decodes as a phantom instruction, desyncing every later pc/branch target.
            one_plus_lit(Rdna2Format::SOPK, ((w >> 23) & 0x1Fu) == 21u);
        else                                       one_plus_lit(Rdna2Format::SOP2, sop_has_literal(w, 2));
    } else {
        switch (w >> 26u) {
            case 0x32: i.fmt = Rdna2Format::VINTRP; i.len_dwords = 1; break;
            case 0x33: {   // VOP3P (0xCC prefix): packed / mixed-precision 3-source VALU (v_fma_mix*).
                // 2 dwords + a trailing literal when any 9-bit src field is 0xFF — the same literal
                // rule as VOP3 (a mis-sized VOP3P desynced the whole stream walk into phantom
                // branches, #273). Round-trip llvm-mc gfx1010: v_fma_mixlo_f16 v0,1.0,v0,v5 =
                // 0xcc210000 0x041600f2.
                i.fmt = Rdna2Format::VOP3P;
                if (max_dwords >= 2) {
                    i.words[1] = code[1];
                    const uint32_t d1 = code[1];
                    const bool lit = (d1 & 0x1FFu) == 0xFFu || ((d1 >> 9) & 0x1FFu) == 0xFFu ||
                                     ((d1 >> 18) & 0x1FFu) == 0xFFu;
                    if (lit && max_dwords >= 3) { i.has_literal = true; i.literal = code[2]; i.len_dwords = 3; }
                    else i.len_dwords = 2;
                } else i.len_dwords = 1;
                break;
            }
            case 0x34: case 0x35: {   // VOP3 (0x34 old-gen, 0x35 RDNA2)
                // 2 dwords, plus a trailing 32-bit literal when any of the three 9-bit src operand
                // fields in dword1 ([8:0], [17:9], [26:18]) is 0xFF (the literal marker) — e.g.
                // v_med3/v_add3/v_fma with an immediate. A 0xFF src field unambiguously means literal
                // (256..511 are VGPRs), so this is exact for both VOP3A and VOP3B. Miss it and the
                // literal dword mis-decodes and derails the rest of the stream.
                i.fmt = Rdna2Format::VOP3;
                if (max_dwords >= 2) {
                    i.words[1] = code[1];
                    const uint32_t d1 = code[1];
                    const bool lit = (d1 & 0x1FFu) == 0xFFu || ((d1 >> 9) & 0x1FFu) == 0xFFu ||
                                     ((d1 >> 18) & 0x1FFu) == 0xFFu;
                    if (lit && max_dwords >= 3) { i.has_literal = true; i.literal = code[2]; i.len_dwords = 3; }
                    else i.len_dwords = 2;
                } else i.len_dwords = 1;
                break;
            }
            case 0x36: two_dword(Rdna2Format::DS);    break;
            case 0x37: two_dword(Rdna2Format::FLAT);  break;
            case 0x38: two_dword(Rdna2Format::MUBUF); break;
            case 0x3a: two_dword(Rdna2Format::MTBUF); break;
            case 0x3c: {
                // MIMG is 2 dwords, plus NSA (Non-Sequential Address) extra dwords holding the split
                // address VGPRs. The NSA field (dword0[2:1], 0..3) gives the extra-dword count — miss
                // it and every NSA image op mis-aligns the rest of the stream. One- and two-extra
                // forms were verified against llvm-mc gfx1030; Astro Bot's
                // IMAGE_BVH_INTERSECT_RAY uses all 3 extra dwords (5 total).
                i.fmt = Rdna2Format::MIMG;
                uint32_t extra = (w >> 1) & 0x3u;
                i.len_dwords = 2 + extra;
                if (max_dwords >= 2) i.words[1] = code[1];
                // NSA extra dwords hold the split (non-sequential) address VGPRs, one per byte; keep them
                // so the recompiler can gather the coordinates (dword2 = addr1..4, dword3 = addr5..8,
                // dword4 = addr9..12).
                if (extra >= 1 && max_dwords >= 3) i.words[2] = code[2];
                if (extra >= 2 && max_dwords >= 4) i.words[3] = code[3];
                if (extra >= 3 && max_dwords >= 5) i.words[4] = code[4];
                break;
            }
            case 0x3d: two_dword(Rdna2Format::SMEM);  break;
            case 0x3e: two_dword(Rdna2Format::EXP);   break;
            default:   i.fmt = Rdna2Format::Unknown;  i.len_dwords = 1; break;
        }
    }
    if (i.len_dwords > max_dwords) i.len_dwords = (uint32_t)max_dwords;   // clamp at buffer end
    decode_operands(i);
    return i;
}

size_t rdna2_walk(const uint32_t* code, size_t dwords, std::vector<Rdna2Inst>& out) {
    size_t pc = 0;
    while (pc < dwords) {
        Rdna2Inst i = rdna2_decode_one(code + pc, dwords - pc);
        i.pc = (uint32_t)pc;
        out.push_back(i);
        if (i.len_dwords == 0) break;               // safety: never advance 0
        pc += i.len_dwords;
        if (i.is_end || i.fmt == Rdna2Format::Unknown) break;
    }
    return pc;
}

bool rdna2_mimg_zero_mip_shape(const Rdna2Inst& in, uint32_t* mip_vgpr) {
    if (in.fmt != Rdna2Format::MIMG || !in.mimg_unorm || !in.mimg_glc ||
        in.mimg_dlc || in.mimg_r128 ||
        in.mimg_tfe || in.mimg_lwe || in.mimg_slc || in.mimg_a16 || in.mimg_d16 ||
        in.mimg_reserved || in.src[0].kind != OperandKind::VGPR || in.src[0].value < 0)
        return false;

    uint32_t reg = 0;
    if (in.opcode == 0x01u && (in.mimg_dmask == 1u || in.mimg_dmask == 0xfu) &&
        (in.mimg_dim == 1u || in.mimg_dim == 5u) &&
        in.mimg_nsa == 0u && in.len_dwords == 2u) {
        // IMAGE_LOAD_MIP: 2D=[x,y,mip], 2D_ARRAY=[x,y,slice,mip].
        reg = static_cast<uint32_t>(in.src[0].value) +
              (in.mimg_dim == 5u ? 3u : 2u);
    } else if (in.opcode == 0x09u && (in.mimg_dmask == 1u || in.mimg_dmask == 0xfu) &&
               in.mimg_dim == 1u &&
               in.mimg_nsa == 1u && in.len_dwords == 3u &&
               (in.words[2] & 0xffff0000u) == 0u) {
        // IMAGE_STORE_MIP NSA 2D: coord0 is VADDR, then word2 bytes name y and mip.
        reg = (in.words[2] >> 8) & 0xffu;
    } else {
        return false;
    }
    if (reg >= 256u) return false;
    if (mip_vgpr) *mip_vgpr = reg;
    return true;
}

uint32_t rdna2_sload_required_bytes(const uint32_t* code, size_t dwords, uint32_t sgpr_base) {
    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code, dwords, instructions);
    uint64_t required = 0;
    for (const auto& in : instructions) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SMEM || in.opcode > 0x04u ||
            in.src[0].value != static_cast<int32_t>(sgpr_base) ||
            in.src[1].kind != OperandKind::Special || in.src[1].value != 125 ||
            static_cast<int32_t>(in.literal) < 0)
            continue;
        uint32_t words = 1;
        if (in.opcode == 0x01u) words = 2;
        else if (in.opcode == 0x02u) words = 4;
        else if (in.opcode == 0x03u) words = 8;
        else if (in.opcode == 0x04u) words = 16;
        required = std::max<uint64_t>(required, static_cast<uint64_t>(in.literal) + words * 4u);
    }
    return required > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(required);
}

} // namespace prosper::gpu
