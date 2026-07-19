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
                if (((sd >> 8) & 7u) == 6u && ((sd >> 16) & 7u) == 6u &&
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
                    // BYTE-select move (#273 — byte unpack): dst DWORD + UNUSED_PAD, src0 BYTE_0..3,
                    // no sext/neg/abs/clamp/omod. The recompiler zero-extends the selected byte.
                    // (llvm-mc gfx1030: 0x7e0c02f9 0x0000060f -> v_mov_b32_sdwa v6, v15
                    // dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0.)
                    else if (dsel == 6u && dun == 0u && s0sel <= 3u &&
                             !((sd >> 19) & 0x9u) && !i.clamp && !i.omod && !i.src_neg[0] && !i.src_abs[0]) {
                        i.sdwa_dst_sel = (uint8_t)dsel; i.sdwa_dst_unused = (uint8_t)dun;
                        i.sdwa_src0_sel = (uint8_t)s0sel;
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
                // Packed unary f16 SDWA selects one source half, writes one destination half, and
                // preserves the other. The producer uses rcp/sqrt/cos (0x54/0x55/0x61).
                else if (((w >> 9) & 0xFFu) == 0x54u ||
                         ((w >> 9) & 0xFFu) == 0x55u ||
                         ((w >> 9) & 0xFFu) == 0x61u) {
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
                // WORD-dst v_mul_f16_sdwa (#273 — the f16 half-packing idiom): dst WORD_0/1 with
                // UNUSED_PRESERVE, src sels WORD/DWORD, no sext/neg/abs/clamp/omod. (llvm-mc gfx1010:
                // 0x6a0000f9 0x0686156a -> v_mul_f16_sdwa v0, vcc_lo, v0 dst_sel:WORD_1
                // dst_unused:UNUSED_PRESERVE — DOLL's box-blur tail.)
                else if (((w >> 25) & 0x3Fu) == 0x32u ||
                         ((w >> 25) & 0x3Fu) == 0x33u ||
                         ((w >> 25) & 0x3Fu) == 0x35u ||
                         ((w >> 25) & 0x3Fu) == 0x39u ||
                         ((w >> 25) & 0x3Fu) == 0x3Au) {
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
                else if (((w >> 25) & 0x3Fu) == 0x01u) {
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
                    const bool cmpx = (i.opcode >= 0x10u && i.opcode <= 0x1Fu) ||
                                      (i.opcode >= 0x90u && i.opcode <= 0x9Fu) ||
                                      (i.opcode >= 0xD0u && i.opcode <= 0xDFu);
                    const uint32_t eff = cmpx ? i.opcode - 0x10u : i.opcode;
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
                    const bool integer_compare =
                        (eff >= 0x81u && eff <= 0x86u) ||
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
            // v_fma_f16 VOP3: OPSEL[2:0] selects each packed source half and OPSEL[3]
            // selects the destination half. Reuse the packed-op selector field for this family.
            if (i.opcode == 0x34Bu)
                i.vop3p_opsel = static_cast<uint8_t>((w >> 11) & 0xFu);
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
            // Packed f16 add/mul (0x0f/0x10) and v_fma_mix_f32/mixlo/mixhi (0x20-0x22): every
            // modifier bit is MODELED (#273). For packed ops OPSEL/NEG select and negate each source
            // independently for the low result; OPSEL_HI/NEG_HI do the same for the high result.
            // For the mix family:
            // OPSEL_HI[k] selects an f16-half read (which half via OPSEL[k]), NEG negates, NEG_HI
            // is ABS for the mix family, CLAMP saturates. (llvm-mc gfx1030 round-trip on the live
            // DOLL bytes: 0xcc200044 0x9a02170b = v_fma_mix_f32 v68, v11, v11, neg(0)
            // op_sel_hi:[1,1,0].) Other VOP3P ops (v_pk_*, per-half packed semantics) still reject
            // on any modifier bit.
            if (i.opcode == 0x0F || i.opcode == 0x10) {
                const uint32_t neg = (d1 >> 29) & 7u;
                for (int k = 0; k < 3; ++k) i.src_neg[k] = ((neg >> k) & 1u) != 0;
                i.vop3p_neg_hi  = static_cast<uint8_t>((w >> 8) & 7u);
                i.vop3p_opsel   = static_cast<uint8_t>((w >> 11) & 7u);
                i.vop3p_opsel_hi = static_cast<uint8_t>(((d1 >> 27) & 3u) | (((w >> 14) & 1u) << 2));
                i.clamp = ((w >> 15) & 1u) != 0;
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
            i.mubuf_glc = ((w >> 14) & 1u) != 0;
            i.dst    = vgpr(d1 >> 8);
            i.src[0] = vgpr(d1);
            i.src[1] = sgpr(((d1 >> 16) & 0x1Fu) << 2);
            i.src[2] = decode_src_field((d1 >> 24) & 0xFFu);
            i.literal = (w & 0xFFFu) | (((w >> 12) & 1u) << 12) |
                        (((w >> 13) & 1u) << 13);
            i.n_src = 3; break;
        }
        case Rdna2Format::MIMG: {
            // Image op. opcode is 8 bits: MSB in dword0 bit 0, low 7 bits in [24:18] (Table 100:
            // "combine bits zero and 18-24" — dropping bit 0 aliased IMAGE_MSAA_LOAD (128) onto
            // IMAGE_LOAD (0) and the _G16/BVH families onto wrong identities); dmask[11:8];
            // unorm[12]; dim[5:3]. dword1: VADDR base[7:0]; VDATA base[15:8]; SRSRC (T# base
            // SGPR, ×4)[20:16]; SSAMP (S# base SGPR, ×4)[25:21].
            // image_sample = opcode 0x20, image_load = 0x00. (Bit layout verified via llvm-mc gfx1030.)
            const uint32_t d1 = i.words[1];
            i.opcode     = ((w & 1u) << 7) | ((w >> 18) & 0x7Fu);
            i.mimg_dmask = (w >> 8)  & 0xFu;
            i.mimg_unorm = (w >> 12) & 0x1u;
            i.mimg_dim   = (w >> 3)  & 0x7u;
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
            // DPP16 (src0 == 0xFA) QUAD_PERM subset (#273): dword1 = SRC0[7:0], DPP_CTRL[16:8]
            // (< 0x100 = quad_perm), FI[18], BC[19], src neg/abs [23:20], bank_mask[27:24],
            // row_mask[31:28]. A full-quad permute (masks 0xf, no neg/abs/FI) is handleable — the
            // recompiler reconstructs the lane value from screen-space derivatives. Anything else
            // (row shifts/broadcasts, partial masks, modifiers) keeps has_modifier -> rejected.
            // (Field layout verified against llvm-mc gfx1010 round-trips of DOLL's live words,
            // e.g. 0x7e0802fa 0xff08a002 -> v_mov_b32_dpp v4, v2 quad_perm:[0,0,2,2]
            // row_mask:0xf bank_mask:0xf bound_ctrl:1.)
            if (src0 == 0xFAu && max_dwords >= 2 && vf != Rdna2Format::VOPC) {
                const uint32_t d1 = code[1];
                const uint32_t ctrl = (d1 >> 8) & 0x1FFu;
                if (ctrl < 0x100u && ((d1 >> 28) & 0xFu) == 0xFu && ((d1 >> 24) & 0xFu) == 0xFu &&
                    ((d1 >> 20) & 0xFu) == 0u && ((d1 >> 18) & 1u) == 0u) {
                    i.has_modifier = false; i.has_dpp = true; i.dpp_ctrl = (uint16_t)ctrl;
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
                // it and every NSA image op mis-aligns the rest of the stream. (Verified against
                // llvm-mc gfx1030: 2D non-NSA=2 dw, NSA 1-extra=3 dw, NSA 2-extra=4 dw.)
                i.fmt = Rdna2Format::MIMG;
                uint32_t extra = (w >> 1) & 0x3u;
                i.len_dwords = 2 + extra;
                if (max_dwords >= 2) i.words[1] = code[1];
                // NSA extra dwords hold the split (non-sequential) address VGPRs, one per byte; keep them
                // so the recompiler can gather the coords (dword2 = addr1..4, dword3 = addr5..8).
                if (extra >= 1 && max_dwords >= 3) i.words[2] = code[2];
                if (extra >= 2 && max_dwords >= 4) i.words[3] = code[3];
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
