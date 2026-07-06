// rdna2_decode.cpp — see rdna2_decode.hpp.
#include "rdna2_decode.hpp"

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
                // "Trivial" SDWA carries no sub-dword effect (all sels = DWORD(6), no clamp/omod/src0 mods),
                // so it equals the base VOP1 (with SGPR-capable operands). Un-flag it so the recompiler
                // handles it normally; any real sub-dword select keeps has_modifier and is rejected.
                if (((sd >> 8) & 7u) == 6u && ((sd >> 16) & 7u) == 6u &&
                    !((sd >> 13) & 1u) && !((sd >> 14) & 3u) && !((sd >> 19) & 0xFu)) i.has_modifier = false;
            } else { i.src[0] = decode_src_field(w & 0x1FFu); i.n_src = 1; }
            break;
        case Rdna2Format::VOP2:
            i.opcode = (w >> 25) & 0x3Fu; i.dst = vgpr(w >> 17);
            if ((w & 0x1FFu) == 0xF9u) {                          // SDWA (src0 in ctrl dword; src1 in dword0)
                const uint32_t sd = i.words[1];
                i.src[0] = ((sd >> 23) & 1u) ? decode_src_field(sd & 0xFFu)     : vgpr(sd & 0xFFu);       // S0
                i.src[1] = ((sd >> 31) & 1u) ? decode_src_field((w >> 9) & 0xFFu) : vgpr((w >> 9) & 0xFFu); // S1
                i.n_src = 2;
                // Trivial (no sub-dword effect): dst/src0/src1 sels all DWORD(6), no clamp/omod/src mods.
                if (((sd >> 8) & 7u) == 6u && ((sd >> 16) & 7u) == 6u && ((sd >> 24) & 7u) == 6u &&
                    !((sd >> 13) & 1u) && !((sd >> 14) & 3u) && !((sd >> 19) & 0xFu) && !((sd >> 27) & 0xFu))
                    i.has_modifier = false;
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
                // VOPC SDWA has NO dst_sel/clamp/omod (byte1 holds SDST/SD, not a VGPR dst_sel); trivial
                // iff both source selects are DWORD and no source modifiers.
                if (((sd >> 16) & 7u) == 6u && ((sd >> 24) & 7u) == 6u &&
                    !((sd >> 19) & 0xFu) && !((sd >> 27) & 0xFu))
                    i.has_modifier = false;
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
        case Rdna2Format::SMEM:
            // Scalar memory load. opcode[25:18]; SDATA (dest SGPR) [12:6]; SBASE (SGPR *pair*, field
            // is the pair index so ×2) [5:0]; 21-bit immediate byte OFFSET in dword1[20:0] (stored in
            // `literal`). Register-offset (SOFFSET) and buffer descriptors are not decoded yet.
            i.opcode = (w >> 18) & 0xFFu;
            i.dst    = sgpr(w >> 6);                 // SDATA
            i.src[0] = sgpr((w & 0x3Fu) << 1);       // SBASE (pair base)
            i.literal = i.words[1] & 0x1FFFFFu;       // immediate byte offset (not a trailing constant)
            i.n_src = 1; break;
        case Rdna2Format::MUBUF: {
            // Untyped buffer op. opcode[24:18]; VDATA (dest/src VGPR) d1[15:8]; VADDR d1[7:0];
            // SRSRC (V# descriptor base SGPR, field ×4) d1[20:16]; SOFFSET d1[30:24]. 12-bit inst
            // offset d0[11:0] + offen d0[12] + idxen d0[13] packed into `literal`.
            const uint32_t d1 = i.words[1];
            i.opcode = (w >> 18) & 0x7Fu;
            i.dst    = vgpr(d1 >> 8);                          // VDATA
            i.src[0] = vgpr(d1);                              // VADDR
            i.src[1] = sgpr(((d1 >> 16) & 0x1Fu) << 2);       // SRSRC (descriptor base)
            i.src[2] = decode_src_field((d1 >> 24) & 0x7Fu);  // SOFFSET
            i.literal = (w & 0xFFFu) | (((w >> 12) & 1u) << 12) | (((w >> 13) & 1u) << 13);
            i.n_src = 3; break;
        }
        case Rdna2Format::MIMG: {
            // Image op. opcode[24:18]; dmask[11:8]; unorm[12]; dim[5:3]. dword1: VADDR base[7:0];
            // VDATA base[15:8]; SRSRC (T# base SGPR, ×4)[20:16]; SSAMP (S# base SGPR, ×4)[25:21].
            // image_sample = opcode 0x20, image_load = 0x00. (Bit layout verified via llvm-mc gfx1030.)
            const uint32_t d1 = i.words[1];
            i.opcode     = (w >> 18) & 0x7Fu;
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
            // DATA1[23:16], VDST[31:24]. (Verified via llvm-mc gfx1030: ds_write_b32=0x0d, ds_read_b32=0x36.)
            const uint32_t d1 = i.words[1];
            i.opcode  = (w >> 18) & 0xFFu;
            i.literal = w & 0xFFFFu;                 // byte offset (offset0:offset1)
            i.src[0]  = vgpr(d1 & 0xFFu);            // ADDR (byte offset into LDS)
            i.src[1]  = vgpr((d1 >> 8) & 0xFFu);     // DATA0 (store source)
            i.dst     = vgpr((d1 >> 24) & 0xFFu);    // VDST (load dest)
            i.n_src = 2; break;
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
        default: break;   // MTBUF/DS/FLAT: operands not decoded at this stage
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
            i.len_dwords = (max_dwords >= 2) ? 2 : 1;
            if (max_dwords >= 2) i.words[1] = code[1];
        } else {
            // The four K-carrying VOP2 mul-adds embed a mandatory 32-bit literal K: v_madmk_f32 (0x20),
            // v_madak_f32 (0x21), v_fmamk_f32 (0x2C), v_fmaak_f32 (0x2D). Miss it and K mis-decodes as a
            // phantom instruction, desyncing the stream. (Verified opcodes via round-trip llvm-mc gfx1010.)
            bool lit = (src0 == 0xFFu);
            if (vf == Rdna2Format::VOP2) { uint32_t op = (w >> 25) & 0x3Fu;
                if (op == 0x20u || op == 0x21u || op == 0x2Cu || op == 0x2Du) lit = true; }
            one_plus_lit(vf, lit);
        }
    } else if ((w & 0xC0000000u) == 0x80000000u) {
        // Scalar group (bits[31:30] == 10). Carve out SOPP/SOPC/SOP1/SOPK before the SOP2 default.
        if      ((w & 0xFF800000u) == 0xBF800000u) { i.fmt = Rdna2Format::SOPP; i.len_dwords = 1;
                                                     i.is_end = (w == S_ENDPGM); }
        else if ((w & 0xFF800000u) == 0xBF000000u) one_plus_lit(Rdna2Format::SOPC, sop_has_literal(w, 2));
        else if ((w & 0xFF800000u) == 0xBE800000u) one_plus_lit(Rdna2Format::SOP1, sop_has_literal(w, 1));
        else if ((w & 0xF0000000u) == 0xB0000000u) { i.fmt = Rdna2Format::SOPK; i.len_dwords = 1; }
        else                                       one_plus_lit(Rdna2Format::SOP2, sop_has_literal(w, 2));
    } else {
        switch (w >> 26u) {
            case 0x32: i.fmt = Rdna2Format::VINTRP; i.len_dwords = 1; break;
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

} // namespace prosper::gpu
