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
            i.src[0] = decode_src_field(w & 0x1FFu); i.n_src = 1; break;
        case Rdna2Format::VOP2:
            i.opcode = (w >> 25) & 0x3Fu; i.dst = vgpr(w >> 17);
            i.src[0] = decode_src_field(w & 0x1FFu); i.src[1] = vgpr(w >> 9); i.n_src = 2; break;
        case Rdna2Format::VOPC:
            i.opcode = (w >> 17) & 0xFFu; i.dst = {OperandKind::Special, 106 /*VCC_LO*/};
            i.src[0] = decode_src_field(w & 0x1FFu); i.src[1] = vgpr(w >> 9); i.n_src = 2; break;
        case Rdna2Format::VOP3: {
            const uint32_t d1 = i.words[1];
            i.opcode = (w >> 16) & 0x3FFu; i.dst = vgpr(w);   // VOP3A: vdst in [7:0] of dword0
            i.src[0] = decode_src_field(d1 & 0x1FFu);
            i.src[1] = decode_src_field((d1 >> 9) & 0x1FFu);
            i.src[2] = decode_src_field((d1 >> 18) & 0x1FFu); i.n_src = 3; break;
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
        default: break;   // MUBUF/MTBUF/MIMG/DS/FLAT/VINTRP: operands not decoded at this stage
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
        if ((w & 0xFE000000u) == 0x7E000000u)      one_plus_lit(Rdna2Format::VOP1, vop_has_literal(w));
        else if ((w & 0xFE000000u) == 0x7C000000u) one_plus_lit(Rdna2Format::VOPC, vop_has_literal(w));
        else                                       one_plus_lit(Rdna2Format::VOP2, vop_has_literal(w));
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
            case 0x34: case 0x35: two_dword(Rdna2Format::VOP3);  break;   // 0x34 old-gen, 0x35 RDNA2
            case 0x36: two_dword(Rdna2Format::DS);    break;
            case 0x37: two_dword(Rdna2Format::FLAT);  break;
            case 0x38: two_dword(Rdna2Format::MUBUF); break;
            case 0x3a: two_dword(Rdna2Format::MTBUF); break;
            case 0x3c: two_dword(Rdna2Format::MIMG);  break;
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
