// VOP2 f16 operation stage. Case bodies moved by extract_case_routes.py.
#include "gpu/recompiler/ir/lower/vector_lowering_internal.hpp"

namespace prosper::gpu::ir {

bool Vop2Lowering::compute_f16(uint32_t& d) {
    switch (in.opcode) {
        case 0x2F:
            d = b.pack_half2x16_rtz(a, c);
            break;      // v_cvt_pkrtz_f16_f32 (e32 form): RTZ clamp (#452)
        case 0x3C: {   // v_pk_fmac_f16: packed dst += src0*src1
            auto fmac_half = [&](uint32_t half) {
                return b.fbin(Op_FAdd,
                              b.fbin(Op_FMul, b.unpack_half(a, half), b.unpack_half(c, half)),
                              b.unpack_half(old_d, half));
            };
            uint32_t lo = b.pack_half_lo(fmac_half(0));
            uint32_t hi = b.ibin(Op_ShiftLeftLogical, b.pack_half_lo(fmac_half(1)), b.uconst(16));
            d = b.ibin(Op_BitwiseOr, lo, hi);
            break;
        }
        case 0x32:
        case 0x33:
        case 0x35:
        case 0x39:
        case 0x3A: {   // v_add/sub/mul/max/min_f16
            // (SDWA WORD_1 = high 16; DWORD/WORD_0 = low 16 — an f16 op reads bits[15:0]);
            // f16xf16 products are exact in f32, so multiply in f32 and round once to f16.
            // The 16-bit result inserts into the selected dest half PRESERVING the other
            // (dst_sel WORD_1 for the SDWA pack idiom; DWORD/WORD_0 = the plain e32 form's
            // "write [15:0], preserve [31:16]" gfx10 f16-VOP2 contract). #273 (DOLL box-blur).
            auto source = [&](uint32_t raw, const Operand& operand, uint8_t sel, int k) {
                // An inline constant in a 16-bit operand carries its f16-width encoding
                // (float — exact even for 1/(2*pi)) or raw two's-complement bits (int:
                // inline 1 is the f16 denormal 0x0001, never 1.0). Register/scalar operands
                // carry packed halves and must be selected before abs/neg.
                uint32_t v = operand.kind == OperandKind::InlineFloat
                                 ? b.unpack_half(b.uconst(inline_float_f16_bits(operand.value)), 0)
                             : operand.kind == OperandKind::InlineInt
                                 ? b.unpack_half(b.uconst(static_cast<uint32_t>(operand.value)),
                                                 sel == 5 ? 1 : 0)
                                 : b.unpack_half(raw, sel == 5 ? 1 : 0);
                if (in.src_abs[k]) v = b.fext1(Glsl_FAbs, v);
                if (in.src_neg[k]) v = b.fneg(v);
                return v;
            };
            uint32_t x = source(a, in.src[0], in.sdwa_src0_sel, 0);
            uint32_t y = source(c, in.src[1], in.sdwa_src1_sel, 1);
            uint32_t p = in.opcode == 0x32   ? b.fbin(Op_FAdd, x, y)
                         : in.opcode == 0x33 ? b.fbin(Op_FSub, x, y)
                         : in.opcode == 0x35 ? b.fbin(Op_FMul, x, y)
                         : in.opcode == 0x39 ? b.fext2(Glsl_NMax, x, y)   // NaN -> other operand
                                             : b.fext2(Glsl_NMin, x, y);
            if (in.clamp) p = b.clamp01(p);
            uint32_t r16 = b.pack_half_lo(p);
            // dst_sel==6 covers TWO encodings the decoder now distinguishes: the SDWA
            // DWORD+UNUSED_PAD form (zero-fill, has_sdwa) and the plain e32 form, whose
            // gfx10 contract writes [15:0] and PRESERVES [31:16] (the comment above always
            // said so; the code zero-filled both, corrupting live packed high halves).
            d = in.sdwa_dst_sel == 6
                    ? (in.has_sdwa
                           ? r16
                           : b.ibin(Op_BitwiseOr,
                                    b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16))
                : (in.sdwa_dst_sel == 5)
                    ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                             b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
                    : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)),
                             r16);
            break;
        }
        default: return false;
    }
    return true;
}

}   // namespace prosper::gpu::ir
