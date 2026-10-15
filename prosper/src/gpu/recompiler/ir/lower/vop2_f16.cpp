// Half-result source selection and destination placement; operation semantics live in ops/.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/ir/lower/vector_lowering_internal.hpp"
#include "gpu/recompiler/ir/ops/vector_half.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
uint32_t Vop2Lowering::half_source(uint32_t raw, const Operand& operand, uint8_t sel, int k) {
    // An inline constant in a 16-bit operand carries its f16-width encoding
    // (float — exact even for 1/(2*pi)) or raw two's-complement bits (int:
    // inline 1 is the f16 denormal 0x0001, never 1.0). Register/scalar operands
    // carry packed halves and must be selected before abs/neg.
    uint32_t v =
        operand.kind == OperandKind::InlineFloat
            ? b.unpack_half(b.uconst(inline_float_f16_bits(operand.value)), 0)
        : operand.kind == OperandKind::InlineInt
            ? b.unpack_half(b.uconst(static_cast<uint32_t>(operand.value)), sel == 5 ? 1 : 0)
            : b.unpack_half(raw, sel == 5 ? 1 : 0);
    if (in.src_abs[k]) v = b.fext1(Glsl_FAbs, v);
    if (in.src_neg[k]) v = b.fneg(v);
    return v;
}
bool Vop2Lowering::compute_f16(uint32_t& d) {
    if (const HalfOpDef* def = find_vector_half_op(in.fmt, in.opcode)) {
        if (!rdna2_instruction_effects(in.fmt, in.opcode).vdst_half) {
            d = emit_half_op(b, *def, a, c);
            return true;
        }
        const uint32_t x = half_source(a, in.src[0], in.sdwa_src0_sel, 0);
        const uint32_t y = half_source(c, in.src[1], in.sdwa_src1_sel, 1);
        const uint32_t r16 = emit_half_op(b, *def, x, y, in.clamp);
            // dst_sel==6 covers TWO encodings the decoder now distinguishes: the SDWA
            // DWORD+UNUSED_PAD form (zero-fill, has_sdwa) and the plain e32 form, whose
            // gfx10 contract writes [15:0] and PRESERVES [31:16] (the comment above always
            // said so; the code zero-filled both, corrupting live packed high halves).
        d = in.sdwa_dst_sel == 6
                ? (in.has_sdwa ? r16
                               : b.ibin(Op_BitwiseOr,
                                        b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16))
            : (in.sdwa_dst_sel == 5)
                ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                         b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
                : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16);

        return true;
    }
    switch (in.opcode) {
        // CONFIDENCE: HIGH: this legacy F32 intermediate double-rounds the half result; #5225.
        // Its body is preserved; no new IR definition claims it implements fused half arithmetic.
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

        default: return false;
    }
    return true;
}
}   // namespace prosper::gpu::ir
