// Selected-half wrapping integer values use shared IR definitions. Source handling and
// opposite-half preservation retain their original positions; the caller owns CLAMP and EXEC.
#include "gpu/recompiler/ir/lower/vop3_integer16_lowering_internal.hpp"
#include "gpu/recompiler/ir/lower/vop3_integer16_lowering.hpp"
#include "gpu/recompiler/ir/ops/vector_integer16.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
uint32_t Vop3Integer16Lowering::u16src(int k) {
    const Operand& operand = in.src[k];
    // A 16-bit operand position carries the 16-bit encoding of an inline
    // constant, not the f32 pattern val() would materialize — and OPSEL[k]
    // selects a half of THAT dword (see inline_16bit_operand_dword). Both
    // inline branches used to ignore the select, so a positive inline int read
    // its own value from the HIGH half where hardware reads 0 (#2119, kernel
    // 32r14e). Negative inline ints were already right by accident: -N is
    // sign-extended, so both halves are 0xffff.
    const uint32_t sel = (in.vop3p_opsel >> k) & 1u;
    uint32_t inline_dword;
    if (inline_16bit_operand_dword(operand, inline_dword))
        return b.uconst((inline_dword >> (16u * sel)) & 0xFFFFu);
    const uint32_t raw = val(operand);
    return b.ibin(Op_BitwiseAnd, sel ? b.ibin(Op_ShiftRightLogical, raw, b.uconst(16)) : raw,
                  b.uconst(0xFFFFu));
}
bool Vop3Integer16Lowering::lower() {
    const auto* def = find_vector_integer16_op(in.fmt, in.opcode);
    if (!def) return false;
    if (in.src_abs[0] || in.src_abs[1] || in.src_abs[2] || in.src_neg[0] || in.src_neg[1] ||
        in.src_neg[2]) {
        ok = false;
        return true;
    }
    const uint32_t s0 = u16src(0), s1 = u16src(1);
    const uint32_t r16 = emit_ternary_integer_op(b, *def, [&](uint32_t slot) {
        return slot == 0 ? s0 : slot == 1 ? s1 : u16src(2);
    });
    vreg[in.dst.value] =
        (in.vop3p_opsel & 8u)
            ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                     b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
            : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16);
    return true;
}
bool lower_vop3_integer16(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                          bool allow_wave, uint32_t old_d) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3Integer16Lowering({b, rs, in, ok, allow_wave}, old_d).lower();
}
} // namespace prosper::gpu::ir
