// Native half extrema mechanically relocated; source selection and destination merging stay here.
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/lower/vop3_minmax_half_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_minmax_half_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/ir/ops/vector_minmax_half.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
bool Vop3MinmaxHalfLowering::lower() {
    const auto* def = find_vector_minmax_half_op(in.fmt, in.opcode);
    if (!def) return false;
    if (!fresult) {
        std::fprintf(stderr, "[ir] half extrema callback missing\n");
        std::abort();
    }
    // Scalar f16 VOP3: one selected half from each packed source and one independently
    // selected destination half. The opposite destination half is preserved. The
    // `v_max3_f16` occurs eight times in Syberia's nonzero 960x544 gameplay composite
    // dispatch; llvm-mc gfx1030 identifies opcode 0x354 exactly, and the same round trip
    // names its two siblings 0x351 v_min3_f16 and 0x357 v_med3_f16 — Sonic Racing:
    // CrossWorlds' upscale kernel emits `d7571003,02020af2`
    // = `v_med3_f16 v3, 1.0, v5, 0 op_sel:[0,1,0,0]` (#2013).
    auto f16src = [&](int k) {
        const Operand& operand = in.src[k];
        uint32_t raw = val(operand);
        // Inline constants supply the dword their operand width produces (see
        // inline_16bit_operand_dword) and OPSEL[k] selects a half of THAT — a float
        // reads 0.0h from the high half rather than the constant again (#2119,
        // kernel 32r14c).
        const uint32_t sel = (in.vop3p_opsel >> k) & 1u;
        uint32_t inline_dword;
        uint32_t value = inline_16bit_operand_dword(operand, inline_dword)
                             ? b.unpack_half(b.uconst(inline_dword), sel)
                             : b.unpack_half(raw, sel);
        if (in.src_abs[k]) value = b.fext1(Glsl_FAbs, value);
        if (in.src_neg[k]) value = b.fneg(value);
        return value;
    };
    uint32_t result =
        emit_ternary_integer_op(b, *def, [&](uint32_t k) { return f16src(static_cast<int>(k)); });
    result = fresult(result);
    uint32_t r16 = b.pack_half_lo(result);
    vreg[in.dst.value] =
        (in.vop3p_opsel & 8u)
            ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                     b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
            : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16);
    return true;
}
bool lower_vop3_minmax_half_erased(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                   bool allow_wave, uint32_t old_d,
                                   const MinmaxHalfResult& fresult) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3MinmaxHalfLowering({b, rs, in, ok, allow_wave}, old_d, fresult).lower();
}
}   // namespace prosper::gpu::ir
