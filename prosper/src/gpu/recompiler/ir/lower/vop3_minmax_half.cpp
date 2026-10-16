// Native half extrema mechanically relocated; source selection and destination merging stay here.
#include "gpu/recompiler/ir/lower/vop3_minmax_half_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_minmax_half_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
bool Vop3MinmaxHalfLowering::lower() {
    if (in.opcode != 0x351 && in.opcode != 0x354 && in.opcode != 0x357) return false;
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
    uint32_t result = 0;
    if (in.opcode == 0x351) {                     // v_min3_f16
        // ISA op 849 is a plain nested min — `V_MIN_F16(V_MIN_F16(S0,S1),S2)` — with
        // NO any-NaN clause of the kind op 855 v_med3_f16 carries. NMin already
        // matches V_MIN_F16's own one-NaN rule (return the other operand), so the
        // bare chain is the whole contract. CONFIDENCE: HIGH.
        result = b.fext2(Glsl_NMin, b.fext2(Glsl_NMin, f16src(0), f16src(1)), f16src(2));
    } else if (in.opcode == 0x354) {                     // v_max3_f16
        // ISA op 852, likewise: `V_MAX_F16(V_MAX_F16(S0,S1),S2)`, no NaN clause.
        result = b.fext2(Glsl_NMax, b.fext2(Glsl_NMax, f16src(0), f16src(1)), f16src(2));
    } else {                                             // v_med3_f16
        // ISA op 855 opens with a SEPARATE any-NaN clause, exactly as op 343
        // v_med3_f32 does (see 0x157 below — same structure, only the width differs):
        //   if (isNan(S0) || isNan(S1) || isNan(S2)) D.f16 = V_MIN3_F16(S0,S1,S2)
        // The plain max(min(...)) median formula does NOT reproduce that fallback on
        // its own, so the clause is emitted explicitly. It is not academic: Sonic
        // Racing: CrossWorlds' NaN-safe saturate idiom `v_med3_f16 v3, 1.0, v5, 0`
        // (#2013) with S1 = NaN must yield MIN3(1.0,NaN,0) = 0.0h, where the bare
        // formula yields 1.0h — black rendered as white. Kernel 32r10n pins it.
        // The min/max legs use NaN-aware NMin/NMax because V_MIN_F16/V_MAX_F16 return
        // the OTHER operand when exactly one input is NaN. CONFIDENCE: HIGH.
        const uint32_t x = f16src(0), y = f16src(1), z = f16src(2);
        const uint32_t mn = b.fext2(Glsl_NMin, x, y), mx = b.fext2(Glsl_NMax, x, y);
        const uint32_t med = b.fext2(Glsl_NMax, mn, b.fext2(Glsl_NMin, mx, z));
        const uint32_t min3 = b.fext2(Glsl_NMin, mn, z);
        const uint32_t nan_any =
            b.lor(b.fcmp(Op_FUnordNotEqual, x, x),
                  b.lor(b.fcmp(Op_FUnordNotEqual, y, y), b.fcmp(Op_FUnordNotEqual, z, z)));
        result = b.sel(nan_any, min3, med);
    }
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
