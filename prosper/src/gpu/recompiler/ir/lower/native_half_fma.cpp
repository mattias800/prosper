// Native half FMA numeric body, mechanically relocated without changing the old graph.
#include "gpu/recompiler/ir/lower/native_half_fma_lowering.hpp"
#include "gpu/recompiler/ir/lower/half_fma_finite.hpp"
#include <array>
#include "gpu/recompiler/ir/lower/native_half_fma_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
void NativeHalfFmaLowering::lower() {
    // Scalar f16 VOP3: one selected half from each packed source and one independently
    // selected destination half. The opposite destination half is preserved. The
    // `v_max3_f16` occurs eight times in Syberia's nonzero 960x544 gameplay composite
    // dispatch; llvm-mc gfx1030 identifies opcode 0x354 exactly, and the same round trip
    // names its two siblings 0x351 v_min3_f16 and 0x357 v_med3_f16 — Sonic Racing:
    // CrossWorlds' upscale kernel emits `d7571003,02020af2`
    // = `v_med3_f16 v3, 1.0, v5, 0 op_sel:[0,1,0,0]` (#2013).
    std::array<uint32_t, 3> raw_sources{};
    auto f16src = [&](int k) {
        const Operand& operand = in.src[k];
        uint32_t raw = val(operand);
        raw_sources[k] = raw;
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
    {   // Deferred fused-FMA behavior stays in the legacy path (#5225).
        result = b.fbin(Op_FAdd, b.fbin(Op_FMul, f16src(0), f16src(1)), f16src(2));
    }
    result = fresult(result);
    uint32_t r16 = b.pack_half_lo(result);
    if (ok && use_half_fma_finite(b, in))
        r16 = emit_native_half_fma_finite(b, in, raw_sources, r16);
    vreg[in.dst.value] =
        (in.vop3p_opsel & 8u)
            ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                     b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
            : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16);
}
void lower_native_half_fma(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                           bool allow_wave, uint32_t old_d, const NativeHalfFmaSource& val,
                           const NativeHalfFmaResult& fresult) {
    NativeHalfFmaLowering({b, rs, in, ok, allow_wave}, old_d, val, fresult).lower();
}
} // namespace prosper::gpu::ir
