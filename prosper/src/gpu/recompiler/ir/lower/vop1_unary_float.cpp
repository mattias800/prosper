// Existing VOP1 F32 unary values, relocated verbatim. Operand and destination stages stay
// in emit_alu, including DPP transforms, modifiers, and the EXEC-predicated write.
#include "gpu/recompiler/ir/lower/vop1_unary_float_lowering.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
bool lower_vop1_unary_float(SpirvCompute& b, const Rdna2Inst& in, uint32_t a, uint32_t& d) {
    if (in.fmt != Rdna2Format::VOP1) return false;
    switch (in.opcode) {
        case 0x20: d = b.fext1(Glsl_Fract, a); break;         // v_fract_f32
        case 0x21: d = b.fext1(Glsl_Trunc, a); break;         // v_trunc_f32
        case 0x22: d = b.fext1(Glsl_Ceil, a); break;          // v_ceil_f32
        case 0x23: d = b.fext1(Glsl_RoundEven, a); break;   // v_rndne_f32 (round to nearest even)
        case 0x24: d = b.fext1(Glsl_Floor, a); break;   // v_floor_f32
        case 0x25: d = b.fext1(Glsl_Exp2, a); break;   // v_exp_f32 (2^x)
        case 0x27: d = b.fext1(Glsl_Log2, a); break;   // v_log_f32 (log2)
        case 0x2A: d = b.frcp(a); break;   // v_rcp_f32
        case 0x2B: d = b.frcp(a); break;   // v_rcp_iflag_f32 (~= v_rcp_f32)
        case 0x2E: d = b.fext1(Glsl_InverseSqrt, a); break;   // v_rsq_f32
        case 0x33:
            d = b.fext1(Glsl_Sqrt, a);
            break;   // v_sqrt_f32
        // v_sin_f32 (0x35) / v_cos_f32 (0x36): the RDNA trig input is in REVOLUTIONS (units of
        // 2π radians) — the compiler pre-multiplies by 1/2π (0.15915494) before these, so
        // d = sin/cos(2π·src). VERIFIED(round-trip llvm-mc gfx1010: 0x7e026b02/0x7e026d02 →
        // v_sin/cos_f32; DOLL's dither PS does exactly `v_mul 0.15915494, x` → `v_cos`).
        // CONFIDENCE: HIGH (RDNA2 ISA V_SIN_F32: D = sin(S0 * 2π)).
        case 0x35:
            d = b.fext1(Glsl_Sin, b.fbin(Op_FMul, a, b.uconst(fbits(6.28318530717958647692f))));
            break;
        case 0x36:
            d = b.fext1(Glsl_Cos, b.fbin(Op_FMul, a, b.uconst(fbits(6.28318530717958647692f))));
            break;
        default: return false;
    }
    return true;
}
}   // namespace prosper::gpu::ir
