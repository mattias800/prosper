// vop2_float.cpp — VOP2 floating-point operation stage, after source modifiers.
// Case bodies relocated by extract_case_routes.py; operand and instruction order are unchanged.
#include "gpu/recompiler/ir/lower/vector_lowering_internal.hpp"

namespace prosper::gpu::ir {

bool Vop2Lowering::compute_float(uint32_t& d) {
    switch (in.opcode) {
        case 0x03:                                          // v_add_f32
            b.observe_fragment_arithmetic_site(in.pc, FragmentArithmeticFamily::Add);
            d = b.fbin(Op_FAdd, a, c);
            break;
        case 0x04: d = b.fbin(Op_FSub, a, c); break;          // v_sub_f32
        case 0x05:
            d = b.fbin(Op_FSub, c, a);
            break;          // v_subrev_f32 (src1 - src0; e32 form of
            // VOP3 0x105 — round-trip llvm-mc gfx1010 0x0a020702)
        case 0x08:   // v_mul_f32
            b.observe_fragment_arithmetic_site(in.pc, FragmentArithmeticFamily::Mul);
            d = b.fbin(Op_FMul, a, c);
            break;
        // v_min/v_max: hardware returns the OTHER operand when exactly one input is NaN
        // (ISA 12.7 ops 15/16: "if (S0 == NaN) D = S1; ..."), which is GLSL NMin/NMax.
        // Plain FMin/FMax are NaN-UNDEFINED — llvmpipe propagates a second-operand NaN, so
        // the ubiquitous max(min(x,K),lo) clamp idiom could yield NaN pixels where hardware
        // yields the bound.
        case 0x0F: d = b.fext2(Glsl_NMin, a, c); break;   // v_min_f32
        case 0x10:
            d = b.fext2(Glsl_NMax, a, c);
            break;   // v_max_f32
        // v_mac_f32 (0x1f) / v_fmac_f32 (0x2b): dst = src0*src1 + dst (accumulate into the dest).
        // mac vs fmac differ only in fused rounding — immaterial here. old_d = the dst accumulator.
        // NOTE(opcode ID): op 0x1f is v_mac_f32 on the PS5's ISA. v_mac_f32 was REMOVED on desktop
        // RDNA2/gfx1030 (where 0x1f is invalid and llvm-mc reads canonical v_dot2c at 0x02), but the
        // PS5 GPU retains the RDNA1/gfx1010 encoding — VERIFIED by round-tripping the scene VS's
        // actual op-0x1f word 0x3e261221 through `llvm-mc -mcpu=gfx1010` → `v_mac_f32_e32`. The VOP3
        // (e64) form of this same op is 0x11f, handled in the VOP3 switch below.
        case 0x1F:
        case 0x2B: d = b.fbin(Op_FAdd, b.fbin(Op_FMul, a, c), old_d); break;
        // The four mul-add-with-literal-K ops (K = in.literal). madmk/fmamk = src0*K + src1;
        // madak/fmaak = src0*src1 + K. (mad vs fma differ only in fused rounding — immaterial here.)
        case 0x20:
        case 0x2C:
            d = b.fbin(Op_FAdd, b.fbin(Op_FMul, a, b.uconst(in.literal)), c);
            break;   // v_madmk / v_fmamk
        case 0x21:
        case 0x2D:
            d = b.fbin(Op_FAdd, b.fbin(Op_FMul, a, c), b.uconst(in.literal));
            break;   // v_madak / v_fmaak
        default: return false;
    }
    return true;
}

}   // namespace prosper::gpu::ir
