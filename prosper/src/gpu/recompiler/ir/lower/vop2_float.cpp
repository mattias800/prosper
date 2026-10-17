// vop2_float.cpp — VOP2 floating-point operation stage, after source modifiers.
// Case bodies relocated by extract_case_routes.py; operand and instruction order are unchanged.
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/fragment_arithmetic_observation.hpp"
#include <cstdint>
#include "gpu/recompiler/ir/lower/vector_lowering_internal.hpp"
#include "gpu/recompiler/ir/ops/vector_float.hpp"
#include "gpu/recompiler/ir/ops/vector_mad_float.hpp"
#include "gpu/recompiler/ir/ops/vector_fma_float.hpp"

namespace prosper::gpu::ir {

bool Vop2Lowering::compute_float(uint32_t& d) {
    if (const VectorOpDef* def = find_vector_float_op(in.fmt, in.opcode)) {
        if (in.opcode == 0x03)
            b.observe_fragment_arithmetic_site(in.pc, FragmentArithmeticFamily::Add);
        if (in.opcode == 0x08)
            b.observe_fragment_arithmetic_site(in.pc, FragmentArithmeticFamily::Mul);
        d = emit_vector_op(b, *def, a, c);
        return true;
    }
    switch (in.opcode) {
        // v_mac_f32 (0x1f) / v_fmac_f32 (0x2b): dst = src0*src1 + dst (accumulate into the dest).
        // Fused forms express fusion through GLSL Fma, with MED Vulkan portability (#5225).
        // CONFIDENCE: HIGH: the 64-lane rounding witness in #5220 differs from AMD 70648.
        // NOTE(opcode ID): op 0x1f is v_mac_f32 on the PS5's ISA. v_mac_f32 was REMOVED on desktop
        // RDNA2/gfx1030 (where 0x1f is invalid and llvm-mc reads canonical v_dot2c at 0x02), but the
        // PS5 GPU retains the RDNA1/gfx1010 encoding — VERIFIED by round-tripping the scene VS's
        // actual op-0x1f word 0x3e261221 through `llvm-mc -mcpu=gfx1010` → `v_mac_f32_e32`. The VOP3
        // (e64) form of this same op is 0x11f, handled in the VOP3 switch below.
        case 0x1F: d = emit_nonfused_mad_values(b, a, c, old_d); break;
        case 0x2B: d = emit_fused_mad_values(b, a, c, old_d); break;
        // The four mul-add-with-literal-K ops (K = in.literal). madmk/fmamk = src0*K + src1;
        // madak/fmaak = src0*src1 + K. Only fused forms use GLSL Fma.
        case 0x20: d = emit_nonfused_mad_values(b, a, b.uconst(in.literal), c); break;
        case 0x2C:
            d = emit_fused_mad_values(b, a, b.uconst(in.literal), c);
            break;   // v_madmk / v_fmamk
        case 0x21: d = emit_nonfused_mad_values(b, a, c, b.uconst(in.literal)); break;
        case 0x2D:
            d = emit_fused_mad_values(b, a, c, b.uconst(in.literal));
            break;   // v_madak / v_fmaak
        default: return false;
    }
    return true;
}

}   // namespace prosper::gpu::ir
