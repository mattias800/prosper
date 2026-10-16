// Existing VOP3 integer triple selection leaves. The caller retains destination modifiers and predication.
#include "gpu/recompiler/ir/lower/vop3_minmax_lowering_internal.hpp"
#include "gpu/recompiler/ir/lower/vop3_minmax_lowering.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
bool Vop3MinmaxLowering::lower() {
    // Temporary sentinel retains the mechanically extracted else-if chain verbatim.
    if (false) {
    } else if (in.opcode == 0x158 || in.opcode == 0x159) {   // v_med3_i32 / v_med3_u32
        // Median of three values: max(min(a,b), min(max(a,b),c)).
        // Astro Bot uses v_med3_u32 to clamp a material index into [0,31] before its world-map
        // depth prepass. VERIFIED(llvm-mc gfx1030: VOP3 0x158 = v_med3_i32, 0x159 = v_med3_u32).
        const uint32_t s0 = val(in.src[0]), s1 = val(in.src[1]), s2 = val(in.src[2]);
        const bool is_signed = (in.opcode == 0x158);
        const uint32_t min_op = is_signed ? (uint32_t)Glsl_SMin : (uint32_t)Glsl_UMin;
        const uint32_t max_op = is_signed ? (uint32_t)Glsl_SMax : (uint32_t)Glsl_UMax;
        auto ext2 = [&](uint32_t op, uint32_t a, uint32_t b_val) {
            return is_signed ? b.sext2(op, a, b_val) : b.uext2(op, a, b_val);
        };
        const uint32_t mn = ext2(min_op, s0, s1);
        const uint32_t mx = ext2(max_op, s0, s1);
        vreg[in.dst.value] = ext2(max_op, mn, ext2(min_op, mx, s2));
    } else if (in.opcode == 0x152 || in.opcode == 0x153 || in.opcode == 0x155 ||
               in.opcode == 0x156) {   // v_min3_{i32,u32} / v_max3_{i32,u32}
        // Signed/unsigned min/max of three values. Sonic Racing: CrossWorlds' post chain rejects on
        // v_max3_u32 four times per boot (#2013); signed forms round out the 32-bit triple-min/max family.
        // VERIFIED(round-trip llvm-mc gfx1030: VOP3 0x152 = v_min3_i32, 0x153 = v_min3_u32,
        // 0x155 = v_max3_i32, 0x156 = v_max3_u32). CONFIDENCE: HIGH.
        const bool is_min = (in.opcode == 0x152 || in.opcode == 0x153);
        const bool is_signed = (in.opcode == 0x152 || in.opcode == 0x155);
        const uint32_t op = is_min ? (is_signed ? (uint32_t)Glsl_SMin : (uint32_t)Glsl_UMin)
                                   : (is_signed ? (uint32_t)Glsl_SMax : (uint32_t)Glsl_UMax);
        vreg[in.dst.value] =
            is_signed ? b.sext2(op, b.sext2(op, val(in.src[0]), val(in.src[1])), val(in.src[2]))
                      : b.uext2(op, b.uext2(op, val(in.src[0]), val(in.src[1])), val(in.src[2]));
    } else {
        return false;
    }
    return true;
}
bool lower_vop3_minmax(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                       bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3MinmaxLowering({b, rs, in, ok, allow_wave}).lower();
}
}   // namespace prosper::gpu::ir
