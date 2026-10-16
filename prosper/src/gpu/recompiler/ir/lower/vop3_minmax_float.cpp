// Existing float min/max/median bodies moved verbatim by extract_case_routes.py.
#include "gpu/recompiler/ir/lower/vop3_minmax_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_minmax_float_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
bool Vop3MinmaxFloatLowering::lower() {
    if (in.opcode != 0x151 && in.opcode != 0x154 && in.opcode != 0x157) return false;
    if (!fv || !fresult) {
        std::fprintf(stderr, "[ir] float callbacks missing\n");
        std::abort();
    }
    if (in.opcode == 0x157) {
    // ISA op 343: "if (isNan(S0) || isNan(S1) || isNan(S2)) D = V_MIN3_F32(S0,S1,S2)".
    // The min/max legs use the NaN-aware NMin/NMax (return-the-other-operand), and the
    // any-NaN case selects min3 explicitly — the plain max(min(...)) formula does not
    // reproduce the documented NaN fallback on its own.
        uint32_t s0 = fv(0), s1 = fv(1), s2 = fv(2);
        uint32_t mn = b.fext2(Glsl_NMin, s0, s1), mx = b.fext2(Glsl_NMax, s0, s1);
        uint32_t med = b.fext2(Glsl_NMax, mn, b.fext2(Glsl_NMin, mx, s2));
        uint32_t min3 = b.fext2(Glsl_NMin, mn, s2);
        uint32_t nan_any =
            b.lor(b.fcmp(Op_FUnordNotEqual, s0, s0),
                  b.lor(b.fcmp(Op_FUnordNotEqual, s1, s1), b.fcmp(Op_FUnordNotEqual, s2, s2)));
        vreg[in.dst.value] = fresult(b.sel(nan_any, min3, med));
    } else {
    // min/max of three floats (DOLL's AA-clamp PS). VERIFIED(round-trip llvm-mc gfx1010:
    // VOP3 0x151 = v_min3_f32, 0x154 = v_max3_f32 — 0xd551…/0xd554…). NaN-aware NMin/
    // NMax per the ISA one-NaN rule. CONFIDENCE: HIGH.
        uint32_t op = in.opcode == 0x151 ? (uint32_t)Glsl_NMin : (uint32_t)Glsl_NMax;
        vreg[in.dst.value] = fresult(b.fext2(op, b.fext2(op, fv(0), fv(1)), fv(2)));
    }
    return true;
}
bool lower_vop3_minmax_float(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                             bool allow_wave, const Vop3MinmaxFloatSource& fv,
                             const Vop3MinmaxFloatResult& fresult) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3MinmaxFloatLowering({b, rs, in, ok, allow_wave}, fv, fresult).lower();
}
}   // namespace prosper::gpu::ir
