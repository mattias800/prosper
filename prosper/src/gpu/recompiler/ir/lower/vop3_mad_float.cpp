// Existing non-fused VOP3 MAD body, mechanically moved; fused FMA stays in the legacy route.
#include "gpu/recompiler/ir/lower/vop3_mad_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_mad_float_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
bool Vop3MadFloatLowering::lower() {
    if (in.opcode != 0x141) return false;
    if (!fv || !fresult) {
        std::fprintf(stderr, "[ir] VOP3 float callbacks missing\n");
        std::abort();
    }
    // v_mad_f32 (op 0x141) is a gfx10.1 (Navi) instruction REMOVED in gfx10.3, so llvm-mc
    // -mcpu=gfx1030 rejects it as invalid — but the PS5 shader compiler targets gfx10.1 and
    // emits it (real game shaders 5,26-29: manual attribute interpolation p0+i*p1). Its result
    // (unfused mul-then-add) maps exactly to OpFMul+OpFAdd; v_fma's fused rounding is
    // immaterial here. VERIFIED(round-trip llvm-mc gfx1010, both directions): VOP3 op 0x141.
    uint32_t m = b.fbin(Op_FMul, fv(0), fv(1));
    vreg[in.dst.value] = fresult(b.fbin(Op_FAdd, m, fv(2)));
    return true;
}
bool lower_vop3_mad_float(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                          bool allow_wave, const Vop3FloatSource& fv,
                          const Vop3FloatResult& fresult) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3MadFloatLowering({b, rs, in, ok, allow_wave}, fv, fresult).lower();
}
}   // namespace prosper::gpu::ir
