// Existing non-fused accumulator body mechanically relocated; oldD is not an explicit FV source.
#include "gpu/recompiler/ir/lower/vop3_mad_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_mac_float_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
bool Vop3MacFloatLowering::lower() {
    if (in.opcode != 0x11f) return false;
    if (!fv || !fresult) {
        std::fprintf(stderr, "[ir] MAC float callbacks missing\n");
        std::abort();
    }
    // dst = src0*src1 + dst, with the VOP3 float source modifiers (neg/abs via fv) and output
    // modifiers (omod/clamp via fresult). The scene VS emits this e64 form with a `-|v10|`
    // modifier (round-trip: `llvm-mc -mcpu=gfx1010` of 0xd51f020a → v_mac_f32_e64 v10,v28,-|v10|).
    vreg[in.dst.value] = fresult(b.fbin(Op_FAdd, b.fbin(Op_FMul, fv(0), fv(1)), old_d));
    return true;
}
bool lower_vop3_mac_float(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                          bool allow_wave, const Vop3FloatSource& fv,
                          const Vop3FloatResult& fresult, uint32_t old_d) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3MacFloatLowering({b, rs, in, ok, allow_wave}, fv, fresult, old_d).lower();
}
}   // namespace prosper::gpu::ir
