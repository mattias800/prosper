// Existing DX9 legacy multiply body mechanically moved, preserving all old call positions.
#include "gpu/recompiler/ir/lower/vop3_legacy_mul_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_legacy_mul_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
bool Vop3LegacyMulLowering::lower() {
    if (in.opcode != 0x107) return false;
    if (!fv || !fresult) {
        std::fprintf(stderr, "[ir] legacy multiply callbacks missing\n");
        std::abort();
    }
    // 0 * x == 0 for ALL x including ±Inf/NaN (that guarantee is WHY compilers emit it,
    // e.g. attenuation=0 times 1/dist). Plain IEEE FMul gives NaN for 0*Inf, so emit
    // select(s0==0 || s1==0, 0, s0*s1). ±0.0 both compare equal to 0.0 under FOrdEqual.
    uint32_t s0b = fv(0), s1b = fv(1), zb = b.uconst(0);
    uint32_t anyz = b.lor(b.fcmp(Op_FOrdEqual, s0b, zb), b.fcmp(Op_FOrdEqual, s1b, zb));
    vreg[in.dst.value] = fresult(b.sel(anyz, zb, b.fbin(Op_FMul, s0b, s1b)));
    return true;
}
bool lower_vop3_legacy_mul_erased(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                  bool allow_wave, const LegacyMulSource& fv,
                                  const LegacyMulResult& fresult) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3LegacyMulLowering({b, rs, in, ok, allow_wave}, fv, fresult).lower();
}
}   // namespace prosper::gpu::ir
