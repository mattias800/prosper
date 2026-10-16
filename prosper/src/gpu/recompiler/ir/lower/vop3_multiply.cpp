// Existing VOP3 word multiply leaves. The caller retains destination modifiers and predication.
#include "gpu/recompiler/ir/lower/vop3_multiply_lowering_internal.hpp"
#include "gpu/recompiler/ir/lower/vop3_multiply_lowering.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
bool Vop3MultiplyLowering::lower() {
    // Temporary sentinel retains the mechanically extracted else-if chain verbatim.
    if (false) {
    } else if (in.opcode == 0x169) {                          // v_mul_lo_u32
        vreg[in.dst.value] = b.ibin(Op_IMul, val(in.src[0]), val(in.src[1]));
    } else if (in.opcode == 0x16a) {                          // v_mul_hi_u32 (high 32 bits)
        vreg[in.dst.value] = b.umul_hi(val(in.src[0]), val(in.src[1]));
    } else if (in.opcode == 0x16c) {   // v_mul_hi_i32 (high 32 bits, signed)
        vreg[in.dst.value] = b.smul_hi(val(in.src[0]), val(in.src[1]));
    } else {
        return false;
    }
    return true;
}
bool lower_vop3_multiply(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                         bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3MultiplyLowering({b, rs, in, ok, allow_wave}).lower();
}
}   // namespace prosper::gpu::ir
