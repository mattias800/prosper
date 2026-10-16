// Existing VOP3 word bitfield leaves. The caller retains destination modifiers and predication.
#include "gpu/recompiler/ir/lower/vop3_bitfield_lowering_internal.hpp"
#include "gpu/recompiler/ir/lower/vop3_bitfield_lowering.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
bool Vop3BitfieldLowering::lower() {
    // Temporary sentinel retains the mechanically extracted else-if chain verbatim.
    if (false) {
    } else if (in.opcode == 0x148 || in.opcode == 0x149) {   // v_bfe_u32 / v_bfe_i32
        uint32_t off = b.ibin(Op_BitwiseAnd, val(in.src[1]), b.uconst(31));
        uint32_t cnt = b.ibin(Op_BitwiseAnd, val(in.src[2]), b.uconst(31));
        vreg[in.dst.value] = (in.opcode == 0x148) ? b.bfe_u(val(in.src[0]), off, cnt)
                                                  : b.bfe_s(val(in.src[0]), off, cnt);
    } else if (in.opcode == 0x14A) {   // v_bfi_b32 = (s0&s1)|(~s0&s2)
        uint32_t s0 = val(in.src[0]);
        uint32_t t1 = b.ibin(Op_BitwiseAnd, s0, val(in.src[1]));
        uint32_t t2 = b.ibin(Op_BitwiseAnd, b.iun(Op_Not, s0), val(in.src[2]));
        vreg[in.dst.value] = b.ibin(Op_BitwiseOr, t1, t2);
    } else {
        return false;
    }
    return true;
}
bool lower_vop3_bitfield(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                         bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3BitfieldLowering({b, rs, in, ok, allow_wave}).lower();
}
} // namespace prosper::gpu::ir
