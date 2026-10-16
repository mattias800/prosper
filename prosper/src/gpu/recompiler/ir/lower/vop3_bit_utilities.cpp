// Existing lane-wise numeric bit utilities, mechanically extracted without changing reads.
#include "gpu/recompiler/ir/lower/vop3_bit_utilities_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_bit_utilities_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
bool Vop3BitUtilitiesLowering::lower() {
    if (false) {
    } else if (in.opcode == 0x363) {   // v_bfm_b32
        // RDNA2: D.u32 = ((1 << S0[4:0]) - 1) << S1[4:0]. Mask both shift
        // amounts before emitting SPIR-V so neither can reach the undefined >=32 range;
        // in particular, an input width of 32 wraps to zero rather than producing all ones.
        // The instruction has no defined modifier forms. OP_SEL is not retained by the
        // generic decoder for this opcode, so reject its raw field explicitly as well.
        const uint32_t opsel = (in.words[0] >> 11) & 0xFu;
        if (in.src_abs[0] || in.src_abs[1] || in.src_abs[2] || in.src_neg[0] || in.src_neg[1] ||
            in.src_neg[2] || in.clamp || in.omod || opsel) {
            ok = false;
        } else {
            const uint32_t width = b.ibin(Op_BitwiseAnd, val(in.src[0]), b.uconst(31));
            const uint32_t offset = b.ibin(Op_BitwiseAnd, val(in.src[1]), b.uconst(31));
            const uint32_t mask =
                b.ibin(Op_ISub, b.ibin(Op_ShiftLeftLogical, b.uconst(1), width), b.uconst(1));
            vreg[in.dst.value] = b.ibin(Op_ShiftLeftLogical, mask, offset);
        }
    } else if (in.opcode == 0x364) {   // v_bcnt_u32_b32
        // AMD RDNA2: D = popcount(S0) + S1. The third VOP3 source field is unused.
        vreg[in.dst.value] = b.ibin(Op_IAdd, b.iun(Op_BitCount, val(in.src[0])), val(in.src[1]));
    } else {
        return false;
    }
    return true;
}
bool lower_vop3_bit_utilities(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                              bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3BitUtilitiesLowering({b, rs, in, ok, allow_wave}).lower();
}
}   // namespace prosper::gpu::ir
