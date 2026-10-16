// Original VOP3 bit/byte alignment value cases, extracted with their modifier guard.
#include "gpu/recompiler/ir/lower/vop3_align_lowering_internal.hpp"
#include "gpu/recompiler/ir/lower/vop3_align_lowering.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
bool Vop3AlignLowering::lower() {
    // Temporary sentinel preserves the extracted else-if chain verbatim.
    if (false) {
    } else if (in.opcode == kVop3OpcodeAlignbitB32 || in.opcode == kVop3OpcodeAlignbyteB32) {
        // RDNA2 defines ALIGNBIT as
        //   D = ({S0,S1} >> S2.u[4:0]) & 0xffffffff
        // and ALIGNBYTE as the same 64-bit join shifted by 8*S2.u[4:0]. S0 is the high
        // dword and S1 the low dword. Keep every emitted SPIR-V shift below 32:
        // OpSelect is not short-circuiting, so even an apparently unused shift by 32 would
        // be undefined. ALIGNBIT's masked bit count always remains in the joined low-dword
        // domain; ALIGNBYTE additionally has the high-dword and zero-fill domains.
        const bool modified = in.src_abs[0] || in.src_abs[1] || in.src_abs[2] || in.src_neg[0] ||
                              in.src_neg[1] || in.src_neg[2] || in.clamp || in.omod;
        if (modified) {
            ok = false;
        } else {
            const uint32_t s0 = val(in.src[0]);
            const uint32_t s1 = val(in.src[1]);
            const uint32_t count = b.ibin(Op_BitwiseAnd, val(in.src[2]), b.uconst(31));
            const uint32_t shift =
                in.opcode == kVop3OpcodeAlignbitB32
                    ? count
                    : b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftLeftLogical, count, b.uconst(3)),
                             b.uconst(31));
            const uint32_t inv_shift =
                b.ibin(Op_BitwiseAnd, b.ibin(Op_ISub, b.uconst(32), shift), b.uconst(31));
            const uint32_t joined = b.ibin(Op_BitwiseOr, b.ibin(Op_ShiftRightLogical, s1, shift),
                                           b.ibin(Op_ShiftLeftLogical, s0, inv_shift));
            const uint32_t low = b.sel(b.ucmp(Op_IEqual, count, b.uconst(0)), s1, joined);
            if (in.opcode == kVop3OpcodeAlignbitB32) {
                vreg[in.dst.value] = low;
            } else {
                const uint32_t upper_byte = b.ibin(Op_ISub, count, b.uconst(4));
                const uint32_t upper_shift =
                    b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftLeftLogical, upper_byte, b.uconst(3)),
                           b.uconst(31));
                const uint32_t upper = b.ibin(Op_ShiftRightLogical, s0, upper_shift);
                vreg[in.dst.value] =
                    b.sel(b.ucmp(Op_ULessThan, count, b.uconst(4)), low,
                          b.sel(b.ucmp(Op_ULessThan, count, b.uconst(8)), upper, b.uconst(0)));
            }
        }
    } else {
        return false;
    }
    return true;
}
bool lower_vop3_align(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                      bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3AlignLowering({b, rs, in, ok, allow_wave}).lower();
}
}   // namespace prosper::gpu::ir
