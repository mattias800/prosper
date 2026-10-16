// Existing late VOP3 integer leaves. Admission and destination stages remain in emit_alu.
#include "gpu/recompiler/ir/lower/vop3_ternary_integer_lowering_internal.hpp"
#include "gpu/recompiler/ir/lower/vop3_ternary_integer_lowering.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
bool Vop3TernaryIntegerLowering::lower() {
    // The sentinel preserves the extracted else-if text verbatim in this relocation.
    if (false) {
    } else if (in.opcode == kVop3OpcodeAdd3U32) {   // v_add3_u32 = s0+s1+s2
        vreg[in.dst.value] =
            b.ibin(Op_IAdd, b.ibin(Op_IAdd, val(in.src[0]), val(in.src[1])), val(in.src[2]));
    } else if (in.opcode == 0x346) {   // v_lshl_add_u32 = (s0<<(s1&31))+s2
        uint32_t sh = b.ibin(Op_BitwiseAnd, val(in.src[1]), b.uconst(31));
        vreg[in.dst.value] =
            b.ibin(Op_IAdd, b.ibin(Op_ShiftLeftLogical, val(in.src[0]), sh), val(in.src[2]));
    } else if (in.opcode == kVop3OpcodeAndOrB32) {   // v_and_or_b32 = (s0&s1)|s2
        uint32_t t = b.ibin(Op_BitwiseAnd, val(in.src[0]), val(in.src[1]));
        vreg[in.dst.value] = b.ibin(Op_BitwiseOr, t, val(in.src[2]));
    } else if (in.opcode == 0x372) {   // v_or3_b32 = s0|s1|s2
        uint32_t t = b.ibin(Op_BitwiseOr, val(in.src[0]), val(in.src[1]));
        vreg[in.dst.value] = b.ibin(Op_BitwiseOr, t, val(in.src[2]));
    } else if (in.opcode == 0x178) {   // v_xor3_b32 = s0^s1^s2
        uint32_t t = b.ibin(Op_BitwiseXor, val(in.src[0]), val(in.src[1]));
        vreg[in.dst.value] = b.ibin(Op_BitwiseXor, t, val(in.src[2]));
    } else if (in.opcode == 0x36F) {   // v_lshl_or_b32 = (s0<<(s1&31))|s2
        uint32_t sh = b.ibin(Op_BitwiseAnd, val(in.src[1]), b.uconst(31));
        vreg[in.dst.value] =
            b.ibin(Op_BitwiseOr, b.ibin(Op_ShiftLeftLogical, val(in.src[0]), sh), val(in.src[2]));
    } else if (in.opcode == 0x345) {   // v_xad_u32 = (s0^s1)+s2
        uint32_t t = b.ibin(Op_BitwiseXor, val(in.src[0]), val(in.src[1]));
        vreg[in.dst.value] = b.ibin(Op_IAdd, t, val(in.src[2]));
    } else if (in.opcode == 0x347) {   // v_add_lshl_u32 = (s0+s1)<<(s2&31)
        uint32_t sh = b.ibin(Op_BitwiseAnd, val(in.src[2]), b.uconst(31));
        vreg[in.dst.value] =
            b.ibin(Op_ShiftLeftLogical, b.ibin(Op_IAdd, val(in.src[0]), val(in.src[1])), sh);
    } else {
        return false;
    }
    return true;
}
bool lower_vop3_ternary_integer(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3TernaryIntegerLowering({b, rs, in, ok, allow_wave}).lower();
}
}   // namespace prosper::gpu::ir
