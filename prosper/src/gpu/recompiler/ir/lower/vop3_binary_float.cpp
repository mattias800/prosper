// Six already admitted binary float cases mechanically moved; observation calls stay in order.
#include "gpu/recompiler/ir/lower/vop3_binary_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_binary_float_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
bool Vop3BinaryFloatLowering::lower() {
    if (in.opcode == 0x103) {   // v_add_f32 (VOP3 form)
        b.observe_fragment_arithmetic_site(in.pc, FragmentArithmeticFamily::Add);
        vreg[in.dst.value] = fresult(b.fbin(Op_FAdd, fv(0), fv(1)));
    } else if (in.opcode == 0x104) {   // v_sub_f32 (VOP3 form) = s0 - s1
        vreg[in.dst.value] = fresult(b.fbin(Op_FSub, fv(0), fv(1)));
    } else if (in.opcode == 0x105) {   // v_subrev_f32 (VOP3 form) = s1 - s0
        vreg[in.dst.value] = fresult(b.fbin(Op_FSub, fv(1), fv(0)));
    } else if (in.opcode == 0x108) {   // v_mul_f32 (VOP3 form)
        b.observe_fragment_arithmetic_site(in.pc, FragmentArithmeticFamily::Mul);
        vreg[in.dst.value] = fresult(b.fbin(Op_FMul, fv(0), fv(1)));
    } else if (in.opcode == 0x10F) {   // v_min_f32 (VOP3 form; NaN -> other operand)
        vreg[in.dst.value] = fresult(b.fext2(Glsl_NMin, fv(0), fv(1)));
    } else if (in.opcode == 0x110) {   // v_max_f32 (VOP3 form; NaN -> other operand)
        vreg[in.dst.value] = fresult(b.fext2(Glsl_NMax, fv(0), fv(1)));
    } else
        return false;
    return true;
}
bool lower_vop3_binary_float_erased(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                    bool allow_wave, const BinaryFloatSource& fv,
                                    const BinaryFloatResult& fresult) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3BinaryFloatLowering({b, rs, in, ok, allow_wave}, fv, fresult).lower();
}
}   // namespace prosper::gpu::ir
