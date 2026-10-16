// Only the four binary value cases move; root keeps half reads, CLAMP, packing and predication.
#include "gpu/recompiler/ir/lower/vop3_binary_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3p_packed_float_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
bool Vop3pPackedFloatLowering::lower() {
    if (in.fmt != Rdna2Format::VOP3P || in.opcode < 0x0f || in.opcode > 0x12) return false;
    if (!half) {
        std::fprintf(stderr, "[ir] packed float half callback missing\n");
        std::abort();
    }
    // 0x0e v_pk_fma_f16 is the only three-source form; min/max return the OTHER operand
    // when exactly one input is NaN (ISA 12.7), which is GLSL NMin/NMax, not FMin/FMax.
    // VERIFIED(round-trip llvm-mc gfx1030) for 0x0e/0x11/0x12.
    switch (in.opcode) {
        case 0x0F: r = b.fbin(Op_FAdd, half(0, high), half(1, high)); break;
        case 0x10: r = b.fbin(Op_FMul, half(0, high), half(1, high)); break;
        case 0x11: r = b.fext2(Glsl_NMin, half(0, high), half(1, high)); break;
        default: r = b.fext2(Glsl_NMax, half(0, high), half(1, high)); break;
    }
    return true;
}
bool lower_vop3p_packed_float(SpirvCompute& b, const Rdna2Inst& in,
                              const PackedFloatHalfSource& half, bool high, uint32_t& r) {
    return Vop3pPackedFloatLowering(b, in, half, high, r).lower();
}
}   // namespace prosper::gpu::ir
