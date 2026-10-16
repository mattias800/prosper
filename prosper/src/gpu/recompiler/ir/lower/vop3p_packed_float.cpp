// Only the four binary value cases move; root keeps half reads, CLAMP, packing and predication.
#include "gpu/recompiler/ir/lower/vop3_binary_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3p_packed_float_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/ir/ops/vector_packed_float.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
bool Vop3pPackedFloatLowering::lower() {
    const auto* def = find_vector_packed_float_op(in.fmt, in.opcode);
    if (!def) return false;
    if (!half) {
        std::fprintf(stderr, "[ir] packed float half callback missing\n");
        std::abort();
    }
    // 0x0e v_pk_fma_f16 is the only three-source form; min/max return the OTHER operand
    // when exactly one input is NaN (ISA 12.7), which is GLSL NMin/NMax, not FMin/FMax.
    // VERIFIED(round-trip llvm-mc gfx1030) for 0x0e/0x11/0x12.
    r = emit_ternary_integer_op(
        b, *def, [&](uint32_t source) { return half(static_cast<int>(source), high); });
    return true;
}
bool lower_vop3p_packed_float(SpirvCompute& b, const Rdna2Inst& in,
                              const PackedFloatHalfSource& half, bool high, uint32_t& r) {
    return Vop3pPackedFloatLowering(b, in, half, high, r).lower();
}
}   // namespace prosper::gpu::ir
