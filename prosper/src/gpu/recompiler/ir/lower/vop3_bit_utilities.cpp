// The original BFM modifier boundary and source reader stay in lowering; IR owns data values.
#include "gpu/recompiler/ir/lower/vop3_bit_utilities_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_bit_utilities_lowering_internal.hpp"
#include "gpu/recompiler/ir/ops/vector_bit_utilities.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
bool Vop3BitUtilitiesLowering::lower() {
    const auto* def = find_vector_bit_utilities_op(in.fmt, in.opcode);
    if (!def) return false;
    if (in.opcode == 0x363) {
        // The generic decoder does not retain BFM's raw OP_SEL field.
        const uint32_t opsel = (in.words[0] >> 11) & 0xfu;
        if (in.src_abs[0] || in.src_abs[1] || in.src_abs[2] || in.src_neg[0] || in.src_neg[1] ||
            in.src_neg[2] || in.clamp || in.omod || opsel) {
            ok = false;
            return true;
        }
    }
    vreg[in.dst.value] =
        emit_ternary_integer_op(b, *def, [&](uint32_t slot) { return val(in.src[slot]); });
    return true;
}
bool lower_vop3_bit_utilities(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                              bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3BitUtilitiesLowering({b, rs, in, ok, allow_wave}).lower();
}
}   // namespace prosper::gpu::ir
