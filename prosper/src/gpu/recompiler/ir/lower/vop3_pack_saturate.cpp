// Shared IR owns saturation and packing; the existing caller owns modifiers and EXEC publication.
#include "gpu/recompiler/ir/lower/vop3_pack_saturate_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_pack_saturate_lowering_internal.hpp"
#include "gpu/recompiler/ir/ops/vector_pack_saturate.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
bool Vop3PackSaturateLowering::lower() {
    const auto* def = find_vector_pack_saturate_op(in.fmt, in.opcode);
    if (!def) return false;
    vreg[in.dst.value] =
        emit_ternary_integer_op(b, *def, [&](uint32_t slot) { return val(in.src[slot]); });
    return true;
}
bool lower_vop3_pack_saturate(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                              bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3PackSaturateLowering({b, rs, in, ok, allow_wave}).lower();
}
}   // namespace prosper::gpu::ir
