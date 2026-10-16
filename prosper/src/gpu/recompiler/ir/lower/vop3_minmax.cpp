// Existing VOP3 integer triple selection leaves. The caller retains destination modifiers and predication.
#include "gpu/recompiler/ir/lower/vop3_minmax_lowering_internal.hpp"
#include "gpu/recompiler/ir/lower/vop3_minmax_lowering.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/ir/ops/vector_minmax.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
bool Vop3MinmaxLowering::lower() {
    const auto* def = find_vector_minmax_op(in.fmt, in.opcode);
    if (!def) return false;
    vreg[in.dst.value] =
        emit_ternary_integer_op(b, *def, [this](uint32_t source) { return val(in.src[source]); });
    return true;
}
bool lower_vop3_minmax(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                       bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3MinmaxLowering({b, rs, in, ok, allow_wave}).lower();
}
}   // namespace prosper::gpu::ir
