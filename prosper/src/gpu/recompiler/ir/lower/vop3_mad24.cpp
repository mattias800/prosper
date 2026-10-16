// Original unsigned 24-bit multiply-add value leaf, extracted without changing its reads.
#include "gpu/recompiler/ir/lower/vop3_mad24_lowering_internal.hpp"
#include "gpu/recompiler/ir/lower/vop3_mad24_lowering.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/ir/ops/vector_mad24.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
bool Vop3Mad24Lowering::lower() {
    const auto* def = find_vector_mad24_op(in.fmt, in.opcode);
    if (!def) return false;
    vreg[in.dst.value] =
        emit_ternary_integer_op(b, *def, [this](uint32_t source) { return val(in.src[source]); });
    return true;
}
bool lower_vop3_mad24(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                      bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3Mad24Lowering({b, rs, in, ok, allow_wave}).lower();
}
}   // namespace prosper::gpu::ir
