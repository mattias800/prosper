// Existing VOP3 word multiply leaves. The caller retains destination modifiers and predication.
#include "gpu/recompiler/ir/lower/vop3_multiply_lowering_internal.hpp"
#include "gpu/recompiler/ir/lower/vop3_multiply_lowering.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/ir/ops/vector_multiply.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
bool Vop3MultiplyLowering::lower() {
    const auto* def = find_vector_multiply_op(in.fmt, in.opcode);
    if (!def) return false;
    vreg[in.dst.value] = emit_vector_op(b, *def, val(in.src[0]), val(in.src[1]));
    return true;
}
bool lower_vop3_multiply(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                         bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3MultiplyLowering({b, rs, in, ok, allow_wave}).lower();
}
}   // namespace prosper::gpu::ir
