// F32 unary value lowering only; source/output stages and EXEC remain in the caller.
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include <cstdint>
#include "gpu/recompiler/ir/lower/vop1_unary_float_lowering.hpp"
#include "gpu/recompiler/ir/ops/vector_float_unary.hpp"
#include "gpu/recompiler/ir/lower/f32_round_input.hpp"
namespace prosper::gpu::ir {
bool lower_vop1_unary_float(SpirvCompute& b, const Rdna2Inst& in, uint32_t a, uint32_t& d,
                            bool& ok) {
    const auto* def = find_vector_float_unary_op(in.fmt, in.opcode);
    if (!def) return false;
    const auto result = emit_f32_rounding_op(b, in, *def, a, ok);
    if (ok) d = result;
    return true;
}
} // namespace prosper::gpu::ir
