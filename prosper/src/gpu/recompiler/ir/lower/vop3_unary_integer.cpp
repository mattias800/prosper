// Canonical integer e64 aliases do not claim the reserved MOV, masks or cross-lane routes.
#include "gpu/recompiler/ir/lower/vop3_unary_integer_lowering.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/ir/ops/vector_unary_integer_e64.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
bool lower_vop3_unary_integer(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                              bool /*allow_wave*/) {
    const auto* def = find_vector_unary_integer_e64_op(in.fmt, in.opcode);
    if (!def) return false;
    // CONFIDENCE: MED -- only zero unused raw SRC1/SRC2 fields are admitted. Literal-width
    // discovery still considers every encoded field; no wider instruction-length rule is changed.
    if (in.has_modifier || in.has_sdwa || in.has_dpp || (in.words[1] & (0x3ffffu << 9)) ||
        ((in.words[0] >> 11) & 0xfu) || in.clamp || in.omod || in.src_abs[0] || in.src_abs[1] ||
        in.src_abs[2] || in.src_neg[0] || in.src_neg[1] || in.src_neg[2]) {
        ok = false;
        return true;
    }
    const auto s0 = operand_bits(b, rs, in, in.src[0], &ok);
    if (ok) rs.vreg[in.dst.value] = emit_vector_op(b, *def, s0, 0);
    return true;
}
}   // namespace prosper::gpu::ir
