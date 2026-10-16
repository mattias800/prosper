// New e64 word-data forms borrow the existing numeric operand service, never mask authority.
#include "gpu/recompiler/ir/lower/vop3_binary_integer_lowering.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/ir/ops/vector_binary_integer.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
bool lower_vop3_binary_integer(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                               bool /*allow_wave*/) {
    const auto* def = find_vector_binary_integer_op(in.fmt, in.opcode);
    if (!def) return false;
    // CONFIDENCE: MED -- admit only canonical zero unused SRC2 fields. The generic decoder's
    // literal-width rule still counts every raw field, so noncanonical fields are not supported.
    if (in.has_modifier || in.has_dpp || in.has_sdwa || (in.words[1] & (0x1ffu << 18)) ||
        ((in.words[0] >> 11) & 0xfu) || in.clamp || in.omod || in.src_abs[0] || in.src_abs[1] ||
        in.src_abs[2] || in.src_neg[0] || in.src_neg[1] || in.src_neg[2]) {
        ok = false;
        return true;
    }
    // Preserve the architectural source stage: both full words are read before arithmetic.
    const auto s0 = operand_bits(b, rs, in, in.src[0], &ok);
    const auto s1 = operand_bits(b, rs, in, in.src[1], &ok);
    if (!ok) return true;
    rs.vreg[in.dst.value] = emit_vector_op(b, *def, s0, s1);
    return true;
}
}   // namespace prosper::gpu::ir
