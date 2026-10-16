// The numeric result is exact; canonical admission leaves all mask and predication state outside.
#include "gpu/recompiler/ir/lower/vop3_subword_to_float_lowering.hpp"
#include "gpu/recompiler/ir/ops/vector_subword_to_float_e64.hpp"
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
bool lower_vop3_subword_to_float(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok) {
    const auto* def = find_vector_subword_to_float_e64_op(in.fmt, in.opcode);
    if (!def) return false;
    // CONFIDENCE: MED -- unused raw sources must be zero; literal discovery is unchanged.
    // This first admission leaves floating-result modifiers and OPSEL unsupported.
    if (in.has_modifier || in.has_sdwa || in.has_dpp || (in.words[1] & (0x3ffffu << 9)) ||
        ((in.words[0] >> 11) & 0xfu) || in.clamp || in.omod || in.src_abs[0] || in.src_abs[1] ||
        in.src_abs[2] || in.src_neg[0] || in.src_neg[1] || in.src_neg[2]) {
        ok = false;
        return true;
    }
    const auto source = operand_bits(b, rs, in, in.src[0], &ok);
    if (ok) rs.vreg[in.dst.value] = emit_conversion_op(b, *def, source);
    return true;
}
} // namespace prosper::gpu::ir
