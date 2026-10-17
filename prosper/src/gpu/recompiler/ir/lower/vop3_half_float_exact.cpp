// AMD70648 6.2/12.8: explicit normal F16 values widen exactly to normal F32 values.
// No rounding, denormal, overflow, zero-sign or nonfinite policy is inferred for other inputs.
#include "gpu/recompiler/ir/lower/vop3_half_float_exact_lowering.hpp"
#include "gpu/recompiler/ir/ops/vector_half_float_exact_e64.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <optional>
namespace prosper::gpu::ir {
namespace {
std::optional<uint32_t> explicit_half_bits(const Rdna2Inst& in) {
    const uint32_t selector = in.words[1] & 0x1ffu;
    if (in.src[0] != decode_src_field(selector)) return std::nullopt;
    if (in.src[0].kind == OperandKind::Literal && in.has_literal && in.len_dwords == 3)
        return in.literal;
    // Match Vop1ConversionLowering's half-width remap, not operand_bits' F32 inline pattern.
    if (in.src[0].kind == OperandKind::InlineFloat && selector >= 240 && selector <= 248 &&
        !in.has_literal && in.len_dwords == 2)
        return inline_float_f16_bits(in.src[0].value);
    return std::nullopt;
}
} // namespace
bool lower_vop3_half_float_exact(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok) {
    const auto* def = find_vector_half_float_exact_e64_op(in.fmt, in.opcode);
    if (!def) return false;
    const auto bits = explicit_half_bits(in);
    if (!bits) return false;
    const uint32_t raw = bits.value_or(0), magnitude = raw & 0x7fffu;
    // Canonical literal high half is zero. The selected half must be finite, normal and nonzero.
    if ((raw & 0xffff0000u) || magnitude < 0x0400u || magnitude > 0x7bffu) return false;
    if (!ok) return true;
    if (in.has_modifier || in.has_sdwa || in.has_dpp || (in.words[0] & 0xff00u) ||
        (in.words[1] & ~0x1ffu) || in.clamp || in.omod || in.src_abs[0] || in.src_abs[1] ||
        in.src_abs[2] || in.src_neg[0] || in.src_neg[1] || in.src_neg[2] || in.vop3p_opsel ||
        in.vop3p_opsel_hi || in.vop3p_neg_hi || in.dst.kind != OperandKind::VGPR || in.n_src != 1) {
        ok = false;
        return true;
    }
    rs.vreg[in.dst.value] = emit_conversion_op(b, *def, b.uconst(raw));
    return true;
}
}   // namespace prosper::gpu::ir
