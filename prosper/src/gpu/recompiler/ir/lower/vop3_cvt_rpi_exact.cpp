// AMD70648 12.8: for x=k/2, |x|<2^23, Floor/frac/add and the final I32 cast are exact.
// Nonzero intermediates are normal; all round/denorm modes agree. FP zero sign cannot affect I32.
#include "gpu/recompiler/ir/lower/vop3_cvt_rpi_exact_lowering.hpp"
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/ir/ops/vector_cvt_rpi_exact_e64.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <bit>
#include <cstdint>
#include <optional>
namespace prosper::gpu::ir {
namespace {
std::optional<uint32_t> explicit_float_bits(const Rdna2Inst& in) {
    const uint32_t selector = in.words[1] & 0x1ffu;
    if (in.src[0] != decode_src_field(selector)) return std::nullopt;
    if (in.src[0].kind == OperandKind::Literal && in.has_literal && in.len_dwords == 3)
        return in.literal;
    if (in.src[0].kind == OperandKind::InlineFloat && selector >= 240 && selector <= 247 &&
        !in.has_literal && in.len_dwords == 2)
        return std::bit_cast<uint32_t>(inline_float_value(in.src[0].value));
    return std::nullopt;
}
bool exact_half_step(uint32_t bits) {
    const uint32_t magnitude = bits & 0x7fffffffu;
    if (!magnitude) return true;
    if (magnitude >= 0x4b000000u) return false; // Includes nonfinite values and |x|>=2^23.
    const uint32_t exponent = magnitude >> 23;
    if (exponent < 126) return false; // A nonzero half-step has magnitude at least 0.5.
    const uint32_t mantissa = 0x800000u | (magnitude & 0x7fffffu);
    return (mantissa & ((1u << (149u - exponent)) - 1u)) == 0;
}
} // namespace
bool lower_vop3_cvt_rpi_exact(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok) {
    const auto* def = find_vector_cvt_rpi_exact_e64_op(in.fmt, in.opcode);
    if (!def) return false;
    const auto bits = explicit_float_bits(in);
    if (!bits || !exact_half_step(bits.value_or(0))) return false;
    if (!ok) return true;
    if (in.has_modifier || in.has_sdwa || in.has_dpp || (in.words[0] & 0xff00u) ||
        (in.words[1] & ~0x1ffu) || in.clamp || in.omod || in.src_abs[0] || in.src_abs[1] ||
        in.src_abs[2] || in.src_neg[0] || in.src_neg[1] || in.src_neg[2] ||
        in.dst.kind != OperandKind::VGPR || in.n_src != 1) {
        ok = false;
        return true;
    }
    rs.vreg[in.dst.value] = emit_conversion_op(b, *def, b.uconst(bits.value_or(0)));
    return true;
}
} // namespace prosper::gpu::ir
