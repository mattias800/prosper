// AMD70648 6.4/12.8: all round modes agree for exactly representable integer inputs.
// Zero stays outside: Vulkan's implicit NSZ on OpConvertSToF/UToF does not prove its raw sign.
#include "gpu/recompiler/ir/lower/vop3_exact_integer_float_lowering.hpp"
#include "gpu/recompiler/ir/ops/vector_exact_integer_float_e64.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <bit>
#include <cstdint>
#include <optional>

namespace prosper::gpu::ir {
namespace {
std::optional<uint32_t> explicit_integer_word(const Rdna2Inst& in) {
    const uint32_t selector = in.words[1] & 0x1ffu;
    if (in.src[0] != decode_src_field(selector)) return std::nullopt;
    if (in.src[0].kind == OperandKind::InlineInt && in.len_dwords == 2 && !in.has_literal)
        return static_cast<uint32_t>(in.src[0].value);
    if (in.src[0].kind == OperandKind::Literal && in.len_dwords == 3 && in.has_literal)
        return in.literal;
    return std::nullopt;
}
bool nonzero_exact(uint32_t bits, bool is_signed) {
    // Unsigned subtraction includes INT_MIN without signed abs/negation overflow.
    const uint32_t magnitude = is_signed && (bits & 0x80000000u) ? 0u - bits : bits;
    if (!magnitude) return false;
    const uint32_t width = std::bit_width(magnitude);
    return width <= 24 || (magnitude & ((1u << (width - 24)) - 1u)) == 0;
}
} // namespace
bool lower_vop3_exact_integer_float(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok) {
    const auto* def = find_vector_exact_integer_float_e64_op(in.fmt, in.opcode);
    if (!def) return false;
    const auto bits = explicit_integer_word(in);
    if (!bits) return false;
    const uint32_t raw = bits.value_or(0);
    if (!nonzero_exact(raw, in.opcode == 0x185)) return false;
    if (!ok) return true;
    // The decoder clears unused decoded operands but still discovers literals in all raw slots.
    // Keep all those slots and source/output modifiers outside this first plain admission.
    if (in.has_modifier || in.has_sdwa || in.has_dpp || (in.words[0] & 0xff00u) ||
        (in.words[1] & ~0x1ffu) || in.clamp || in.omod || in.src_abs[0] || in.src_abs[1] ||
        in.src_abs[2] || in.src_neg[0] || in.src_neg[1] || in.src_neg[2] ||
        in.dst.kind != OperandKind::VGPR || in.n_src != 1) {
        ok = false;
        return true;
    }
    rs.vreg[in.dst.value] = emit_conversion_op(b, *def, b.uconst(raw));
    return true;
}
}   // namespace prosper::gpu::ir
