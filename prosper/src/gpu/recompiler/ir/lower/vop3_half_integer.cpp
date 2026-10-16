// Canonical e64 half-to-integer aliases preserve the existing high destination half.
#include "gpu/recompiler/ir/lower/vop3_half_integer_lowering.hpp"
#include "gpu/recompiler/ir/ops/vector_half_integer_e64.hpp"
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
bool lower_vop3_half_integer(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                             bool /*allow_wave*/, uint32_t old_d) {
    const auto* def = find_vector_half_integer_e64_op(in.fmt, in.opcode);
    if (!def) return false;
    // CONFIDENCE: MED -- canonical zero unused fields only; literal-width discovery is unchanged.
    // AMD70648 6.2.5 excludes OPSEL for these VOP1 aliases. CLAMP enables INEXACT, which this
    // numeric stage does not model. This admission implements only the unmodified form.
    if (in.has_modifier || in.has_sdwa || in.has_dpp || (in.words[1] & (0x3ffffu << 9)) ||
        ((in.words[0] >> 11) & 0xfu) || in.clamp || in.omod || in.src_abs[0] || in.src_abs[1] ||
        in.src_abs[2] || in.src_neg[0] || in.src_neg[1] || in.src_neg[2]) {
        ok = false;
        return true;
    }
    uint32_t raw = operand_bits(b, rs, in, in.src[0], &ok);
    if (in.src[0].kind == OperandKind::InlineFloat)
        raw = b.uconst(inline_float_f16_bits(in.src[0].value));
    else if (in.src[0].kind == OperandKind::InlineInt)
        raw = b.uconst(static_cast<uint32_t>(in.src[0].value));
    if (ok) {
        const uint32_t r16 = emit_conversion_op(b, *def, raw);
        rs.vreg[in.dst.value] =
            b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xffff0000u)), r16);
    }
    return true;
}
}   // namespace prosper::gpu::ir
