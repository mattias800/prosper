// AMD 70648 6.2, 12.8: ABS then NEG on the float source; OMOD ignores integer results.
#include "gpu/recompiler/ir/lower/vop3_float_integer_lowering.hpp"
#include "gpu/recompiler/ir/ops/vector_float_integer_e64.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
bool lower_vop3_float_integer(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok) {
    const auto* def = find_vector_float_integer_e64_op(in.fmt, in.opcode);
    const bool plain_flr = !def;
    if (!def) def = find_vector_cvt_flr_e64_op(in.fmt, in.opcode);
    if (!def) return false;
    // CONFIDENCE: MED -- unused raw sources and OPSEL must be zero. This does not widen
    // literal discovery. CLAMP enables the INEXACT exception on these conversions, rather
    // than saturating the integer result; its observable exception contract remains unsupported.
    if (in.has_modifier || in.has_sdwa || in.has_dpp || (in.words[1] & (0x3ffffu << 9)) ||
        ((in.words[0] >> 11) & 0xfu) || in.clamp || in.src_abs[1] || in.src_abs[2] ||
        in.src_neg[1] || in.src_neg[2] ||
        (plain_flr && (in.src_abs[0] || in.src_neg[0] || in.omod))) {
        ok = false;
        return true;
    }
    auto source = operand_bits(b, rs, in, in.src[0], &ok);
    if (!ok) return true;
    // The existing truncating aliases retain their exact sign-bit modifiers and NaN-zero
    // contract. The new plain FLR shape has no modifiers and reuses its legacy graph.
    if (in.src_abs[0]) source = b.ibin(Op_BitwiseAnd, source, b.uconst(0x7fffffffu));
    if (in.src_neg[0]) source = b.ibin(Op_BitwiseXor, source, b.uconst(0x80000000u));
    rs.vreg[in.dst.value] = emit_conversion_op(b, *def, source);
    return true;
}
}   // namespace prosper::gpu::ir
