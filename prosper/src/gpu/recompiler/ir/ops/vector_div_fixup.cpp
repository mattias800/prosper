// AMD70648 op351: shared special-case numerics, with the original append and nesting order.
#include "gpu/recompiler/ir/ops/vector_div_fixup.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
Value build_fixup(Builder& ir, Value s0, Value s1, Value s2) {
    // AMD RDNA2 ISA (document 70648): V_DIV_FIXUP_F32 (opcode 0x15f)
    // S0 = quotient, S1 = denominator, S2 = numerator
    //
    // IEEE 754 division corner case resolution per RDNA2 ISA pseudocode:
    //   sign_out = sign(S1.f) ^ sign(S2.f)
    //   if (S2.f == NAN) D.f = Quiet(S2.f)
    //   else if (S1.f == NAN) D.f = Quiet(S1.f)
    //   else if (S1.f == 0 && S2.f == 0) D.f = 0xffc00000 (indeterminate NaN)
    //   else if (abs(S1.f) == INF && abs(S2.f) == INF) D.f = 0xffc00000
    //   else if (S1.f == 0 || abs(S2.f) == INF) D.f = sign_out ? -INF : +INF
    //   else if (abs(S1.f) == INF || S2.f == 0) D.f = sign_out ? -0.0f : +0.0f
    //   else if ((exponent(S2.f) - exponent(S1.f)) < -150) D.f = sign_out ? -underflow : underflow
    //   else D.f = sign_out ? -abs(S0.f) : abs(S0.f)
    //
    // Note on exponent(S1)==255: Document 70648 includes an "else if (exponent(S1.f) == 255) overflow"
    // clause, but biased exponent 255 represents Inf or NaN, which earlier clauses already resolve.
    // Note on underflow: Document 70648 specifies an underflow clause for difference < -150, signed zero
    // is selected for underflow. CONFIDENCE: MED (exact underflow magnitude unspecified in doc).
    // Note on S0 NaN: Follows documented pseudocode falling through to sign_out ? -abs(S0) : abs(S0).
    const Value mag_mask = ir.c32(0x7FFFFFFFu);
    const Value exp_inf = ir.c32(0x7F800000u);
    const Value sign_mask = ir.c32(0x80000000u);
    const Value quiet_bit = ir.c32(0x00400000u);
    const Value indet_nan = ir.c32(0xFFC00000u);

    const Value mag0 = ir.and_(s0, mag_mask);
    const Value mag1 = ir.and_(s1, mag_mask);
    const Value mag2 = ir.and_(s2, mag_mask);

    const Value is_nan1 = ir.ugt(mag1, exp_inf);
    const Value is_nan2 = ir.ugt(mag2, exp_inf);

    const Value is_inf1 = ir.ieq(mag1, exp_inf);
    const Value is_inf2 = ir.ieq(mag2, exp_inf);

    const Value is_zero1 = ir.ieq(mag1, ir.c32(0));
    const Value is_zero2 = ir.ieq(mag2, ir.c32(0));

    const Value sign_out = ir.and_(ir.xor_(s1, s2), sign_mask);
    const Value signed_inf = ir.or_(sign_out, exp_inf);
    const Value signed_zero = sign_out;

    // Normal pass-through: sign_out ? -abs(S0) : abs(S0)
    const Value normal = ir.or_(mag0, sign_out);

    // Exponent underflow clause: (exponent(S2.f) - exponent(S1.f)) < -150
    // Biased exponents: (s >> 23) & 0xff
    const Value exp1 = ir.and_(ir.lshr(s1, ir.c32(23)), ir.c32(0xff));
    const Value exp2 = ir.and_(ir.lshr(s2, ir.c32(23)), ir.c32(0xff));
    const Value exp_diff = ir.isub(exp2, exp1);
    const Value is_underflow = ir.slt(exp_diff, ir.c32(static_cast<uint32_t>(-150)));
    const Value res0 = ir.select(is_underflow, signed_zero, normal);

    const Value res1 = ir.select(ir.boolor_(is_inf1, is_zero2), signed_zero, res0);
    const Value res2 = ir.select(ir.boolor_(is_zero1, is_inf2), signed_inf, res1);

    const Value both_zero = ir.booland_(is_zero1, is_zero2);
    const Value both_inf = ir.booland_(is_inf1, is_inf2);
    const Value res3 = ir.select(ir.boolor_(both_zero, both_inf), indet_nan, res2);

    const Value res4 = ir.select(is_nan1, ir.or_(s1, quiet_bit), res3);
    return ir.select(is_nan2, ir.or_(s2, quiet_bit), res4);
}
constexpr TernaryIntegerDef kDefs[] = {
    {Rdna2Format::VOP3, 0x15f,
     [](Builder& ir, const TernaryInputs& x) { return build_fixup(ir, x.s0(), x.s1(), x.s2()); }}};
}   // namespace
std::span<const TernaryIntegerDef> vector_div_fixup_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_div_fixup_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
