#include "gpu/recompiler/rdna2_div_fixup.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {

uint32_t emit_v_div_fixup_f32(SpirvCompute& b, uint32_t s0, uint32_t s1, uint32_t s2) {
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
    const uint32_t mag_mask = b.uconst(0x7FFFFFFFu);
    const uint32_t exp_inf = b.uconst(0x7F800000u);
    const uint32_t sign_mask = b.uconst(0x80000000u);
    const uint32_t quiet_bit = b.uconst(0x00400000u);
    const uint32_t indet_nan = b.uconst(0xFFC00000u);

    const uint32_t mag0 = b.ibin(Op_BitwiseAnd, s0, mag_mask);
    const uint32_t mag1 = b.ibin(Op_BitwiseAnd, s1, mag_mask);
    const uint32_t mag2 = b.ibin(Op_BitwiseAnd, s2, mag_mask);

    const uint32_t is_nan1 = b.ucmp(Op_UGreaterThan, mag1, exp_inf);
    const uint32_t is_nan2 = b.ucmp(Op_UGreaterThan, mag2, exp_inf);

    const uint32_t is_inf1 = b.ucmp(Op_IEqual, mag1, exp_inf);
    const uint32_t is_inf2 = b.ucmp(Op_IEqual, mag2, exp_inf);

    const uint32_t is_zero1 = b.ucmp(Op_IEqual, mag1, b.uconst(0));
    const uint32_t is_zero2 = b.ucmp(Op_IEqual, mag2, b.uconst(0));

    const uint32_t sign_out = b.ibin(Op_BitwiseAnd, b.ibin(Op_BitwiseXor, s1, s2), sign_mask);
    const uint32_t signed_inf = b.ibin(Op_BitwiseOr, sign_out, exp_inf);
    const uint32_t signed_zero = sign_out;

    // Normal pass-through: sign_out ? -abs(S0) : abs(S0)
    const uint32_t normal = b.ibin(Op_BitwiseOr, mag0, sign_out);

    // Exponent underflow clause: (exponent(S2.f) - exponent(S1.f)) < -150
    // Biased exponents: (s >> 23) & 0xff
    const uint32_t exp1 =
        b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, s1, b.uconst(23)), b.uconst(0xff));
    const uint32_t exp2 =
        b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, s2, b.uconst(23)), b.uconst(0xff));
    const uint32_t exp_diff = b.ibin(Op_ISub, exp2, exp1);
    const uint32_t is_underflow =
        b.scmp(Op_SLessThan, exp_diff, b.uconst(static_cast<uint32_t>(-150)));
    const uint32_t res0 = b.sel(is_underflow, signed_zero, normal);

    const uint32_t res1 = b.sel(b.lor(is_inf1, is_zero2), signed_zero, res0);
    const uint32_t res2 = b.sel(b.lor(is_zero1, is_inf2), signed_inf, res1);

    const uint32_t both_zero = b.land(is_zero1, is_zero2);
    const uint32_t both_inf = b.land(is_inf1, is_inf2);
    const uint32_t res3 = b.sel(b.lor(both_zero, both_inf), indet_nan, res2);

    const uint32_t res4 = b.sel(is_nan1, b.ibin(Op_BitwiseOr, s1, quiet_bit), res3);
    return b.sel(is_nan2, b.ibin(Op_BitwiseOr, s2, quiet_bit), res4);
}

}  // namespace prosper::gpu
