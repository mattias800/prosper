#include "gpu/recompiler/rdna2_mad_64.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {

Mad64Result emit_mad_64_32(SpirvCompute& b, uint32_t s0, uint32_t s1, uint32_t add_lo,
                           uint32_t add_hi, bool is_signed) {
    // AMD RDNA2 (document 70648) / RDNA3 and CDNA3 pseudocode:
    //   0x176: {carry, D.u64} = S0.u32 * S1.u32 + S2.u64 (unsigned 65-bit sum).
    //   0x177: {carry, D.i64} = 65'(S0.i32 * S1.i32) + 65'(S2.i64); carry is bit 64 of the
    //          sign-extended 65-bit sum (the sign of the 65-bit result), NOT signed overflow.
    //   CONFIDENCE: MED for 0x177 (the 65-bit definition is stated in the RDNA3/CDNA3 manuals;
    //   70648 only names {vcc_out, D.i64}).
    const uint32_t mul_lo = b.ibin(Op_IMul, s0, s1);
    const uint32_t mul_hi = is_signed ? b.smul_hi(s0, s1) : b.umul_hi(s0, s1);
    const uint32_t result_lo = b.ibin(Op_IAdd, mul_lo, add_lo);
    const uint32_t carry_lo = b.ucmp(Op_ULessThan, result_lo, mul_lo);
    const uint32_t high_sum = b.ibin(Op_IAdd, mul_hi, add_hi);
    const uint32_t carry_hi0 = b.ucmp(Op_ULessThan, high_sum, mul_hi);
    const uint32_t carry_word = b.sel(carry_lo, b.uconst(1), b.uconst(0));
    const uint32_t result_hi = b.ibin(Op_IAdd, high_sum, carry_word);
    const uint32_t carry_hi1 = b.ucmp(Op_ULessThan, result_hi, high_sum);
    uint32_t carry = b.lor(carry_hi0, carry_hi1);   // unsigned carry out of bit 63
    if (is_signed) {
        // Bit 64 of the sign-extended sum = sign(mul) ^ sign(add) ^ unsigned carry out of bit 63.
        const uint32_t signs_differ =
            b.ibin(Op_ShiftRightLogical, b.ibin(Op_BitwiseXor, mul_hi, add_hi), b.uconst(31));
        carry = b.ucmp(Op_INotEqual, signs_differ, b.sel(carry, b.uconst(1), b.uconst(0)));
    }
    return {result_lo, result_hi, carry};
}

}  // namespace prosper::gpu
