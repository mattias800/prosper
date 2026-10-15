// scalar_alu.cpp — definitions of the SOP1, SOP2 and SOPK scalar data operations.
//
// Semantics from AMD "RDNA 2" ISA (doc 70648) 12.1-12.3; opcode numbers from the instruction-effects
// table, which was cross-checked against llvm-mc. Each definition appends its instructions in
// the order the pre-IR emitter called SpirvCompute, and with the same expression nesting, so the
// emitted SPIR-V is byte-identical (HOW_TO_ADD_A_FAMILY.md, "byte-identical").
#include "gpu/recompiler/ir/ops/scalar_alu.hpp"

namespace prosper::gpu::ir {

namespace {

using F = Rdna2Format;
using O = ScalarOperands;
using In = ScalarOpIn;
using Out = ScalarOpOut;

// Logical and shift results set SCC = (D != 0).
Out nonzero(Builder& ir, Value d) { return {d, ir.ine(d, ir.c32(0))}; }

// SCC is the signed overflow: bit 31 of `overflow_bits`.
Value sign_bit_set(Builder& ir, Value overflow_bits) {
    return ir.ine(ir.lshr(overflow_bits, ir.c32(31)), ir.c32(0));
}

Out add_i32(Builder& ir, const In& x) {
    const Value d = ir.iadd(x.s0, x.s1);
    // Overflow: operands of equal sign and a result of the other sign.
    const Value o = ir.and_(ir.not_(ir.xor_(x.s0, x.s1)), ir.xor_(x.s0, d));
    return {d, sign_bit_set(ir, o)};
}

Out sub_i32(Builder& ir, const In& x) {
    const Value d = ir.isub(x.s0, x.s1);
    const Value o = ir.and_(ir.xor_(x.s0, x.s1), ir.xor_(x.s0, d));
    return {d, sign_bit_set(ir, o)};
}

// s_addc_u32: D = S0 + S1 + SCC, SCC = carry out of the 33-bit sum; the high half of a 64-bit add.
// llvm-mc gfx1010: 0x82252580 is s_addc_u32 s37, 0, s37.
Out addc_u32(Builder& ir, const In& x) {
    const Value carry_in = ir.select(x.scc, ir.c32(1), ir.c32(0));
    const Value sum = ir.iadd(x.s0, x.s1);
    const Value first_carry = ir.ult(sum, x.s0);
    const Value d = ir.iadd(sum, carry_in);
    const Value second_carry = ir.ult(d, sum);
    return {d, ir.bselect(first_carry, ir.ctrue(), second_carry)};
}

// s_subb_u32: D = S0 - S1 - SCC, SCC = borrow out.
Out subb_u32(Builder& ir, const In& x) {
    const Value borrow_in = ir.select(x.scc, ir.c32(1), ir.c32(0));
    const Value difference = ir.isub(x.s0, x.s1);
    const Value first_borrow = ir.ult(x.s0, x.s1);
    const Value d = ir.isub(difference, borrow_in);
    const Value second_borrow = ir.ult(difference, borrow_in);
    return {d, ir.bselect(first_borrow, ir.ctrue(), second_borrow)};
}

// s_bfm_b32: D = ((1 << S0[4:0]) - 1) << S1[4:0]. No SCC.
Out bfm_b32(Builder& ir, const In& x) {
    const Value width = ir.and_(x.s0, ir.c32(31));
    const Value offset = ir.and_(x.s1, ir.c32(31));
    const Value ones = ir.isub(ir.shl(ir.c32(1), width), ir.c32(1));
    return {ir.shl(ones, offset), {}};
}

// s_bfe_u32: offset = S1[4:0], width = S1[22:16]; SCC = (D != 0).
Out bfe_u32(Builder& ir, const In& x) {
    const Value offset = ir.and_(x.s1, ir.c32(0x1f));
    const Value width = ir.and_(ir.lshr(x.s1, ir.c32(16)), ir.c32(0x7f));
    return nonzero(ir, ir.bfe_u(x.s0, offset, width));
}

// s_lshl<N>_add_u32: D = (S0 << N) + S1; SCC = carry out of the full 64-bit sum: either S0's top N
// bits were set, or the low-dword addition wrapped.
template <uint32_t N>
Out lshl_add_u32(Builder& ir, const In& x) {
    const Value shifted = ir.shl(x.s0, ir.c32(N));
    const Value d = ir.iadd(shifted, x.s1);
    const Value shifted_out = ir.ine(ir.lshr(x.s0, ir.c32(32u - N)), ir.c32(0));
    const Value wrapped = ir.ult(d, x.s1);
    return {d, ir.bselect(shifted_out, ir.ctrue(), wrapped)};
}

// s_pack_{ll,lh,hh}_b32_b16: D = {S1 half, S0 half}. No SCC.
template <bool S0_HIGH, bool S1_HIGH>
Out pack_b32_b16(Builder& ir, const In& x) {
    const Value lo = S0_HIGH ? ir.lshr(x.s0, ir.c32(16)) : ir.and_(x.s0, ir.c32(0xffff));
    const Value hi = S1_HIGH ? ir.and_(x.s1, ir.c32(0xffff0000))
                             : ir.shl(ir.and_(x.s1, ir.c32(0xffff)), ir.c32(16));
    return {ir.or_(lo, hi), {}};
}

// s_abs_i32: INT_MIN stays INT_MIN; SCC = (D != 0).
Out abs_i32(Builder& ir, const In& x) {
    const Value negative = ir.slt(x.s0, ir.c32(0));
    return nonzero(ir, ir.select(negative, ir.isub(ir.c32(0), x.s0), x.s0));
}

constexpr ScalarOpDef kDefs[] = {
    // SOP2
    {F::SOP2, 0x00, O::Src0Src1, [](Builder& ir, const In& x) -> Out {   // s_add_u32
         const Value d = ir.iadd(x.s0, x.s1);
         return {d, ir.ult(d, x.s0)};
     }, Fold::Yes},
    {F::SOP2, 0x01, O::Src0Src1, [](Builder& ir, const In& x) -> Out {   // s_sub_u32
         return {ir.isub(x.s0, x.s1), ir.ult(x.s0, x.s1)};
     }, Fold::Yes},
    {F::SOP2, 0x02, O::Src0Src1, add_i32, Fold::Yes},
    {F::SOP2, 0x03, O::Src0Src1, sub_i32, Fold::Yes},
    {F::SOP2, 0x04, O::Src0Src1, addc_u32},
    {F::SOP2, 0x05, O::Src0Src1, subb_u32},
    // s_min/s_max: SCC = "S0 was selected", strict in both directions, so a tie gives SCC = 0
    // (ISA 12.1 ops 6-9, re-derived from the PDF in #879 after #397 made max non-strict). llvm-mc
    // gfx1010: 0x83000201 / 0x83800201 / 0x84000201 / 0x84800201. SCC is appended before D, the
    // old emitter's order.
    {F::SOP2, 0x06, O::Src0Src1, [](Builder& ir, const In& x) -> Out {
         const Value scc = ir.slt(x.s0, x.s1);
         return {ir.smin(x.s0, x.s1), scc};
     }},
    {F::SOP2, 0x07, O::Src0Src1, [](Builder& ir, const In& x) -> Out {
         const Value scc = ir.ult(x.s0, x.s1);
         return {ir.umin(x.s0, x.s1), scc};
     }},
    {F::SOP2, 0x08, O::Src0Src1, [](Builder& ir, const In& x) -> Out {
         const Value scc = ir.sgt(x.s0, x.s1);
         return {ir.smax(x.s0, x.s1), scc};
     }},
    {F::SOP2, 0x09, O::Src0Src1, [](Builder& ir, const In& x) -> Out {
         const Value scc = ir.ugt(x.s0, x.s1);
         return {ir.umax(x.s0, x.s1), scc};
     }},
    {F::SOP2, kSop2OpcodeCselectB32, O::Src0Src1, [](Builder& ir, const In& x) -> Out {
         return {ir.select(x.scc, x.s0, x.s1), {}};
     }},
    {F::SOP2, kSop2OpcodeAndB32, O::Src0Src1, [](Builder& ir, const In& x) -> Out {
         return nonzero(ir, ir.and_(x.s0, x.s1));
     }, Fold::Yes},
    {F::SOP2, kSop2OpcodeOrB32, O::Src0Src1, [](Builder& ir, const In& x) -> Out {
         return nonzero(ir, ir.or_(x.s0, x.s1));
     }, Fold::Yes},
    {F::SOP2, kSop2OpcodeXorB32, O::Src0Src1, [](Builder& ir, const In& x) -> Out {
         return nonzero(ir, ir.xor_(x.s0, x.s1));
     }, Fold::Yes},
    {F::SOP2, kSop2OpcodeAndn2B32, O::Src0Src1, [](Builder& ir, const In& x) -> Out {
         return nonzero(ir, ir.and_(x.s0, ir.not_(x.s1)));
     }},
    {F::SOP2, kSop2OpcodeOrn2B32, O::Src0Src1, [](Builder& ir, const In& x) -> Out {
         return nonzero(ir, ir.or_(x.s0, ir.not_(x.s1)));
     }},
    {F::SOP2, kSop2OpcodeNandB32, O::Src0Src1, [](Builder& ir, const In& x) -> Out {
         return nonzero(ir, ir.not_(ir.and_(x.s0, x.s1)));
     }},
    {F::SOP2, kSop2OpcodeNorB32, O::Src0Src1, [](Builder& ir, const In& x) -> Out {
         return nonzero(ir, ir.not_(ir.or_(x.s0, x.s1)));
     }},
    {F::SOP2, kSop2OpcodeXnorB32, O::Src0Src1, [](Builder& ir, const In& x) -> Out {
         return nonzero(ir, ir.not_(ir.xor_(x.s0, x.s1)));
     }},
    // Shifts take the amount from S1[4:0].
    {F::SOP2, 0x1e, O::Src0Src1, [](Builder& ir, const In& x) -> Out {   // s_lshl_b32
         const Value amount = ir.and_(x.s1, ir.c32(31));
         return nonzero(ir, ir.shl(x.s0, amount));
     }, Fold::Yes},
    {F::SOP2, 0x20, O::Src0Src1, [](Builder& ir, const In& x) -> Out {   // s_lshr_b32
         const Value amount = ir.and_(x.s1, ir.c32(31));
         return nonzero(ir, ir.lshr(x.s0, amount));
     }, Fold::Yes},
    {F::SOP2, 0x22, O::Src0Src1, [](Builder& ir, const In& x) -> Out {   // s_ashr_i32
         const Value amount = ir.and_(x.s1, ir.c32(31));
         return nonzero(ir, ir.ashr(x.s0, amount));
     }, Fold::Yes},
    {F::SOP2, kSop2OpcodeBfmB32, O::Src0Src1, bfm_b32},
    {F::SOP2, 0x26, O::Src0Src1, [](Builder& ir, const In& x) -> Out {   // s_mul_i32, no SCC
         return {ir.imul(x.s0, x.s1), {}};
     }, Fold::Yes},
    {F::SOP2, 0x27, O::Src0Src1, bfe_u32},
    {F::SOP2, 0x2e, O::Src0Src1, lshl_add_u32<1>},
    {F::SOP2, 0x2f, O::Src0Src1, lshl_add_u32<2>},
    {F::SOP2, 0x30, O::Src0Src1, lshl_add_u32<3>},
    {F::SOP2, 0x31, O::Src0Src1, lshl_add_u32<4>},
    {F::SOP2, kSop2OpcodePackLlB32B16, O::Src0Src1, pack_b32_b16<false, false>},
    {F::SOP2, 0x33, O::Src0Src1, pack_b32_b16<false, true>},   // s_pack_lh_b32_b16
    {F::SOP2, 0x34, O::Src0Src1, pack_b32_b16<true, true>},    // s_pack_hh_b32_b16
    // s_mul_hi_u32 / s_mul_hi_i32 (gfx10 SOP2 0x35 / 0x36, llvm-mc; #462), no SCC.
    {F::SOP2, 0x35, O::Src0Src1, [](Builder& ir, const In& x) -> Out {
         return {ir.umulhi(x.s0, x.s1), {}};
     }},
    {F::SOP2, 0x36, O::Src0Src1, [](Builder& ir, const In& x) -> Out {
         return {ir.smulhi(x.s0, x.s1), {}};
     }},
    // SOP1
    {F::SOP1, kSop1OpcodeNotB32, O::Src0, [](Builder& ir, const In& x) -> Out {
         return nonzero(ir, ir.not_(x.s0));
     }},
    {F::SOP1, kSop1OpcodeBrevB32, O::Src0, [](Builder& ir, const In& x) -> Out {   // no SCC
         return {ir.bitreverse(x.s0), {}};
     }},
    {F::SOP1, kSop1OpcodeAbsI32, O::Src0, abs_i32},
    // SOPK: the immediate forms reuse the SOP2 definitions.
    {F::SOPK, kSopkOpcodeMovkI32, O::SimmSext, [](Builder&, const In& x) -> Out {
         return {x.s1, {}};
     }},
    {F::SOPK, kSopkOpcodeAddkI32, O::DstSimmSext, add_i32},
    {F::SOPK, kSopkOpcodeMulkI32, O::DstSimmSext, [](Builder& ir, const In& x) -> Out {
         return {ir.imul(x.s0, x.s1), {}};   // s_mulk_i32: no SCC
     }},
};

} // namespace

std::span<const ScalarOpDef> scalar_alu_defs() { return kDefs; }

} // namespace prosper::gpu::ir
