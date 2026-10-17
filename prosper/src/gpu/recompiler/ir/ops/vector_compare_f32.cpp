// vector_compare_f32.cpp — V_CMP_*_F32 (VOPC 0x00-0x0F; V_CMPX_*_F32 at 0x10-0x1F share them).
//
// AMD "RDNA 2" ISA (doc 70648) 12.9, Table 61. The relations are IEEE-754's: a NaN operand makes
// LT/EQ/LE/GT/LG/GE false (ordered), and the N-forms are the plain negations of those, so a NaN
// makes NGE/NLG/NGT/NLE/NEQ/NLT true ("With NAN inputs this is not the same operation as <" etc).
// LG is ordered not-equal; NEQ is its unordered counterpart. +0 and -0 are equal. Each definition
// appends in the order, and with the nesting, the pre-IR emitter used (HOW_TO_ADD_A_FAMILY.md).
#include "gpu/recompiler/ir/ops/vector_compare.hpp"

namespace prosper::gpu::ir {

namespace {

using In = VectorCompareIn;

Value cmp_f(Builder& ir, const In&) { return ir.cfalse(); }
Value cmp_lt(Builder& ir, const In& x) { return ir.f32lt(x.s0, x.s1); }
Value cmp_eq(Builder& ir, const In& x) { return ir.f32eq(x.s0, x.s1); }
Value cmp_le(Builder& ir, const In& x) { return ir.f32le(x.s0, x.s1); }
Value cmp_gt(Builder& ir, const In& x) { return ir.f32gt(x.s0, x.s1); }
Value cmp_lg(Builder& ir, const In& x) { return ir.f32lg(x.s0, x.s1); }
Value cmp_ge(Builder& ir, const In& x) { return ir.f32ge(x.s0, x.s1); }

// O: neither source is a NaN. A word equals itself exactly when it is not a NaN.
Value cmp_o(Builder& ir, const In& x) {
    const Value s0_ordered = ir.f32eq(x.s0, x.s0);
    const Value s1_ordered = ir.f32eq(x.s1, x.s1);
    return ir.booland_(s0_ordered, s1_ordered);
}

// U: either source is a NaN.
Value cmp_u(Builder& ir, const In& x) {
    const Value s0_nan = ir.f32unordne(x.s0, x.s0);
    const Value s1_nan = ir.f32unordne(x.s1, x.s1);
    return ir.boolor_(s0_nan, s1_nan);
}

Value cmp_nge(Builder& ir, const In& x) { return ir.f32unordlt(x.s0, x.s1); }   // !(S0 >= S1)
Value cmp_nlg(Builder& ir, const In& x) { return ir.f32unordeq(x.s0, x.s1); }   // !(S0 <> S1)
Value cmp_ngt(Builder& ir, const In& x) { return ir.f32unordle(x.s0, x.s1); }   // !(S0 > S1)
Value cmp_nle(Builder& ir, const In& x) { return ir.f32unordgt(x.s0, x.s1); }   // !(S0 <= S1)
Value cmp_neq(Builder& ir, const In& x) { return ir.f32unordne(x.s0, x.s1); }   // !(S0 == S1)
Value cmp_nlt(Builder& ir, const In& x) { return ir.f32unordge(x.s0, x.s1); }   // !(S0 < S1)
Value cmp_tru(Builder& ir, const In&) { return ir.ctrue(); }

constexpr VectorCompareDef kDefs[] = {
    {0x00, cmp_f},   {0x01, cmp_lt},  {0x02, cmp_eq},  {0x03, cmp_le},
    {0x04, cmp_gt},  {0x05, cmp_lg},  {0x06, cmp_ge},  {0x07, cmp_o},
    {0x08, cmp_u},   {0x09, cmp_nge}, {0x0a, cmp_nlg}, {0x0b, cmp_ngt},
    {0x0c, cmp_nle}, {0x0d, cmp_neq}, {0x0e, cmp_nlt}, {0x0f, cmp_tru},
};

}   // namespace

std::span<const VectorCompareDef> vector_compare_f32_defs() { return kDefs; }

} // namespace prosper::gpu::ir
