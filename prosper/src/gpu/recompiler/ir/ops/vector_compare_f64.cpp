// vector_compare_f64.cpp — V_CMP_*_F64 (VOPC 0x20-0x2F; V_CMPX_*_F64 at 0x30-0x3F share them).
//
// AMD "RDNA 2" ISA (doc 70648) 12.9, Table 61: the same sixteen relations as the F32 group, on
// binary64 values. Ordered LT/EQ/LE/GT/LG/GE are false on a NaN; each N-form is the negation of
// its ordered relation, so it is true on a NaN; LG is ordered not-equal and NEQ its unordered
// twin; O is "neither is a NaN", U "either is". +0 equals -0.
//
// Each definition reads both halves of each source (pair_sources) and packs them into one U64
// word, which the F64 relations read as a binary64 value (ir.hpp, "Binary64 relations"). The
// lowering prepares the halves: the register pair or the 64-bit constant, ABS/NEG on the sign
// bit, and the MODE input-denormal flush (lower/vopc_float64.cpp). This group is an ADMISSION:
// the pre-IR emitter had no F64 compare, so there are no old bytes to match.
#include "gpu/recompiler/ir/ops/vector_compare.hpp"

#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"

#include <span>

namespace prosper::gpu::ir {
namespace {
using In = VectorCompareIn;

struct Words {
    Value lhs, rhs;   // hi:lo of S0 and S1
};
Words words(Builder& ir, const In& x) {
    const Value lhs = ir.pack_u64(x.s0, x.s0_hi);
    const Value rhs = ir.pack_u64(x.s1, x.s1_hi);
    return {lhs, rhs};
}

// One relation of the two packed sources, in S0, S1 order.
using Relation = Value (Builder::*)(Value, Value);
template <Relation R>
Value relation(Builder& ir, const In& x) {
    const Words w = words(ir, x);
    return (ir.*R)(w.lhs, w.rhs);
}

Value cmp_f(Builder& ir, const In&) { return ir.cfalse(); }
Value cmp_tru(Builder& ir, const In&) { return ir.ctrue(); }

// O: neither source is a NaN. A value equals itself exactly when it is not a NaN.
Value cmp_o(Builder& ir, const In& x) {
    const Words w = words(ir, x);
    const Value s0_ordered = ir.f64eq(w.lhs, w.lhs);
    const Value s1_ordered = ir.f64eq(w.rhs, w.rhs);
    return ir.booland_(s0_ordered, s1_ordered);
}

// U: either source is a NaN.
Value cmp_u(Builder& ir, const In& x) {
    const Words w = words(ir, x);
    const Value s0_nan = ir.f64unordne(w.lhs, w.lhs);
    const Value s1_nan = ir.f64unordne(w.rhs, w.rhs);
    return ir.boolor_(s0_nan, s1_nan);
}

// The N-forms: NGE = !(S0 >= S1) = unordered S0 < S1, and so on.
constexpr VectorCompareDef kDefs[] = {
    {0x20, cmp_f, true},
    {0x21, relation<&Builder::f64lt>, true},
    {0x22, relation<&Builder::f64eq>, true},
    {0x23, relation<&Builder::f64le>, true},
    {0x24, relation<&Builder::f64gt>, true},
    {0x25, relation<&Builder::f64lg>, true},   // ordered not-equal
    {0x26, relation<&Builder::f64ge>, true},
    {0x27, cmp_o, true},
    {0x28, cmp_u, true},
    {0x29, relation<&Builder::f64unordlt>, true},   // NGE
    {0x2a, relation<&Builder::f64unordeq>, true},   // NLG
    {0x2b, relation<&Builder::f64unordle>, true},   // NGT
    {0x2c, relation<&Builder::f64unordgt>, true},   // NLE
    {0x2d, relation<&Builder::f64unordne>, true},   // NEQ
    {0x2e, relation<&Builder::f64unordge>, true},   // NLT
    {0x2f, cmp_tru, true},
};
} // namespace

std::span<const VectorCompareDef> vector_compare_f64_defs() {
    return kDefs;
}

} // namespace prosper::gpu::ir
