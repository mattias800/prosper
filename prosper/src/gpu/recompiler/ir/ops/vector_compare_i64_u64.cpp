// VOPC I64/U64 values: AMD RDNA2 ISA 70648, section 12.9 (V_CMP_LT/EQ/LE/GT/NE/GE_I64 = 161-166,
// _U64 = 225-230; CMPX at +16). CMPX and e64 reuse these bases. Each definition reads both halves
// of each source (pair_sources) and packs them into one U64 word.
//
// The bytes are the pre-IR emitter's: every definition packs S0, then S1, and a signed one then
// packs both again with the sign bit flipped (pack_u64_signed_order) before the relation. So an
// I64 EQ/NE also emits the two unused order keys, and an I64 LT..GE the two unused plain packs.
// Dropping them is a byte change for its own, corpus-gated PR.
#include "gpu/recompiler/ir/ops/vector_compare.hpp"

#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"

#include <span>

namespace prosper::gpu::ir {
namespace {
using In = VectorCompareIn;

struct Words {
    Value lhs, rhs;   // the plain packs: hi:lo of S0 and S1
};
// Signed: the plain packs, then the order keys whose unsigned order is the signed order.
struct SignedWords {
    Value lhs, rhs, ordered_lhs, ordered_rhs;
};

Words unsigned_words(Builder& ir, const In& x) {
    const Value lhs = ir.pack_u64(x.s0, x.s0_hi);
    const Value rhs = ir.pack_u64(x.s1, x.s1_hi);
    return {lhs, rhs};
}
SignedWords signed_words(Builder& ir, const In& x) {
    const Words plain = unsigned_words(ir, x);
    const Value ordered_lhs = ir.pack_u64_signed_order(x.s0, x.s0_hi);
    const Value ordered_rhs = ir.pack_u64_signed_order(x.s1, x.s1_hi);
    return {plain.lhs, plain.rhs, ordered_lhs, ordered_rhs};
}

Value cmp_lt_i64(Builder& ir, const In& x) {
    const auto w = signed_words(ir, x);
    return ir.u64lt(w.ordered_lhs, w.ordered_rhs);
}
Value cmp_eq_i64(Builder& ir, const In& x) {
    const auto w = signed_words(ir, x);
    return ir.u64eq(w.lhs, w.rhs);
}
Value cmp_le_i64(Builder& ir, const In& x) {
    const auto w = signed_words(ir, x);
    return ir.u64le(w.ordered_lhs, w.ordered_rhs);
}
Value cmp_gt_i64(Builder& ir, const In& x) {
    const auto w = signed_words(ir, x);
    return ir.u64gt(w.ordered_lhs, w.ordered_rhs);
}
Value cmp_ne_i64(Builder& ir, const In& x) {
    const auto w = signed_words(ir, x);
    return ir.u64ne(w.lhs, w.rhs);
}
Value cmp_ge_i64(Builder& ir, const In& x) {
    const auto w = signed_words(ir, x);
    return ir.u64ge(w.ordered_lhs, w.ordered_rhs);
}
Value cmp_lt_u64(Builder& ir, const In& x) {
    const auto w = unsigned_words(ir, x);
    return ir.u64lt(w.lhs, w.rhs);
}
Value cmp_eq_u64(Builder& ir, const In& x) {
    const auto w = unsigned_words(ir, x);
    return ir.u64eq(w.lhs, w.rhs);
}
Value cmp_le_u64(Builder& ir, const In& x) {
    const auto w = unsigned_words(ir, x);
    return ir.u64le(w.lhs, w.rhs);
}
Value cmp_gt_u64(Builder& ir, const In& x) {
    const auto w = unsigned_words(ir, x);
    return ir.u64gt(w.lhs, w.rhs);
}
Value cmp_ne_u64(Builder& ir, const In& x) {
    const auto w = unsigned_words(ir, x);
    return ir.u64ne(w.lhs, w.rhs);
}
Value cmp_ge_u64(Builder& ir, const In& x) {
    const auto w = unsigned_words(ir, x);
    return ir.u64ge(w.lhs, w.rhs);
}

constexpr VectorCompareDef kDefs[] = {
    {0xa1, cmp_lt_i64, true}, {0xa2, cmp_eq_i64, true}, {0xa3, cmp_le_i64, true},
    {0xa4, cmp_gt_i64, true}, {0xa5, cmp_ne_i64, true}, {0xa6, cmp_ge_i64, true},
    {0xe1, cmp_lt_u64, true}, {0xe2, cmp_eq_u64, true}, {0xe3, cmp_le_u64, true},
    {0xe4, cmp_gt_u64, true}, {0xe5, cmp_ne_u64, true}, {0xe6, cmp_ge_u64, true},
};
} // namespace

std::span<const VectorCompareDef> vector_compare_i64_u64_defs() {
    return kDefs;
}

} // namespace prosper::gpu::ir
