// VOPC I16/U16 values: AMD 70648 section 12.9, Table 61. CMPX and e64 reuse these bases.
// Extend source 0, then source 1, before the relation, including EQ/NE. The nesting matches
// the old compare_integer16 emitter so the existing source and mask routes keep their bytes.
#include "gpu/recompiler/ir/ops/vector_compare.hpp"

namespace prosper::gpu::ir {
namespace {
using In = VectorCompareIn;

In signed_sources(Builder& ir, const In& in) {
    const Value x = ir.bfe_s(in.s0, ir.c32(0), ir.c32(16));
    const Value y = ir.bfe_s(in.s1, ir.c32(0), ir.c32(16));
    return {x, y};
}
In unsigned_sources(Builder& ir, const In& in) {
    const Value x = ir.bfe_u(in.s0, ir.c32(0), ir.c32(16));
    const Value y = ir.bfe_u(in.s1, ir.c32(0), ir.c32(16));
    return {x, y};
}

Value cmp_lt_i16(Builder& ir, const In& in) {
    const auto x = signed_sources(ir, in);
    return ir.slt(x.s0, x.s1);
}
Value cmp_eq_i16(Builder& ir, const In& in) {
    const auto x = signed_sources(ir, in);
    return ir.ieq(x.s0, x.s1);
}
Value cmp_le_i16(Builder& ir, const In& in) {
    const auto x = signed_sources(ir, in);
    return ir.sle(x.s0, x.s1);
}
Value cmp_gt_i16(Builder& ir, const In& in) {
    const auto x = signed_sources(ir, in);
    return ir.sgt(x.s0, x.s1);
}
Value cmp_ne_i16(Builder& ir, const In& in) {
    const auto x = signed_sources(ir, in);
    return ir.ine(x.s0, x.s1);
}
Value cmp_ge_i16(Builder& ir, const In& in) {
    const auto x = signed_sources(ir, in);
    return ir.sge(x.s0, x.s1);
}
Value cmp_lt_u16(Builder& ir, const In& in) {
    const auto x = unsigned_sources(ir, in);
    return ir.ult(x.s0, x.s1);
}
Value cmp_eq_u16(Builder& ir, const In& in) {
    const auto x = unsigned_sources(ir, in);
    return ir.ieq(x.s0, x.s1);
}
Value cmp_le_u16(Builder& ir, const In& in) {
    const auto x = unsigned_sources(ir, in);
    return ir.ule(x.s0, x.s1);
}
Value cmp_gt_u16(Builder& ir, const In& in) {
    const auto x = unsigned_sources(ir, in);
    return ir.ugt(x.s0, x.s1);
}
Value cmp_ne_u16(Builder& ir, const In& in) {
    const auto x = unsigned_sources(ir, in);
    return ir.ine(x.s0, x.s1);
}
Value cmp_ge_u16(Builder& ir, const In& in) {
    const auto x = unsigned_sources(ir, in);
    return ir.uge(x.s0, x.s1);
}

constexpr VectorCompareDef kDefs[] = {
    {0x89, cmp_lt_i16}, {0x8a, cmp_eq_i16}, {0x8b, cmp_le_i16}, {0x8c, cmp_gt_i16},
    {0x8d, cmp_ne_i16}, {0x8e, cmp_ge_i16}, {0xa9, cmp_lt_u16}, {0xaa, cmp_eq_u16},
    {0xab, cmp_le_u16}, {0xac, cmp_gt_u16}, {0xad, cmp_ne_u16}, {0xae, cmp_ge_u16},
};
} // namespace

std::span<const VectorCompareDef> vector_compare_i16_u16_defs() {
    return kDefs;
}

} // namespace prosper::gpu::ir
