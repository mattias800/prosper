// AMD 70648 section 12.9, Table 61: the six existing ordered F16 relations.
// The lowering unpacks the operands before these definitions. Binary16 embeds in
// binary32, so the relation uses prepared F32 bits; no conversion belongs here.
#include "gpu/recompiler/ir/ops/vector_compare.hpp"

#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"

#include <span>

namespace prosper::gpu::ir {
namespace {
using In = VectorCompareIn;

Value cmp_lt(Builder& ir, const In& x) {
    return ir.f32lt(x.s0, x.s1);
}
Value cmp_eq(Builder& ir, const In& x) {
    return ir.f32eq(x.s0, x.s1);
}
Value cmp_le(Builder& ir, const In& x) {
    return ir.f32le(x.s0, x.s1);
}
Value cmp_gt(Builder& ir, const In& x) {
    return ir.f32gt(x.s0, x.s1);
}
Value cmp_lg(Builder& ir, const In& x) {
    return ir.f32lg(x.s0, x.s1);
}
Value cmp_ge(Builder& ir, const In& x) {
    return ir.f32ge(x.s0, x.s1);
}

constexpr VectorCompareDef kDefs[] = {
    {0xc9, cmp_lt}, {0xca, cmp_eq}, {0xcb, cmp_le}, {0xcc, cmp_gt}, {0xcd, cmp_lg}, {0xce, cmp_ge},
};
} // namespace

std::span<const VectorCompareDef> vector_compare_f16_defs() {
    return kDefs;
}

} // namespace prosper::gpu::ir
