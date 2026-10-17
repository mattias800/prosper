// CLASS_F32 tests raw IEEE-754 classes: AMD 70648, 12.9 Table 61.
// Inputs are SRC0 after integer ABS/NEG preparation and the raw SRC1 mask.
// Preserve the old integer graph, constant creation and nested-select order.
#include "gpu/recompiler/ir/ops/vector_compare.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"

#include <span>

namespace prosper::gpu::ir {
namespace {
Value class_f32(Builder& ir, const VectorCompareIn& x) {
    const Value sign = ir.ine(ir.and_(x.s0, ir.c32(0x80000000u)), ir.c32(0));
    const Value exponent = ir.and_(x.s0, ir.c32(0x7f800000u));
    const Value mantissa = ir.and_(x.s0, ir.c32(0x007fffffu));
    const Value exponent_zero = ir.ieq(exponent, ir.c32(0));
    const Value exponent_all = ir.ieq(exponent, ir.c32(0x7f800000u));
    const Value mantissa_zero = ir.ieq(mantissa, ir.c32(0));
    const Value quiet_nan = ir.ine(ir.and_(mantissa, ir.c32(0x00400000u)), ir.c32(0));

    // AMD's mask order is sNaN, qNaN, -Inf, -normal, -subnormal, -zero,
    // +zero, +subnormal, +normal, +Inf. Select the input's one-hot class bit,
    // then test it against the raw SRC1 class mask.
    const Value nan_class = ir.select(quiet_nan, ir.c32(2), ir.c32(1));
    const Value inf_class = ir.select(sign, ir.c32(4), ir.c32(512));
    const Value exp_all_class = ir.select(mantissa_zero, inf_class, nan_class);
    const Value zero_class = ir.select(sign, ir.c32(32), ir.c32(64));
    const Value subnormal_class = ir.select(sign, ir.c32(16), ir.c32(128));
    const Value exp_zero_class = ir.select(mantissa_zero, zero_class, subnormal_class);
    const Value normal_class = ir.select(sign, ir.c32(8), ir.c32(256));
    const Value class_bit = ir.select(exponent_all, exp_all_class,
                                      ir.select(exponent_zero, exp_zero_class, normal_class));
    return ir.ine(ir.and_(x.s1, class_bit), ir.c32(0));
}
constexpr VectorCompareDef kDefs[] = {{0x88, class_f32}};
} // namespace
std::span<const VectorCompareDef> vector_compare_class_f32_defs() {
    return kDefs;
}
} // namespace prosper::gpu::ir
