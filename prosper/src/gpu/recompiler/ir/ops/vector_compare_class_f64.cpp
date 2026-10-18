// AMD 70648 Table 61: CLASS_F64 inspects the raw IEEE-754 bits, without FP arithmetic.
#include "gpu/recompiler/ir/ops/vector_compare.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"

#include <span>

namespace prosper::gpu::ir {
namespace {
Value class_f64(Builder& ir, const VectorCompareIn& x) {
    const Value sign = ir.ine(ir.and_(x.s0_hi, ir.c32(0x80000000u)), ir.c32(0));
    const Value exponent = ir.and_(x.s0_hi, ir.c32(0x7ff00000u));
    const Value fraction = ir.or_(x.s0, ir.and_(x.s0_hi, ir.c32(0x000fffffu)));
    const Value exponent_zero = ir.ieq(exponent, ir.c32(0));
    const Value exponent_all = ir.ieq(exponent, ir.c32(0x7ff00000u));
    const Value fraction_zero = ir.ieq(fraction, ir.c32(0));
    const Value quiet_nan = ir.ine(ir.and_(x.s0_hi, ir.c32(0x00080000u)), ir.c32(0));
    const Value nan_class = ir.select(quiet_nan, ir.c32(2), ir.c32(1));
    const Value inf_class = ir.select(sign, ir.c32(4), ir.c32(512));
    const Value exp_all_class = ir.select(fraction_zero, inf_class, nan_class);
    const Value zero_class = ir.select(sign, ir.c32(32), ir.c32(64));
    const Value subnormal_class = ir.select(sign, ir.c32(16), ir.c32(128));
    const Value exp_zero_class = ir.select(fraction_zero, zero_class, subnormal_class);
    const Value normal_class = ir.select(sign, ir.c32(8), ir.c32(256));
    const Value class_bit = ir.select(exponent_all, exp_all_class,
                                      ir.select(exponent_zero, exp_zero_class, normal_class));
    return ir.ine(ir.and_(x.s1, class_bit), ir.c32(0));
}
constexpr VectorCompareDef kDefs[] = {{0xa8, class_f64, true}};
} // namespace
std::span<const VectorCompareDef> vector_compare_class_f64_defs() {
    return kDefs;
}
} // namespace prosper::gpu::ir
