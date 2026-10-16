// Integer-only power-of-two output scaling: host FP/half packing cannot override guest modes.
#include "gpu/recompiler/ir/ops/float_output_modifier.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
namespace {
struct Normalized {
    Value significand, exponent, magnitude, negative, sign;
};
Normalized normalize(Builder& ir, Value bits, uint8_t scale) {
    const Value magnitude = ir.and_(bits, ir.c32(0x7fffffffu));
    const Value fraction = ir.and_(bits, ir.c32(0x7fffffu));
    const Value raw_exp = ir.lshr(magnitude, ir.c32(23));
    const Value subnormal = ir.ieq(raw_exp, ir.c32(0));
    const Value shift = ir.isub(ir.ffbh_u32(ir.or_(fraction, ir.c32(1))), ir.c32(8));
    const Value significand =
        ir.select(subnormal, ir.shl(fraction, shift), ir.or_(fraction, ir.c32(0x800000u)));
    const Value exponent =
        ir.select(subnormal, ir.isub(ir.c32(uint32_t(-126)), shift), ir.isub(raw_exp, ir.c32(127)));
    const Value sign = ir.and_(bits, ir.c32(0x80000000u));
    const int32_t delta = scale == 3 ? -1 : scale;
    return {significand, ir.iadd(exponent, ir.c32(uint32_t(delta))), magnitude,
            ir.ine(sign, ir.c32(0)), sign};
}
Value round_significand(Builder& ir, const Normalized& x, Value shift, uint8_t rounding) {
    // Every SSA shift executes, even on an unselected path. Clamp to [1,31] and select the
    // exact zero-shift/large-shift cases; no OpShift operand can reach the word width.
    const Value safe = ir.umin(ir.umax(shift, ir.c32(1)), ir.c32(31));
    Value truncated = ir.select(ir.uge(shift, ir.c32(32)), ir.c32(0), ir.lshr(x.significand, safe));
    truncated = ir.select(ir.ieq(shift, ir.c32(0)), x.significand, truncated);
    const Value remainder = ir.isub(x.significand, ir.shl(truncated, safe));
    const Value nonzero = ir.booland_(ir.ine(shift, ir.c32(0)), ir.ine(remainder, ir.c32(0)));
    Value increment = ir.cfalse();
    if (rounding == 0) {
        const Value halfway = ir.shl(ir.c32(1), ir.isub(safe, ir.c32(1)));
        const Value tie = ir.booland_(ir.ieq(remainder, halfway),
                                      ir.ine(ir.and_(truncated, ir.c32(1)), ir.c32(0)));
        increment = ir.booland_(nonzero, ir.booland_(ir.ule(shift, ir.c32(24)),
                                                     ir.boolor_(ir.ugt(remainder, halfway), tie)));
    } else if (rounding == 1) {
        increment = ir.booland_(nonzero, ir.ieq(x.sign, ir.c32(0)));
    } else if (rounding == 2) {
        increment = ir.booland_(nonzero, x.negative);
    }
    return ir.iadd(truncated, ir.select(increment, ir.c32(1), ir.c32(0)));
}
Value scale(Builder& ir, Value bits, FloatOutputModifier m) {
    const uint32_t fraction_bits = m.width == FloatResultWidth::F16 ? 10 : 23;
    const int32_t minimum = m.width == FloatResultWidth::F16 ? -14 : -126;
    const int32_t maximum = m.width == FloatResultWidth::F16 ? 15 : 127;
    const auto x = normalize(ir, bits, m.scale);
    const Value under_shift = ir.smax(ir.isub(ir.c32(uint32_t(minimum)), x.exponent), ir.c32(0));
    const Value shift = ir.iadd(under_shift, ir.c32(23 - fraction_bits));
    const Value rounded = round_significand(ir, x, shift, m.rounding);
    const Value carry = ir.uge(rounded, ir.c32(2u << fraction_bits));
    const Value significand = ir.select(carry, ir.lshr(rounded, ir.c32(1)), rounded);
    const Value exponent = ir.iadd(ir.smax(x.exponent, ir.c32(uint32_t(minimum))),
                                   ir.select(carry, ir.c32(1), ir.c32(0)));
    const Value fraction =
        ir.shl(ir.and_(significand, ir.c32((1u << fraction_bits) - 1)), ir.c32(23 - fraction_bits));
    const Value normal = ir.or_(ir.shl(ir.iadd(exponent, ir.c32(127)), ir.c32(23)), fraction);
    const Value max_finite = ir.c32(m.width == FloatResultWidth::F16 ? 0x477fe000 : 0x7f7fffff);
    Value overflow = ir.c32(0x7f800000);
    if (m.rounding == 3)
        overflow = max_finite;
    else if (m.rounding == 1)
        overflow = ir.select(x.negative, max_finite, overflow);
    else if (m.rounding == 2)
        overflow = ir.select(x.negative, overflow, max_finite);
    Value result =
        ir.or_(x.sign, ir.select(ir.sgt(exponent, ir.c32(uint32_t(maximum))), overflow, normal));
    // Round first: the largest subnormal candidate can round up to the minimum normal.
    result = ir.select(ir.ult(significand, ir.c32(1u << fraction_bits)), ir.c32(0), result);
    result = ir.select(ir.ieq(x.magnitude, ir.c32(0)), ir.c32(0), result);
    const Value special =
        ir.select(ir.ugt(x.magnitude, ir.c32(0x7f800000)), ir.or_(bits, ir.c32(0x400000)), bits);
    return ir.select(ir.uge(x.magnitude, ir.c32(0x7f800000)), special, result);
}
struct Built {
    Block block;
    Value result;
};
Built build(FloatOutputModifier modifier) {
    Built out;
    Builder ir(out.block);
    out.result = build_float_output_modifier(ir, ir.input(Type::U32), modifier);
    return out;
}
} // namespace
Value build_float_output_modifier(Builder& ir, Value bits, FloatOutputModifier modifier) {
    if (!modifier.scale) return bits;
    if (modifier.scale > 3 || modifier.rounding > 3 || modifier.width == FloatResultWidth::F64) {
        std::fprintf(stderr, "[ir] unsupported output modifier representation\n");
        std::abort();
    }
    return scale(ir, bits, modifier);
}
Known eval_float_output_modifier(Known bits, FloatOutputModifier modifier) {
    const auto built = build(modifier);
    const Known inputs[] = {bits};
    return evaluate(built.block, inputs)[built.result.index];
}
uint32_t emit_float_output_modifier(SpirvCompute& b, uint32_t bits, FloatOutputModifier modifier) {
    if (!modifier.scale) return bits;
    const auto built = build(modifier);
    const uint32_t inputs[] = {bits};
    return emit_spirv(b, built.block, inputs)[built.result.index];
}
} // namespace prosper::gpu::ir
