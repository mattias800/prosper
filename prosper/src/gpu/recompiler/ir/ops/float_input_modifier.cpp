// AMD70648 section 6.4: flush an input subnormal to zero, retaining its sign. Integer
// transport also preserves that zero after GLSL math without relying on host FP controls.
#include "gpu/recompiler/ir/ops/float_input_modifier.hpp"
namespace prosper::gpu::ir {
Value build_flush_f32_input(Builder& ir, Value bits) {
    const auto magnitude = ir.and_(bits, ir.c32(0x7fffffffu));
    const auto sign = ir.and_(bits, ir.c32(0x80000000u));
    return ir.select(ir.ult(magnitude, ir.c32(0x00800000u)), sign, bits);
}
Value build_preserve_f32_zero(Builder& ir, Value input, Value result) {
    const auto zero = ir.ieq(ir.and_(input, ir.c32(0x7fffffffu)), ir.c32(0));
    return ir.select(zero, input, result);
}
} // namespace prosper::gpu::ir
