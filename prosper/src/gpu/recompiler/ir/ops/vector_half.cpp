// VOP2 f16 arithmetic and RTZ packing. AMD 70648 12.7; expression order follows the old
// lowering exactly. The packed accumulator remains legacy until fused rounding is established.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/ir/ops/vector_half.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
Value build_half_rtz(Builder& ir, Value a, Value b) {
    // GLSL PackHalf2x16 does not guarantee RTZ, and its preceding floating-point min/max can
    // turn a NaN or true infinity into a finite endpoint. V_CVT_PKRTZ_F16_F32 always rounds
    // finite operands toward zero. Work on the source bits so host float controls and GLSL
    // exceptional-value min/max do not change the input class.
    const auto half_rtz = [&](Value bits) {
        const Value sign = ir.and_(ir.lshr(bits, ir.c32(16)), ir.c32(0x8000u));
        const Value exponent = ir.and_(ir.lshr(bits, ir.c32(23)), ir.c32(0xffu));
        const Value fraction = ir.and_(bits, ir.c32(0x7fffffu));

        const Value normal_exponent = ir.shl(ir.isub(exponent, ir.c32(112)), ir.c32(10));
        const Value normal_fraction = ir.lshr(fraction, ir.c32(13));
        const Value normal = ir.or_(normal_exponent, normal_fraction);

        // For f32 exponents 103..112, the hidden bit contributes to a half subnormal.
        // The shift expression is evaluated in SSA for every input, including exponents
        // outside that range. Bound it before OpShiftRightLogical (shift >= 32 is undefined).
        const Value sub_shift = ir.umin(ir.isub(ir.c32(126), exponent), ir.c32(31));
        const Value subnormal = ir.lshr(ir.or_(fraction, ir.c32(0x800000u)), sub_shift);
        const Value has_subnormal =
            ir.booland_(ir.uge(exponent, ir.c32(103)), ir.ult(exponent, ir.c32(113)));
        Value magnitude = ir.select(has_subnormal, subnormal, ir.c32(0));
        const Value has_normal =
            ir.booland_(ir.uge(exponent, ir.c32(113)), ir.ult(exponent, ir.c32(143)));
        magnitude = ir.select(has_normal, normal, magnitude);
        const Value finite_overflow =
            ir.booland_(ir.uge(exponent, ir.c32(143)), ir.ult(exponent, ir.c32(255)));
        magnitude = ir.select(finite_overflow, ir.c32(0x7bffu), magnitude);
        // True infinities stay infinite; NaN payloads become a signed quiet half NaN.
        const Value special =
            ir.select(ir.ine(fraction, ir.c32(0)), ir.c32(0x7e00u), ir.c32(0x7c00u));
        magnitude = ir.select(ir.ieq(exponent, ir.c32(255)), special, magnitude);
        return ir.or_(sign, magnitude);
    };
    return ir.or_(half_rtz(a), ir.shl(half_rtz(b), ir.c32(16)));
}
namespace {
using F = Rdna2Format;
using In = HalfOpIn;
Value pack_rtz(Builder& ir, const In& x) {
    return build_half_rtz(ir, x.s0, x.s1);
}
Value finish(Builder& ir, Value value, bool clamp) {
    if (clamp) value = ir.f32clamp01(value);
    return ir.pack_half_lo(value);
}
constexpr HalfOpDef kRtz{F::VOP2, 0x2f, pack_rtz};
constexpr HalfOpDef kDefs[] = {
    kRtz,
    {F::VOP2, 0x32,
     [](Builder& ir, const In& x) { return finish(ir, ir.f32add(x.s0, x.s1), x.clamp); }},
    {F::VOP2, 0x33,
     [](Builder& ir, const In& x) { return finish(ir, ir.f32sub(x.s0, x.s1), x.clamp); }},
    {F::VOP2, 0x35,
     [](Builder& ir, const In& x) { return finish(ir, ir.f32mul(x.s0, x.s1), x.clamp); }},
    {F::VOP2, 0x39,
     [](Builder& ir, const In& x) { return finish(ir, ir.f32nmax(x.s0, x.s1), x.clamp); }},
    {F::VOP2, 0x3a,
     [](Builder& ir, const In& x) { return finish(ir, ir.f32nmin(x.s0, x.s1), x.clamp); }},
};
struct Built {
    Block block;
    Value d;
};
Built build(const HalfOpDef& def, bool clamp) {
    Built out;
    Builder ir(out.block);
    const Value a = ir.input(Type::U32), b = ir.input(Type::U32);
    out.d = def.build(ir, {a, b, clamp});
    return out;
}
}   // namespace
std::span<const HalfOpDef> vector_half_defs() {
    return kDefs;
}
const HalfOpDef* find_vector_half_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
Known eval_half_op(const HalfOpDef& def, Known s0, Known s1, bool clamp) {
    const auto built = build(def, clamp);
    const Known inputs[] = {s0, s1};
    return evaluate(built.block, inputs)[built.d.index];
}
uint32_t emit_half_op(SpirvCompute& b, const HalfOpDef& def, uint32_t s0, uint32_t s1, bool clamp) {
    const auto built = build(def, clamp);
    const uint32_t inputs[] = {s0, s1};
    return emit_spirv(b, built.block, inputs)[built.d.index];
}
uint32_t emit_half_rtz(SpirvCompute& b, uint32_t s0, uint32_t s1) {
    return emit_half_op(b, kRtz, s0, s1);
}
}   // namespace prosper::gpu::ir
