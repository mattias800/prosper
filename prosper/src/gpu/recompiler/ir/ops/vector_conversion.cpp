// AMD 70648 12.8: existing VOP1 conversions, preserving expression nesting/emission order.
// No additional guest constant-folding domain or encoding is admitted.
// CONFIDENCE: MED: preserves implicit FP modes; eval/oracle use nominal RTE.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
namespace prosper::gpu::ir {
namespace {
using F = Rdna2Format;
using In = ConversionIn;
constexpr ConversionDef kDefs[] = {
    {F::VOP1, 0x05, [](Builder& ir, const In& x) { return ir.i32_to_f32(x.s0); }},
    {F::VOP1, 0x06, [](Builder& ir, const In& x) { return ir.u32_to_f32(x.s0); }},
    {F::VOP1, 0x07, [](Builder& ir, const In& x) { return ir.f32_to_u32_sat(x.s0); }},
    {F::VOP1, 0x08, [](Builder& ir, const In& x) { return ir.f32_to_i32_sat(x.s0); }},
    {F::VOP1, 0x0a, [](Builder& ir, const In& x) { return ir.pack_half_lo(x.s0); }},
    {F::VOP1, 0x0b, [](Builder& ir, const In& x) { return ir.unpack_half(x.s0, x.high ? 1 : 0); }},
    {F::VOP1, 0x0c,
     [](Builder& ir, const In& x) {
         const Value lower = ir.f32floor(x.s0);
         const Value fraction = ir.f32sub(x.s0, lower);
         const Value round_up = ir.f32ge(fraction, ir.c32(0x3f000000));
         const Value rounded = ir.f32add(lower, ir.select(round_up, ir.c32(0x3f800000), ir.c32(0)));
         return ir.f32_to_i32_sat(rounded);
     }},
    {F::VOP1, 0x0d, [](Builder& ir, const In& x) { return ir.f32_to_i32_sat(ir.f32floor(x.s0)); }},
    {F::VOP1, 0x0e,
     [](Builder& ir, const In& x) {
         return ir.f32mul(ir.i32_to_f32(ir.bfe_s(x.s0, ir.c32(0), ir.c32(4))), ir.c32(0x3d800000));
     }},
    {F::VOP1, 0x11,
     [](Builder& ir, const In& x) { return ir.u32_to_f32(ir.bfe_u(x.s0, ir.c32(0), ir.c32(8))); }},
    {F::VOP1, 0x12,
     [](Builder& ir, const In& x) { return ir.u32_to_f32(ir.bfe_u(x.s0, ir.c32(8), ir.c32(8))); }},
    {F::VOP1, 0x13,
     [](Builder& ir, const In& x) { return ir.u32_to_f32(ir.bfe_u(x.s0, ir.c32(16), ir.c32(8))); }},
    {F::VOP1, 0x14,
     [](Builder& ir, const In& x) { return ir.u32_to_f32(ir.bfe_u(x.s0, ir.c32(24), ir.c32(8))); }},
};
constexpr ConversionDef kHalfIntegerDefs[] = {
    {F::VOP1, 0x50,
     [](Builder& ir, const In& x) {
         const Value word = ir.and_(x.s0, ir.c32(0xffff));
         const Value result = ir.pack_half_lo(ir.u32_to_f32(word));
         return ir.and_(result, ir.c32(0xffff));
     }},
    {F::VOP1, 0x51,
     [](Builder& ir, const In& x) {
         const Value word = ir.and_(x.s0, ir.c32(0xffff));
         const Value result = ir.pack_half_lo(ir.i32_to_f32(ir.bfe_s(word, ir.c32(0), ir.c32(16))));
         return ir.and_(result, ir.c32(0xffff));
     }},
    {F::VOP1, 0x52,
     [](Builder& ir, const In& x) {
         const Value f = ir.unpack_half(x.s0, 0);
         const Value result = ir.umin(ir.f32_to_u32_sat(f), ir.c32(0xffff));
         return ir.and_(result, ir.c32(0xffff));
     }},
    {F::VOP1, 0x53,
     [](Builder& ir, const In& x) {
         const Value f = ir.unpack_half(x.s0, 0);
         const Value result =
             ir.smin(ir.smax(ir.f32_to_i32_sat(f), ir.c32(0xffff8000)), ir.c32(0x7fff));
         return ir.and_(result, ir.c32(0xffff));
     }},
};
struct Built {
    Block block;
    Value d;
};
Built build(const ConversionDef& def, bool high) {
    Built out;
    Builder ir(out.block);
    out.d = def.build(ir, {ir.input(Type::U32), high});
    return out;
}
const ConversionDef* find(std::span<const ConversionDef> defs, Rdna2Format fmt, uint32_t op) {
    for (const auto& def : defs)
        if (def.fmt == fmt && def.opcode == op) return &def;
    return nullptr;
}
}   // namespace
std::span<const ConversionDef> vector_conversion_defs() {
    return kDefs;
}
std::span<const ConversionDef> vector_half_integer_conversion_defs() {
    return kHalfIntegerDefs;
}
const ConversionDef* find_vector_conversion_op(Rdna2Format fmt, uint32_t op) {
    return find(kDefs, fmt, op);
}
const ConversionDef* find_vector_half_integer_conversion_op(Rdna2Format fmt, uint32_t op) {
    return find(kHalfIntegerDefs, fmt, op);
}
Known eval_conversion_op(const ConversionDef& def, Known s0, bool high) {
    const Built out = build(def, high);
    const Known inputs[] = {s0};
    return evaluate(out.block, inputs)[out.d.index];
}
uint32_t emit_conversion_op(SpirvCompute& b, const ConversionDef& def, uint32_t s0, bool high) {
    const Built out = build(def, high);
    const uint32_t inputs[] = {s0};
    return emit_spirv(b, out.block, inputs)[out.d.index];
}
}   // namespace prosper::gpu::ir
