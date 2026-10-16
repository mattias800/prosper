// AMD 70648 12.8: per-lane F32 unary operations. Trig takes revolutions; preserve the
// old rounded 2*pi multiplication and nesting. CONFIDENCE: MED: the retained GLSL math
// inherits driver approximation/FP-mode behavior, not native-unit bit identity. IFLAG's
// numeric reciprocal is unchanged; this migration adds no floating/integer exception model.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <span>
#include <cstdint>
#include "gpu/recompiler/ir/ops/vector_float_unary.hpp"
namespace prosper::gpu::ir {
namespace {
using F = Rdna2Format;
using In = VectorOpIn;
constexpr VectorOpDef kDefs[] = {
    {F::VOP1, 0x20, [](Builder& ir, const In& x) { return ir.f32fract(x.s0); }},
    {F::VOP1, 0x21, [](Builder& ir, const In& x) { return ir.f32trunc(x.s0); }},
    {F::VOP1, 0x22, [](Builder& ir, const In& x) { return ir.f32ceil(x.s0); }},
    {F::VOP1, 0x23, [](Builder& ir, const In& x) { return ir.f32roundeven(x.s0); }},
    {F::VOP1, 0x24, [](Builder& ir, const In& x) { return ir.f32floor(x.s0); }},
    {F::VOP1, 0x25, [](Builder& ir, const In& x) { return ir.f32exp2(x.s0); }},
    {F::VOP1, 0x27, [](Builder& ir, const In& x) { return ir.f32log2(x.s0); }},
    {F::VOP1, 0x2a, [](Builder& ir, const In& x) { return ir.f32rcp(x.s0); }},
    {F::VOP1, 0x2b, [](Builder& ir, const In& x) { return ir.f32rcp(x.s0); }},
    {F::VOP1, 0x2e, [](Builder& ir, const In& x) { return ir.f32inversesqrt(x.s0); }},
    {F::VOP1, 0x33, [](Builder& ir, const In& x) { return ir.f32sqrt(x.s0); }},
    {F::VOP1, 0x35,
     [](Builder& ir, const In& x) { return ir.f32sin(ir.f32mul(x.s0, ir.c32(0x40c90fdb))); }},
    {F::VOP1, 0x36,
     [](Builder& ir, const In& x) { return ir.f32cos(ir.f32mul(x.s0, ir.c32(0x40c90fdb))); }},
};
}   // namespace
std::span<const VectorOpDef> vector_float_unary_defs() {
    return kDefs;
}
const VectorOpDef* find_vector_float_unary_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
} // namespace prosper::gpu::ir
