// AMD 70648 VOP1 f16 unary values, retaining the existing unpack/GLSL-math/pack lowering.
// CONFIDENCE: MED: nominal F32 math and RTE half reference do not establish native-unit
// approximation bits or a driver's implicit FP mode. No guest folding domain is widened.
#include "gpu/recompiler/ir/ops/vector_half_unary.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
Value finish(Builder& ir, Value value) {
    return ir.pack_half_lo(value);
}
using F = Rdna2Format;
using In = VectorOpIn;
constexpr VectorOpDef kDefs[] = {
    {F::VOP1, 0x54, [](Builder& ir, const In& x) { return finish(ir, ir.f32rcp(x.s0)); }},
    {F::VOP1, 0x55, [](Builder& ir, const In& x) { return finish(ir, ir.f32sqrt(x.s0)); }},
    {F::VOP1, 0x56, [](Builder& ir, const In& x) { return finish(ir, ir.f32inversesqrt(x.s0)); }},
    {F::VOP1, 0x57, [](Builder& ir, const In& x) { return finish(ir, ir.f32log2(x.s0)); }},
    {F::VOP1, 0x58, [](Builder& ir, const In& x) { return finish(ir, ir.f32exp2(x.s0)); }},
    {F::VOP1, 0x5b, [](Builder& ir, const In& x) { return finish(ir, ir.f32floor(x.s0)); }},
    {F::VOP1, 0x5c, [](Builder& ir, const In& x) { return finish(ir, ir.f32ceil(x.s0)); }},
    {F::VOP1, 0x5d, [](Builder& ir, const In& x) { return finish(ir, ir.f32trunc(x.s0)); }},
    {F::VOP1, 0x5e, [](Builder& ir, const In& x) { return finish(ir, ir.f32roundeven(x.s0)); }},
    {F::VOP1, 0x5f, [](Builder& ir, const In& x) { return finish(ir, ir.f32fract(x.s0)); }},
    {F::VOP1, 0x60,
     [](Builder& ir, const In& x) {
         return finish(ir, ir.f32sin(ir.f32mul(x.s0, ir.c32(0x40c90fdb))));
     }},
    {F::VOP1, 0x61,
     [](Builder& ir, const In& x) {
         return finish(ir, ir.f32cos(ir.f32mul(x.s0, ir.c32(0x40c90fdb))));
     }},
};
}   // namespace
std::span<const VectorOpDef> vector_half_unary_defs() {
    return kDefs;
}
const VectorOpDef* find_vector_half_unary_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
} // namespace prosper::gpu::ir
