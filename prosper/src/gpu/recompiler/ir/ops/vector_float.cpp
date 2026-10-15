// VOP2 binary32 operations, AMD RDNA2 ISA 70648 section 12.7. The lowering retains the
// existing SPIR-V operation/order; FMA rounding discrepancies remain in the relocated switch.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
#include "gpu/recompiler/ir/ops/vector_float.hpp"
namespace prosper::gpu::ir {
namespace {
using F = Rdna2Format;
using In = VectorOpIn;
constexpr VectorOpDef kDefs[] = {
    {F::VOP2, 0x03, [](Builder& ir, const In& x) { return ir.f32add(x.s0, x.s1); }},
    {F::VOP2, 0x04, [](Builder& ir, const In& x) { return ir.f32sub(x.s0, x.s1); }},
    {F::VOP2, 0x05, [](Builder& ir, const In& x) { return ir.f32sub(x.s1, x.s0); }},
    {F::VOP2, 0x08, [](Builder& ir, const In& x) { return ir.f32mul(x.s0, x.s1); }},
    {F::VOP2, 0x0f, [](Builder& ir, const In& x) { return ir.f32nmin(x.s0, x.s1); }},
    {F::VOP2, 0x10, [](Builder& ir, const In& x) { return ir.f32nmax(x.s0, x.s1); }},
};
}   // namespace
std::span<const VectorOpDef> vector_float_defs() {
    return kDefs;
}
const VectorOpDef* find_vector_float_op(Rdna2Format fmt, uint32_t opcode) {
    for (const VectorOpDef& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
} // namespace prosper::gpu::ir
