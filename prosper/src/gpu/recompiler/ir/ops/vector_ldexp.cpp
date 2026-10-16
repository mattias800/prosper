// AMD70648 12.12 op866: S0.f * 2**S1.i, with the old emitter's raw-word scaling primitive.
#include "gpu/recompiler/ir/ops/vector_ldexp.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
constexpr TernaryIntegerDef kDefs[] = {
    {Rdna2Format::VOP3, 0x362,
     [](Builder& ir, const TernaryInputs& x) { return ir.f32ldexp(x.s0(), x.s1()); }}};
}   // namespace
std::span<const TernaryIntegerDef> vector_ldexp_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_ldexp_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
