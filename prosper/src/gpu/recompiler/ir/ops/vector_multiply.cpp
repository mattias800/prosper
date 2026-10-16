// AMD 70648 12.12: low 32 bits, high unsigned 32 bits and high signed 32 bits.
// Match the old emitter's expression nesting; modifiers and the predicated write stay outside.
#include "gpu/recompiler/ir/ops/vector_multiply.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
using F = Rdna2Format;
using In = VectorOpIn;
constexpr VectorOpDef kDefs[] = {
    {F::VOP3, kVop3OpcodeMulLoU32, [](Builder& ir, const In& x) { return ir.imul(x.s0, x.s1); }},
    {F::VOP3, kVop3OpcodeMulHiU32, [](Builder& ir, const In& x) { return ir.umulhi(x.s0, x.s1); }},
    {F::VOP3, kVop3OpcodeMulHiI32, [](Builder& ir, const In& x) { return ir.smulhi(x.s0, x.s1); }},
};
}   // namespace
std::span<const VectorOpDef> vector_multiply_defs() {
    return kDefs;
}
const VectorOpDef* find_vector_multiply_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
