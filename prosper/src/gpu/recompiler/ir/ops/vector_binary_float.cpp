// AMD70648 VOP2 F32 arithmetic, existing VOP3 encodings. Root modifiers and
// source operand evaluation order remain unchanged; no new guest form or fold is admitted.
#include "gpu/recompiler/ir/ops/vector_binary_float.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
using In = TernaryInputs;
Value add(Builder& ir, const In& x) {
    return ir.f32add(x.s0(), x.s1());
}
Value sub(Builder& ir, const In& x) {
    return ir.f32sub(x.s0(), x.s1());
}
Value subrev(Builder& ir, const In& x) {
    return ir.f32sub(x.s1(), x.s0());
}
Value mul(Builder& ir, const In& x) {
    return ir.f32mul(x.s0(), x.s1());
}
Value min(Builder& ir, const In& x) {
    return ir.f32nmin(x.s0(), x.s1());
}
Value max(Builder& ir, const In& x) {
    return ir.f32nmax(x.s0(), x.s1());
}
constexpr TernaryIntegerDef kDefs[] = {
    {Rdna2Format::VOP3, 0x103, add},    {Rdna2Format::VOP3, 0x104, sub},
    {Rdna2Format::VOP3, 0x105, subrev}, {Rdna2Format::VOP3, 0x108, mul},
    {Rdna2Format::VOP3, 0x10f, min},    {Rdna2Format::VOP3, 0x110, max}};
}   // namespace
std::span<const TernaryIntegerDef> vector_binary_float_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_binary_float_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
