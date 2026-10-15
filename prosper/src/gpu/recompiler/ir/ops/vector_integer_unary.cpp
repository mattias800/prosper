// AMD RDNA2 ISA 70648 12.8: MOV, NOT, reverse and unsigned first-bit scans. Their source is
// one word. The generic VectorOpIn's second slot is unused, not an invented guest operand.
#include "gpu/recompiler/ir/ops/vector_integer_unary.hpp"
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
    {F::VOP1, 0x01, [](Builder&, const In& x) { return x.s0; }},
    {F::VOP1, 0x37, [](Builder& ir, const In& x) { return ir.not_(x.s0); }},
    {F::VOP1, 0x38, [](Builder& ir, const In& x) { return ir.bitreverse(x.s0); }},
    {F::VOP1, kVop1OpcodeFfbhU32, [](Builder& ir, const In& x) { return ir.ffbh_u32(x.s0); }},
    {F::VOP1, kVop1OpcodeFfblB32,
     [](Builder& ir, const In& x) { return ir.ffbh_u32(ir.bitreverse(x.s0)); }},
};
}   // namespace
std::span<const VectorOpDef> vector_integer_unary_defs() {
    return kDefs;
}
const VectorOpDef* find_vector_integer_unary_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
} // namespace prosper::gpu::ir
