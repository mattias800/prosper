// AMD70648 12.12: a wrapped width/offset bitfield and a per-word bit count plus accumulator.
// Preserve the old statements and nested lazy source positions for byte-identical SPIR-V.
#include "gpu/recompiler/ir/ops/vector_bit_utilities.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
constexpr TernaryIntegerDef kDefs[] = {
    {Rdna2Format::VOP3, 0x363,
     [](Builder& ir, const TernaryInputs& x) {
         const Value width = ir.and_(x.s0(), ir.c32(31));
         const Value offset = ir.and_(x.s1(), ir.c32(31));
         const Value mask = ir.isub(ir.shl(ir.c32(1), width), ir.c32(1));
         return ir.shl(mask, offset);
     }},
    {Rdna2Format::VOP3, 0x364,
     [](Builder& ir, const TernaryInputs& x) { return ir.iadd(ir.bit_count(x.s0()), x.s1()); }},
};
}   // namespace
std::span<const TernaryIntegerDef> vector_bit_utilities_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_bit_utilities_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
