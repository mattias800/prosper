// AMD 70648 12.12: low24 unsigned multiplicands, full32 accumulator, modulo2^32 result.
// The original mask, nested input reads, product and late S2 read retain their order.
#include "gpu/recompiler/ir/ops/vector_mad24.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
constexpr TernaryIntegerDef kDefs[] = {
    {Rdna2Format::VOP3, 0x143,
     [](Builder& ir, const TernaryInputs& x) {
         const Value m24 = ir.c32(0xffffff);
         const Value product = ir.imul(ir.and_(x.s0(), m24), ir.and_(x.s1(), m24));
         return ir.iadd(product, x.s2());
     }},
};
}   // namespace
std::span<const TernaryIntegerDef> vector_mad24_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_mad24_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
