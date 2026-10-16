// AMD 70648 12.12: BFE masks offset and width to five bits; BFI selects bits within one lane.
// Keep every source input at its old expression point, including both BFI temporaries.
#include "gpu/recompiler/ir/ops/vector_bitfield.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
using F = Rdna2Format;
using In = TernaryInputs;
constexpr TernaryIntegerDef kDefs[] = {
    {F::VOP3, 0x148,
     [](Builder& ir, const In& x) {
         const Value off = ir.and_(x.s1(), ir.c32(31));
         const Value cnt = ir.and_(x.s2(), ir.c32(31));
         return ir.bfe_u(x.s0(), off, cnt);
     }},
    {F::VOP3, 0x149,
     [](Builder& ir, const In& x) {
         const Value off = ir.and_(x.s1(), ir.c32(31));
         const Value cnt = ir.and_(x.s2(), ir.c32(31));
         return ir.bfe_s(x.s0(), off, cnt);
     }},
    {F::VOP3, 0x14a,
     [](Builder& ir, const In& x) {
         const Value s0 = x.s0();
         const Value t1 = ir.and_(s0, x.s1());
         const Value t2 = ir.and_(ir.not_(s0), x.s2());
         return ir.or_(t1, t2);
     }},
};
}   // namespace
std::span<const TernaryIntegerDef> vector_bitfield_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_bitfield_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
