// AMD 70648 12.12: shift the within-lane {S0,S1} join; byte counts beyond7 zero-fill.
// All intermediate shifts stay below32, including select arms not chosen at runtime.
#include "gpu/recompiler/ir/ops/vector_align.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
Value align(Builder& ir, const TernaryInputs& x, bool bytes) {
    const Value s0 = x.s0();
    const Value s1 = x.s1();
    const Value count = ir.and_(x.s2(), ir.c32(31));
    const Value shift = bytes ? ir.and_(ir.shl(count, ir.c32(3)), ir.c32(31)) : count;
    const Value inverse = ir.and_(ir.isub(ir.c32(32), shift), ir.c32(31));
    const Value joined = ir.or_(ir.lshr(s1, shift), ir.shl(s0, inverse));
    const Value low = ir.select(ir.ieq(count, ir.c32(0)), s1, joined);
    if (!bytes) return low;
    const Value upper_byte = ir.isub(count, ir.c32(4));
    const Value upper_shift = ir.and_(ir.shl(upper_byte, ir.c32(3)), ir.c32(31));
    const Value upper = ir.lshr(s0, upper_shift);
    return ir.select(ir.ult(count, ir.c32(4)), low,
                     ir.select(ir.ult(count, ir.c32(8)), upper, ir.c32(0)));
}
constexpr TernaryIntegerDef kDefs[] = {
    {Rdna2Format::VOP3, kVop3OpcodeAlignbitB32,
     [](Builder& ir, const TernaryInputs& x) { return align(ir, x, false); }},
    {Rdna2Format::VOP3, kVop3OpcodeAlignbyteB32,
     [](Builder& ir, const TernaryInputs& x) { return align(ir, x, true); }},
};
}   // namespace
std::span<const TernaryIntegerDef> vector_align_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_align_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
