// AMD 70648 12.12 native16 integer arithmetic, shifts and order statistics.
// Keep the old expression nesting and helper boundaries: lazy S2 and constant allocation
// order are observable in the byte-identical SPIR-V migration gate.
#include "gpu/recompiler/ir/ops/vector_integer16.hpp"
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
Value s16(Builder& ir, Value value) {
    return ir.bfe_s(value, ir.c32(0), ir.c32(16));
}
Value low16(Builder& ir, Value value) {
    return ir.and_(value, ir.c32(0xffff));
}
Value median(Builder& ir, Value x, Value y, Value z, bool signed_words) {
    auto mn = [&](Value p, Value q) { return signed_words ? ir.smin(p, q) : ir.umin(p, q); };
    auto mx = [&](Value p, Value q) { return signed_words ? ir.smax(p, q) : ir.umax(p, q); };
    return mx(mn(x, y), mx(mn(x, z), mn(y, z)));
}
constexpr TernaryIntegerDef kDefs[] = {
    {F::VOP3, 0x340,
     [](Builder& ir, const In& x) { return low16(ir, ir.iadd(ir.imul(x.s0(), x.s1()), x.s2())); }},
    {F::VOP3, 0x35e,
     [](Builder& ir, const In& x) { return low16(ir, ir.iadd(ir.imul(x.s0(), x.s1()), x.s2())); }},
    {F::VOP3, 0x353,
     [](Builder& ir, const In& x) { return low16(ir, ir.umin(ir.umin(x.s0(), x.s1()), x.s2())); }},
    {F::VOP3, 0x356,
     [](Builder& ir, const In& x) { return low16(ir, ir.umax(ir.umax(x.s0(), x.s1()), x.s2())); }},
    {F::VOP3, 0x352,
     [](Builder& ir, const In& x) {
         return low16(ir, ir.smin(ir.smin(s16(ir, x.s0()), s16(ir, x.s1())), s16(ir, x.s2())));
     }},
    {F::VOP3, 0x355,
     [](Builder& ir, const In& x) {
         return low16(ir, ir.smax(ir.smax(s16(ir, x.s0()), s16(ir, x.s1())), s16(ir, x.s2())));
     }},
    {F::VOP3, 0x359,
     [](Builder& ir, const In& x) { return low16(ir, median(ir, x.s0(), x.s1(), x.s2(), false)); }},
    {F::VOP3, 0x358,
     [](Builder& ir, const In& x) {
         return low16(ir, median(ir, s16(ir, x.s0()), s16(ir, x.s1()), s16(ir, x.s2()), true));
     }},
    {F::VOP3, 0x303, [](Builder& ir, const In& x) { return low16(ir, ir.iadd(x.s0(), x.s1())); }},
    {F::VOP3, 0x30d, [](Builder& ir, const In& x) { return low16(ir, ir.iadd(x.s0(), x.s1())); }},
    {F::VOP3, 0x304, [](Builder& ir, const In& x) { return low16(ir, ir.isub(x.s0(), x.s1())); }},
    {F::VOP3, 0x30e, [](Builder& ir, const In& x) { return low16(ir, ir.isub(x.s0(), x.s1())); }},
    {F::VOP3, 0x305, [](Builder& ir, const In& x) { return low16(ir, ir.imul(x.s0(), x.s1())); }},
    {F::VOP3, 0x309, [](Builder& ir, const In& x) { return low16(ir, ir.umax(x.s0(), x.s1())); }},
    {F::VOP3, 0x30b, [](Builder& ir, const In& x) { return low16(ir, ir.umin(x.s0(), x.s1())); }},
    {F::VOP3, 0x30a,
     [](Builder& ir, const In& x) { return low16(ir, ir.smax(s16(ir, x.s0()), s16(ir, x.s1()))); }},
    {F::VOP3, 0x30c,
     [](Builder& ir, const In& x) { return low16(ir, ir.smin(s16(ir, x.s0()), s16(ir, x.s1()))); }},
    {F::VOP3, 0x314,
     [](Builder& ir, const In& x) {
         return low16(ir, ir.shl(x.s1(), ir.and_(x.s0(), ir.c32(15))));
     }},
    {F::VOP3, 0x307,
     [](Builder& ir, const In& x) {
         return low16(ir, ir.lshr(x.s1(), ir.and_(x.s0(), ir.c32(15))));
     }},
    {F::VOP3, 0x308,
     [](Builder& ir, const In& x) {
         return low16(ir, ir.ashr(s16(ir, x.s1()), ir.and_(x.s0(), ir.c32(15))));
     }},
};
} // namespace
std::span<const TernaryIntegerDef> vector_integer16_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_integer16_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
} // namespace prosper::gpu::ir
