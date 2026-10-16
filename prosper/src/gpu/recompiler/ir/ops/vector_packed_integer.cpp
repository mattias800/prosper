// AMD 70648 §12.10/table87: existing packed integer half operations, modulo2^16.
// Selectors, NEG and inline materialization stay in the unchanged lowering wrapper.
#include "gpu/recompiler/ir/ops/vector_packed_integer.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
using In = TernaryInputs;
Value s16(Builder& ir, Value v) {
    return ir.bfe_s(v, ir.c32(0), ir.c32(16));
}
Value low16(Builder& ir, Value v) {
    return ir.and_(v, ir.c32(0xffff));
}
Value mad(Builder& ir, const In& x) {
    return low16(ir, ir.iadd(ir.imul(x.s0(), x.s1()), x.s2()));
}
Value mul(Builder& ir, const In& x) {
    return low16(ir, ir.imul(x.s0(), x.s1()));
}
Value add(Builder& ir, const In& x) {
    return low16(ir, ir.iadd(x.s0(), x.s1()));
}
Value sub(Builder& ir, const In& x) {
    return low16(ir, ir.isub(x.s0(), x.s1()));
}
Value shl(Builder& ir, const In& x) {
    return low16(ir, ir.shl(x.s1(), ir.and_(x.s0(), ir.c32(15))));
}
Value lshr(Builder& ir, const In& x) {
    return low16(ir, ir.lshr(x.s1(), ir.and_(x.s0(), ir.c32(15))));
}
Value ashr(Builder& ir, const In& x) {
    return low16(ir, ir.ashr(s16(ir, x.s1()), ir.and_(x.s0(), ir.c32(15))));
}
Value smax(Builder& ir, const In& x) {
    return low16(ir, ir.smax(s16(ir, x.s0()), s16(ir, x.s1())));
}
Value smin(Builder& ir, const In& x) {
    return low16(ir, ir.smin(s16(ir, x.s0()), s16(ir, x.s1())));
}
Value umax(Builder& ir, const In& x) {
    return low16(ir, ir.umax(x.s0(), x.s1()));
}
Value umin(Builder& ir, const In& x) {
    return low16(ir, ir.umin(x.s0(), x.s1()));
}
constexpr TernaryIntegerDef kDefs[] = {
    {Rdna2Format::VOP3P, 0x00, mad},  {Rdna2Format::VOP3P, 0x01, mul},
    {Rdna2Format::VOP3P, 0x02, add},  {Rdna2Format::VOP3P, 0x03, sub},
    {Rdna2Format::VOP3P, 0x04, shl},  {Rdna2Format::VOP3P, 0x05, lshr},
    {Rdna2Format::VOP3P, 0x06, ashr}, {Rdna2Format::VOP3P, 0x07, smax},
    {Rdna2Format::VOP3P, 0x08, smin}, {Rdna2Format::VOP3P, 0x09, mad},
    {Rdna2Format::VOP3P, 0x0a, add},  {Rdna2Format::VOP3P, 0x0b, sub},
    {Rdna2Format::VOP3P, 0x0c, umax}, {Rdna2Format::VOP3P, 0x0d, umin}};
}   // namespace
std::span<const TernaryIntegerDef> vector_packed_integer_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_packed_integer_op(Rdna2Format fmt, uint32_t op) {
    for (const auto& d : kDefs)
        if (d.fmt == fmt && d.opcode == op) return &d;
    return nullptr;
}
}   // namespace prosper::gpu::ir
