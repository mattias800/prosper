// vector_carry.cpp — the carry ops' per-lane values (AMD 70648 6.5; VOP2 0x28-0x2A, VOP3B
// 0x128-0x12A and 0x30F/0x310/0x319). Every result is modulo 2^32.
//
//   v_add_co_u32      D = S0 + S1,        carry  = the sum overflowed 32 bits
//   v_sub_co_u32      D = S0 - S1,        borrow = S0 < S1 (unsigned)
//   v_subrev_co_u32   D = S1 - S0,        borrow = S1 < S0
//   v_add_co_ci_u32   D = S0 + S1 + CIN,  carry  = either addition overflowed
//   v_sub_co_ci_u32   D = S0 - S1 - CIN,  borrow = either subtraction borrowed
//   v_subrev_co_ci_u32 D = S1 - S0 - CIN, borrow likewise
//
// The append order and nesting are the pre-IR emitter's (HOW_TO_ADD_A_FAMILY.md, step 3): the
// carry-in widens with Select(cin, 1, 0) first, and the two partial carries are joined with
// BoolSelect(c1, true, c2), which is c1 || c2.
#include "gpu/recompiler/ir/ops/vector_carry.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace prosper::gpu::ir {

namespace {

using F = Rdna2Format;
using In = VectorCarryIn;
using Out = VectorCarryOut;

Out add_co(Builder& ir, const In& x) {
    const Value d = ir.iadd(x.s0, x.s1);
    return {d, ir.ult(d, x.s0)};
}
// x - y with its borrow; sub passes (S0, S1), subrev (S1, S0).
Out sub_co(Builder& ir, Value x, Value y) {
    const Value d = ir.isub(x, y);
    return {d, ir.ult(x, y)};
}
Out add_co_ci(Builder& ir, const In& x) {
    const Value cin = ir.select(x.cin, ir.c32(1), ir.c32(0));
    const Value s1 = ir.iadd(x.s0, x.s1);
    const Value c1 = ir.ult(s1, x.s0);   // wrap in S0 + S1
    const Value d = ir.iadd(s1, cin);
    const Value c2 = ir.ult(d, s1);      // wrap in + CIN
    return {d, ir.bselect(c1, ir.ctrue(), c2)};
}
Out sub_co_ci(Builder& ir, Value cin_bit, Value x, Value y) {
    const Value cin = ir.select(cin_bit, ir.c32(1), ir.c32(0));
    const Value s1 = ir.isub(x, y);
    const Value b1 = ir.ult(x, y);       // borrow in x - y
    const Value d = ir.isub(s1, cin);
    const Value b2 = ir.ult(s1, cin);    // borrow in - CIN
    return {d, ir.bselect(b1, ir.ctrue(), b2)};
}

Out sub(Builder& ir, const In& x) { return sub_co(ir, x.s0, x.s1); }
Out subrev(Builder& ir, const In& x) { return sub_co(ir, x.s1, x.s0); }
Out sub_ci(Builder& ir, const In& x) { return sub_co_ci(ir, x.cin, x.s0, x.s1); }
Out subrev_ci(Builder& ir, const In& x) { return sub_co_ci(ir, x.cin, x.s1, x.s0); }

constexpr VectorCarryDef kDefs[] = {
    {F::VOP3, 0x30F, false, add_co},
    {F::VOP3, 0x310, false, sub},
    {F::VOP3, 0x319, false, subrev},
    {F::VOP2, 0x28, true, add_co_ci},
    {F::VOP2, 0x29, true, sub_ci},
    {F::VOP2, 0x2A, true, subrev_ci},
    {F::VOP3, 0x128, true, add_co_ci},
    {F::VOP3, 0x129, true, sub_ci},
    {F::VOP3, 0x12A, true, subrev_ci},
};

struct Built {
    Block block;
    Out out;
};
// Inputs: slot 0 = S0, slot 1 = S1, slot 2 = the carry-in Bool (reads_carry_in only).
Built build(const VectorCarryDef& def) {
    Built built;
    Builder ir(built.block);
    In x{ir.input(Type::U32), ir.input(Type::U32)};
    if (def.reads_carry_in) x.cin = ir.input(Type::Bool);
    built.out = def.build(ir, x);
    return built;
}

}   // namespace

std::span<const VectorCarryDef> vector_carry_defs() { return kDefs; }

const VectorCarryDef* find_vector_carry(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}

KnownCarry eval_vector_carry(const VectorCarryDef& def, Known s0, Known s1, Known cin) {
    const Built built = build(def);
    std::vector<Known> inputs{s0, s1};
    if (def.reads_carry_in) inputs.push_back(cin);
    const std::vector<Known> v = evaluate(built.block, inputs);
    return {v[built.out.d.index], v[built.out.carry.index]};
}

CarryIds emit_vector_carry(SpirvCompute& b, const VectorCarryDef& def, uint32_t s0, uint32_t s1,
                           uint32_t cin_mask) {
    if (def.reads_carry_in != (cin_mask != 0)) {
        std::fprintf(stderr, "[ir] carry-in mask %s for opcode 0x%x\n",
                     cin_mask ? "given to a carry op without one" : "missing", def.opcode);
        std::abort();
    }
    const Built built = build(def);
    const uint32_t inputs[] = {s0, s1, cin_mask};
    const std::vector<uint32_t> ids = emit_spirv(b, built.block, inputs);
    return {ids[built.out.d.index], ids[built.out.carry.index]};
}

}   // namespace prosper::gpu::ir
