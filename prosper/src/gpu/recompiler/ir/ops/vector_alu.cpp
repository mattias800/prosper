// vector_alu.cpp — definitions of the VOP2 integer and bitwise operations, and their lookup,
// evaluation and emission.
//
// Semantics from AMD "RDNA 2" ISA (doc 70648) 12.7 (VOP2). Each definition appends in the order, and
// with the expression nesting, the pre-IR emitter used, so the emitted SPIR-V is byte-identical
// (HOW_TO_ADD_A_FAMILY.md). The *rev* forms take the shift amount (or the subtrahend) from S0.
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"

#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"

namespace prosper::gpu::ir {

Value build_vector_mul_u32_u24(Builder& ir, const VectorOpIn& x) {
    const Value mask = ir.c32(0x00ffffffu);
    return ir.imul(ir.and_(x.s0, mask), ir.and_(x.s1, mask));
}

Value build_vector_min_i32(Builder& ir, const VectorOpIn& x) {
    return ir.smin(x.s0, x.s1);
}

Value build_vector_max_i32(Builder& ir, const VectorOpIn& x) {
    return ir.smax(x.s0, x.s1);
}

Value build_vector_min_u32(Builder& ir, const VectorOpIn& x) {
    return ir.umin(x.s0, x.s1);
}

Value build_vector_max_u32(Builder& ir, const VectorOpIn& x) {
    return ir.umax(x.s0, x.s1);
}

Value build_vector_lshrrev_b32(Builder& ir, const VectorOpIn& x) {
    const Value amount = ir.and_(x.s0, ir.c32(31));
    return ir.lshr(x.s1, amount);
}

Value build_vector_ashrrev_i32(Builder& ir, const VectorOpIn& x) {
    const Value amount = ir.and_(x.s0, ir.c32(31));
    return ir.ashr(x.s1, amount);
}

Value build_vector_lshlrev_b32(Builder& ir, const VectorOpIn& x) {
    const Value amount = ir.and_(x.s0, ir.c32(31));
    return ir.shl(x.s1, amount);
}

Value build_vector_and_b32(Builder& ir, const VectorOpIn& x) {
    return ir.and_(x.s0, x.s1);
}

Value build_vector_or_b32(Builder& ir, const VectorOpIn& x) {
    return ir.or_(x.s0, x.s1);
}

Value build_vector_xor_b32(Builder& ir, const VectorOpIn& x) {
    return ir.xor_(x.s0, x.s1);
}

Value build_vector_xnor_b32(Builder& ir, const VectorOpIn& x) {   // v_xnor_b32
    return ir.not_(ir.xor_(x.s0, x.s1));
}

Value build_vector_add_nc_u32(Builder& ir, const VectorOpIn& x) {
    return ir.iadd(x.s0, x.s1);
}

Value build_vector_sub_nc_u32(Builder& ir, const VectorOpIn& x) {
    return ir.isub(x.s0, x.s1);
}

Value build_vector_subrev_nc_u32(Builder& ir, const VectorOpIn& x) {
    return ir.isub(x.s1, x.s0);
}

namespace {

using F = Rdna2Format;
using In = VectorOpIn;

constexpr VectorOpDef kDefs[] = {
    // v_mul_u32_u24: only the low 24 bits of each source; D is the low dword of the product.
    {F::VOP2, 0x0b, build_vector_mul_u32_u24},
    {F::VOP2, 0x11, build_vector_min_i32},   // v_min_i32
    {F::VOP2, 0x12, build_vector_max_i32},   // v_max_i32
    {F::VOP2, 0x13, build_vector_min_u32},   // v_min_u32
    {F::VOP2, 0x14, build_vector_max_u32},   // v_max_u32
    // v_lshrrev_b32 / v_ashrrev_i32 / v_lshlrev_b32: D = S1 shifted by S0[4:0].
    {F::VOP2, 0x16, build_vector_lshrrev_b32},
    {F::VOP2, 0x18, build_vector_ashrrev_i32},
    {F::VOP2, 0x1a, build_vector_lshlrev_b32},
    {F::VOP2, 0x1b, build_vector_and_b32},   // v_and_b32
    {F::VOP2, 0x1c, build_vector_or_b32},   // v_or_b32
    {F::VOP2, 0x1d, build_vector_xor_b32},   // v_xor_b32
    {F::VOP2, 0x1e, build_vector_xnor_b32},
    {F::VOP2, 0x25, build_vector_add_nc_u32},   // v_add_nc_u32
    {F::VOP2, 0x26, build_vector_sub_nc_u32},   // v_sub_nc_u32
    {F::VOP2, 0x27, build_vector_subrev_nc_u32},   // v_subrev_nc_u32
};

struct BuiltOp {
    Block block;
    Value d;
};

BuiltOp build(const VectorOpDef& def) {
    BuiltOp built;
    Builder ir(built.block);
    In in;
    in.s0 = ir.input(Type::U32);
    in.s1 = ir.input(Type::U32);
    built.d = def.build(ir, in);
    return built;
}

}   // namespace

std::span<const VectorOpDef> vector_alu_defs() { return kDefs; }

const VectorOpDef* find_vector_op(Rdna2Format fmt, uint32_t opcode) {
    for (const VectorOpDef& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}

Known eval_vector_op(const VectorOpDef& def, Known s0, Known s1) {
    const BuiltOp built = build(def);
    const Known inputs[] = {s0, s1};
    return evaluate(built.block, inputs)[built.d.index];
}

uint32_t emit_vector_op(SpirvCompute& b, const VectorOpDef& def, uint32_t s0, uint32_t s1) {
    const BuiltOp built = build(def);
    const uint32_t inputs[] = {s0, s1};
    return emit_spirv(b, built.block, inputs)[built.d.index];
}

} // namespace prosper::gpu::ir
