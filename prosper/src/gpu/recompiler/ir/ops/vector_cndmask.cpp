// vector_cndmask.cpp — v_cndmask_b32 (AMD 70648 6.5; VOP2 0x01, VOP3 0x101): D = mask ? S1 : S0.
#include "gpu/recompiler/ir/ops/vector_cndmask.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"

#include <vector>

namespace prosper::gpu::ir {

namespace {

Value cndmask(Builder& ir, Value mask, Value when_set, Value when_clear) {
    return ir.select(mask, when_set, when_clear);
}

constexpr VectorCndmaskDef kDefs[] = {
    {Rdna2Format::VOP2, 0x01, cndmask},
    {Rdna2Format::VOP3, 0x101, cndmask},
};

struct Built {
    Block block;
    Value d;
};
// Inputs: slot 0 = the mask bit (Bool), slot 1 = S1, slot 2 = S0.
Built build(const VectorCndmaskDef& def) {
    Built built;
    Builder ir(built.block);
    const Value mask = ir.input(Type::Bool);
    const Value s1 = ir.input(Type::U32);
    const Value s0 = ir.input(Type::U32);
    built.d = def.build(ir, mask, s1, s0);
    return built;
}

}   // namespace

std::span<const VectorCndmaskDef> vector_cndmask_defs() {
    return kDefs;
}

const VectorCndmaskDef* find_vector_cndmask(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}

Known eval_vector_cndmask(const VectorCndmaskDef& def, Known mask, Known s0, Known s1) {
    const Built built = build(def);
    const Known inputs[] = {mask, s1, s0};
    return evaluate(built.block, inputs)[built.d.index];
}

uint32_t emit_vector_cndmask(SpirvCompute& b, const VectorCndmaskDef& def, uint32_t mask,
                             uint32_t s1, uint32_t s0) {
    const Built built = build(def);
    const uint32_t inputs[] = {mask, s1, s0};
    return emit_spirv(b, built.block, inputs)[built.d.index];
}

}   // namespace prosper::gpu::ir
