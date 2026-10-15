// scalar_ops_common.cpp — lookup, evaluation and emission shared by every ScalarOpDef.
#include "gpu/recompiler/ir/ops/scalar_alu.hpp"

#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"

namespace prosper::gpu::ir {

namespace {

struct BuiltOp {
    Block block;
    ScalarOpOut out;
};

BuiltOp build(const ScalarOpDef& def) {
    BuiltOp built;
    Builder ir(built.block);
    ScalarOpIn in;
    in.s0 = ir.input(Type::U32);
    in.s1 = ir.input(Type::U32);
    in.scc = ir.input(Type::Bool);
    built.out = def.build(ir, in);
    return built;
}

} // namespace

const ScalarOpDef* find_scalar_op(Rdna2Format fmt, uint32_t opcode) {
    for (auto defs : {scalar_alu_defs(), scalar_compare_defs()})
        for (const ScalarOpDef& def : defs)
            if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}

uint32_t scalar_op_simm16(const ScalarOpDef& def, const Rdna2Inst& in) {
    return def.operands == ScalarOperands::DstSimmZext
        ? static_cast<uint32_t>(static_cast<uint16_t>(in.simm16))
        : static_cast<uint32_t>(in.simm16);
}

ScalarOpValue eval_scalar_op(const ScalarOpDef& def, Known s0, Known s1, Known scc) {
    const BuiltOp built = build(def);
    const Known inputs[] = {s0, s1, scc};
    const std::vector<Known> values = evaluate(built.block, inputs);
    ScalarOpValue result;
    if (built.out.d.valid()) result.d = values[built.out.d.index];
    if (built.out.scc.valid()) result.scc = values[built.out.scc.index];
    return result;
}

ScalarOpIds emit_scalar_op(SpirvCompute& b, const ScalarOpDef& def, uint32_t s0, uint32_t s1,
                           uint32_t scc) {
    const BuiltOp built = build(def);
    const uint32_t inputs[] = {s0, s1, scc};
    const std::vector<uint32_t> ids = emit_spirv(b, built.block, inputs);
    ScalarOpIds result;
    if (built.out.d.valid()) result.d = ids[built.out.d.index];
    if (built.out.scc.valid()) result.scc = ids[built.out.scc.index];
    return result;
}

} // namespace prosper::gpu::ir
