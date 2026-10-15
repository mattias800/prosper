// The IR core (gpu/recompiler/ir): Block construction rules, constant evaluation, and SPIR-V
// emission order. Evaluation-vs-SPIR-V agreement per opcode is test_ir_scalar_alu_execution's job.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;

TEST(IrCore, MalformedBlocksStopLoudly) {
    EXPECT_DEATH({ Block b; b.append({Opcode::IAdd, Type::U32, 0, {}}); }, "undefined argument");
    EXPECT_DEATH(
        {
            Block b;
            Builder ir(b);
            const Value flag = ir.ctrue();
            ir.iadd(flag, flag);
        },
        "argument type mismatch");
    EXPECT_DEATH({ Block b; b.append({Opcode::IEq, Type::U32, 0, {}}); }, "wrong result type");
    EXPECT_DEATH({ Block b; b.append({Opcode::Input, Type::U32, 1, {}}); }, "numbered in order");
}

TEST(IrCore, UnknownInputsPropagateAndKnownOnesFold) {
    Block block;
    Builder ir(block);
    const Value a = ir.input(Type::U32), b = ir.input(Type::U32);
    const Value sum = ir.iadd(a, ir.c32(1));
    const Value product = ir.imul(a, b);
    const Known inputs[] = {Known(41u), std::nullopt};
    const auto values = evaluate(block, inputs);
    EXPECT_EQ(values[sum.index], 42u);
    EXPECT_FALSE(values[product.index]);
}

// Hand-derived from the SPIR-V and GLSL.std.450 definitions, not from eval.cpp.
TEST(IrCore, EvaluationEdges) {
    EXPECT_FALSE(evaluate_op(Opcode::Shl, 0, 1, 32, 0)) << "a shift by 32 is undefined";
    EXPECT_EQ(evaluate_op(Opcode::Shl, 0, 1, 31, 0), 0x80000000u);
    EXPECT_EQ(evaluate_op(Opcode::AShr, 0, 0x80000000u, 31, 0), 0xffffffffu);
    EXPECT_EQ(evaluate_op(Opcode::BfeU, 0, 0xabcd1234u, 8, 8), 0x12u);
    EXPECT_EQ(evaluate_op(Opcode::BfeU, 0, 0xabcd1234u, 28, 8), 0xau) << "count clamps to 32-offset";
    EXPECT_EQ(evaluate_op(Opcode::BfeU, 0, 0xabcd1234u, 40, 8), 0u) << "offset clamps to 32";
    EXPECT_EQ(evaluate_op(Opcode::BfeU, 0, 0xabcd1234u, 0, 32), 0xabcd1234u);
    EXPECT_EQ(evaluate_op(Opcode::SMulHi, 0, 0xffffffffu, 2, 0), 0xffffffffu);   // -1 * 2
    EXPECT_EQ(evaluate_op(Opcode::UMulHi, 0, 0xffffffffu, 2, 0), 1u);
    EXPECT_EQ(evaluate_op(Opcode::BitReverse, 0, 1, 0, 0), 0x80000000u);
    EXPECT_EQ(evaluate_op(Opcode::SLt, 0, 0xffffffffu, 0, 0), 1u);
    EXPECT_EQ(evaluate_op(Opcode::ULt, 0, 0xffffffffu, 0, 0), 0u);
    EXPECT_EQ(evaluate_op(Opcode::SMax, 0, 0xffffffffu, 1, 0), 1u);
    EXPECT_EQ(evaluate_op(Opcode::UMax, 0, 0xffffffffu, 1, 0), 0xffffffffu);
}

// Emission follows Block order and reuses SpirvCompute's own calls (the cached constant, a fresh
// result id per instruction) -- the property that keeps lowered SPIR-V byte-identical.
TEST(IrCore, EmissionFollowsBlockOrder) {
    SpirvCompute b;
    b.begin(16, nullptr, 64, 1, 1, 64, 0);
    const uint32_t seven = b.uconst(7);
    Block block;
    Builder ir(block);
    const Value in = ir.input(Type::U32);
    const Value c = ir.c32(7);
    const Value first = ir.isub(in, c);
    const Value second = ir.iadd(first, in);
    const Value flag = ir.ult(second, c);
    const size_t before = b.code.size();
    const uint32_t inputs[] = {seven};
    const auto ids = emit_spirv(b, block, inputs);
    EXPECT_EQ(ids[in.index], seven);
    EXPECT_EQ(ids[c.index], seven) << "a constant is SpirvCompute's cached one";
    EXPECT_LT(ids[first.index], ids[second.index]);
    EXPECT_LT(ids[second.index], ids[flag.index]);
    EXPECT_EQ(b.code.size() - before, 3u * 5u) << "three 5-word instructions";
}
