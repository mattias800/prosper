// The IR core (gpu/recompiler/ir): Block construction rules, constant evaluation, and SPIR-V
// emission order. Evaluation-vs-SPIR-V agreement per opcode is test_ir_scalar_alu_execution's job.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>
#include "support/death_test.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

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

TEST(IrCore, InputResolverPreservesConstantsAndInterveningOperations) {
    SpirvCompute b;
    b.begin(16, nullptr, 64, 1, 1, 64, 0);
    Block block;
    Builder ir(block);
    const Value mask = ir.c32(0x13579bdf);
    const Value first = ir.input(Type::U32);
    const Value masked = ir.and_(first, mask);
    const Value second = ir.input(Type::U32);
    const Value d = ir.iadd(masked, second);
    std::vector<size_t> code_sizes;
    const auto ids = emit_spirv_resolving_inputs(b, block, [&](uint32_t slot) {
        EXPECT_TRUE(b.uconst_cache.contains(0x13579bdf));
        code_sizes.push_back(b.code.size());
        return b.uconst(slot == 0 ? 59 : 62);
    });
    ASSERT_EQ(code_sizes.size(), 2u);
    EXPECT_GT(code_sizes[1], code_sizes[0]);
    EXPECT_NE(ids[d.index], 0u);
}
TEST(IrCore, MissingInputResolverIsLoudOnlyWhenThereAreInputs) {
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            SpirvCompute b;
            Block block;
            Builder ir(block);
            ir.input(Type::U32);
            emit_spirv_resolving_inputs(b, block, {});
        },
        "input resolver missing");
    SpirvCompute b;
    b.begin(16, nullptr, 64, 1, 1, 64, 0);
    Block block;
    Builder ir(block);
    const Value d = ir.c32(19);
    const auto ids = emit_spirv_resolving_inputs(b, block, {});
    EXPECT_EQ(ids[d.index], b.uconst_cache.at(19));
}

// SPIR-V OpBitCount counts all 32 bits of a numeric word, independently of its signed appearance.
TEST(IrCore, NumericBitCountFactsEdgesAndUnknownInputs) {
    EXPECT_EQ(opcode_info(Opcode::BitCount).arity, 1u);
    EXPECT_EQ(opcode_info(Opcode::BitCount).result, Type::U32);
    Block block;
    Builder ir(block);
    const Value input = ir.input(Type::U32);
    const Value count = ir.bit_count(input);
    struct Example {
        uint32_t bits, want;
    };
    for (auto x : {Example{0, 0}, Example{1, 1}, Example{0x80000000, 1}, Example{0xaaaaaaaa, 16},
                   Example{0x7fffffff, 31}, Example{0xffffffff, 32}}) {
        const Known values[] = {Known(x.bits)};
        EXPECT_EQ(evaluate(block, values)[count.index], Known(x.want));
    }
    EXPECT_FALSE(evaluate(block, {})[count.index]);
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            Block b;
            Builder value(b);
            value.bit_count(value.ctrue());
        },
        "argument type mismatch");
}
