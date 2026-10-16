// SPIR-V ordered equality must distinguish NaN from itself while matching both zero signs.
// Incorrect zero detection can publish a NaN instead of the DX9 zero result.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "support/death_test.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <optional>
using namespace prosper::gpu::ir;
TEST(IrCoreLegacyFloat, OrderedEqualityHasBooleanMetadata) {
    const auto& info = opcode_info(Opcode::F32Eq);
    EXPECT_EQ(info.arity, 2u);
    EXPECT_EQ(info.result, Type::Bool);
    Block b;
    Builder ir(b);
    auto a = ir.input(Type::U32), c = ir.input(Type::U32);
    auto result = ir.f32eq(a, c);
    EXPECT_EQ(b.at(result).type, Type::Bool);
}
TEST(IrCoreLegacyFloat, HandWorkedOrderedEqualityIncludesZeroSignsAndRejectsNan) {
 // Ordered equality: +/-0 equal; NaNs are unordered even with identical payloads.
    const std::array<std::array<uint32_t, 3>, 9> rows = {{{0, 0x80000000, 1},
                                                          {0x3f800000, 0x3f800000, 1},
                                                          {0x3f800000, 0x3f800001, 0},
                                                          {0x7fc01234, 0x7fc01234, 0},
                                                          {0x7fa00001, 0, 0},
                                                          {0x7f800000, 0x7f800000, 1},
                                                          {0x7f800000, 0xff800000, 0},
                                                          {1, 0, 0},
                                                          {0xbf800000, 0xbf800000, 1}}};
    for (const auto& row : rows)
        EXPECT_EQ(evaluate_op(Opcode::F32Eq, 0, row[0], row[1], 0), Known(row[2]));
}
TEST(IrCoreLegacyFloat, AnUnknownOperandDoesNotBecomeAComparisonResult) {
    Block b;
    Builder ir(b);
    auto a = ir.input(Type::U32), c = ir.input(Type::U32);
    auto result = ir.f32eq(a, c);
    const std::array<Known, 2> inputs = {0, std::nullopt};
    EXPECT_FALSE(evaluate(b, inputs)[result.index]);
}
TEST(IrCoreLegacyFloat, BooleanOperandsAreRejectedAsFloatWords) {
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            Block b;
            Builder ir(b);
            ir.f32eq(ir.ctrue(), ir.c32(0));
        },
        "argument type");
}
