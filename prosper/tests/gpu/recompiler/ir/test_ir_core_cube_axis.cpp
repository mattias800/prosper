// New numeric IR primitives pin ordered comparisons, signed-zero absolute value and unknowns.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "support/death_test.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <optional>
using namespace prosper::gpu::ir;
TEST(IrCoreCubeAxis, AbsoluteValueHasHandDerivedRawBits) {
    const std::array<std::array<uint32_t, 2>, 7> cases = {{{0, 0},
                                                           {0x80000000, 0},
                                                           {0xbf800000, 0x3f800000},
                                                           {0x80000001, 1},
                                                           {0xff7fffff, 0x7f7fffff},
                                                           {0xff800000, 0x7f800000},
                                                           {0x7f800000, 0x7f800000}}};
    for (auto x : cases) EXPECT_EQ(evaluate_op(Opcode::F32Abs, 0, x[0], 0, 0), Known(x[1]));
    EXPECT_FALSE(evaluate_op(Opcode::F32Abs, 0, 0x7fc01234, 0, 0));
    EXPECT_FALSE(evaluate_op(Opcode::F32Abs, 0, 0xff800001, 0, 0));
    EXPECT_EQ(opcode_info(Opcode::F32Abs).arity, 1u);
    EXPECT_EQ(opcode_info(Opcode::F32Abs).result, Type::U32);
}
TEST(IrCoreCubeAxis, OrderedLessThanUsesFloatOrderAndRejectsNaNs) {
    const std::array<std::array<uint32_t, 3>, 8> cases = {{{0xbf800000, 0, 1},
                                                           {0, 0xbf800000, 0},
                                                           {0x80000000, 0, 0},
                                                           {0, 0x80000000, 0},
                                                           {0x3f800000, 0x40000000, 1},
                                                           {0xff800000, 0x7f800000, 1},
                                                           {0x7fc01234, 0, 0},
                                                           {0, 0xff800001, 0}}};
    for (auto x : cases) EXPECT_EQ(evaluate_op(Opcode::F32Lt, 0, x[0], x[1], 0), Known(x[2]));
    EXPECT_EQ(opcode_info(Opcode::F32Lt).arity, 2u);
    EXPECT_EQ(opcode_info(Opcode::F32Lt).result, Type::Bool);
}
TEST(IrCoreCubeAxis, UnknownsRemainUnknownThroughBothPrimitives) {
    Block b;
    Builder ir(b);
    const auto input = ir.input(Type::U32);
    const auto abs = ir.f32abs(input), lt = ir.f32lt(input, ir.c32(0));
    const Known sources[] = {std::nullopt};
    const auto result = evaluate(b, sources);
    EXPECT_FALSE(result[abs.index]);
    EXPECT_FALSE(result[lt.index]);
}
TEST(IrCoreCubeAxis, NumericArgumentsCannotBeWaveBooleans) {
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            Block b;
            Builder ir(b);
            ir.f32abs(ir.ctrue());
        },
        "argument type mismatch");
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            Block b;
            Builder ir(b);
            ir.f32lt(ir.c32(0), ir.ctrue());
        },
        "argument type mismatch");
}
