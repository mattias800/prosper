// Hand-derived SPIR-V unordered comparison and logical-OR truth tables, including unknown inputs.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "support/death_test.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <optional>
using namespace prosper::gpu::ir;
TEST(IrCoreFloatPredicates, BoolOrTruthTable) {
    for (uint32_t a : {0u, 1u})
        for (uint32_t b : {0u, 1u})
            EXPECT_EQ(evaluate_op(Opcode::BoolOr, 0, a, b, 0), Known(a | b));
}
TEST(IrCoreFloatPredicates, UnorderedNotEqualIncludesEitherNanAndEquatesSignedZeros) {
    struct X {
        uint32_t a, b, want;
    };
    for (auto x : {X{0, 0x80000000, 0}, X{0x3f800000, 0x3f800000, 0}, X{0x3f800000, 0x40000000, 1},
                   X{0x7f800000, 0x7f800000, 0}, X{0x7fc01234, 0x3f800000, 1},
                   X{0x3f800000, 0xffc12345, 1}, X{0x7fc01234, 0x7fc01234, 1}})
        EXPECT_EQ(evaluate_op(Opcode::F32UnordNe, 0, x.a, x.b, 0), Known(x.want));
}
TEST(IrCoreFloatPredicates, UnknownOperandsStayUnknown) {
    Block b;
    Builder ir(b);
    const auto a = ir.input(Type::U32), c = ir.input(Type::U32);
    const auto cmp = ir.f32unordne(a, c);
    const auto result = ir.boolor_(cmp, ir.cfalse());
    const std::array<Known, 2> in = {0x3f800000, std::nullopt};
    const auto v = evaluate(b, in);
    EXPECT_FALSE(v[cmp.index]);
    EXPECT_FALSE(v[result.index]);
}
TEST(IrCoreFloatPredicates, OperandTypesAreEnforced) {
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            Block b;
            Builder ir(b);
            auto word = ir.c32(1);
            ir.boolor_(ir.ctrue(), word);
        },
        "argument type mismatch");
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            Block b;
            Builder ir(b);
            auto flag = ir.ctrue();
            ir.f32unordne(flag, ir.c32(0));
        },
        "argument type mismatch");
}
