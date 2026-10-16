// Typed normalized16 primitive: hand-worked RNE conversions and invalid mode/type arms.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "support/death_test.hpp"
#include <gtest/gtest.h>
#include <array>
#include <optional>
using namespace prosper::gpu::ir;
TEST(IrCorePackNorm, HandWorkedHalfwayAndSaturationValues) {
 // UNORM half*65535=32767.5 rounds to32768; SNORM half*32767=16383.5 rounds to16384.
    EXPECT_EQ(evaluate_op(Opcode::PackNorm16, 0, 0x3f000000, 0, 0), Known(0x8000));
    EXPECT_EQ(evaluate_op(Opcode::PackNorm16, 1, 0x3f000000, 0, 0), Known(0x4000));
    EXPECT_EQ(evaluate_op(Opcode::PackNorm16, 1, 0xbf000000, 0, 0), Known(0xc000));
    EXPECT_EQ(evaluate_op(Opcode::PackNorm16, 1, 0xbf800000, 0, 0), Known(0x8001));
    EXPECT_EQ(evaluate_op(Opcode::PackNorm16, 0, 0xff800000, 0, 0), Known(0));
    EXPECT_EQ(evaluate_op(Opcode::PackNorm16, 0, 0x7f800000, 0, 0), Known(0xffff));
    EXPECT_FALSE(evaluate_op(Opcode::PackNorm16, 0, 0x7fc01234, 0, 0));
    EXPECT_FALSE(evaluate_op(Opcode::PackNorm16, 2, 0x3f800000, 0, 0));
}
TEST(IrCorePackNorm, UnknownInputsStayUnknown) {
    Block b;
    Builder ir(b);
    auto a = ir.input(Type::U32);
    auto u = ir.pack_norm16(a, false), s = ir.pack_norm16(a, true);
    const std::array<Known, 1> inputs = {std::nullopt};
    const auto values = evaluate(b, inputs);
    EXPECT_FALSE(values[u.index]);
    EXPECT_FALSE(values[s.index]);
}
TEST(IrCorePackNorm, InvalidModeAndOperandTypesStopLoudly) {
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            Block b;
            Builder ir(b);
            auto a = ir.c32(0);
            b.append({Opcode::PackNorm16, Type::U32, 2, {a}});
        },
        "mode out of range");
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            Block b;
            Builder ir(b);
            ir.pack_norm16(ir.ctrue(), false);
        },
        "argument type mismatch");
}
