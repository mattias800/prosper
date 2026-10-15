// The moved value stage owns only F32 unary operations, without changing its caller's
// register bookkeeping, source transforms or EXEC-predicated write.
#include "gpu/recompiler/ir/lower/vop1_unary_float_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <cstdint>
using namespace prosper::gpu;
using namespace prosper::test::ir;
TEST(IrLowerVop1UnaryFloat, FractEmitsTheExistingBitcastsAndExtendedInstruction) {
    LoweringFixture t;
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP1;
    in.opcode = 0x20;
    const auto source = t.b.uconst(0x3fa00000), old = t.b.uconst(7);
    auto d = old;
    const auto words = t.code_words();
    EXPECT_TRUE(ir::lower_vop1_unary_float(t.b, in, source, d));
    EXPECT_NE(d, old);
    EXPECT_EQ(t.code_words() - words, 14u);
}
TEST(IrLowerVop1UnaryFloat, TrigRetainsItsRevolutionsConstant) {
    LoweringFixture t;
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP1;
    in.opcode = 0x35;
    const auto source = t.b.uconst(0x3e800000);
    auto d = source;
    EXPECT_TRUE(ir::lower_vop1_unary_float(t.b, in, source, d));
    EXPECT_TRUE(t.constant_cached(0x40c90fdb));
    EXPECT_FALSE(t.constant_cached(0x3f800000));
}
TEST(IrLowerVop1UnaryFloat, OtherFamiliesAndReservedFormsRemainUntouched) {
    for (auto fmt : {Rdna2Format::VOP1, Rdna2Format::VOP3}) {
        LoweringFixture t;
        Rdna2Inst in{};
        in.fmt = fmt;
        in.opcode = fmt == Rdna2Format::VOP1 ? 0x02 : 0x20;
        const auto old = t.b.uconst(7);
        auto d = old;
        const auto words = t.code_words();
        EXPECT_FALSE(ir::lower_vop1_unary_float(t.b, in, old, d));
        EXPECT_EQ(d, old);
        EXPECT_EQ(t.code_words(), words);
    }
}
