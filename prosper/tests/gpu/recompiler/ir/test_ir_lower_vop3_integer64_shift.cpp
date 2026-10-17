// The relocated integer64 shifts keep exact source/modifier admission and both destination words.
#include "gpu/recompiler/ir/lower/vop3_integer64_shift_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(bool left) {
    return LoweringFixture::decode(
        {left ? 0xd6ff0001u : 0xd7000001u, left ? 0x00010302u : 0x00020b02u});
}
void sources(LoweringFixture& t) {
    t.rs.vreg[1] = t.b.uconst(0xabc00123);
    t.rs.vreg[2] = t.b.uconst(33);
    t.rs.vreg[5] = t.b.uconst(0x89abcdef);
    t.rs.vreg[6] = t.b.uconst(0x01234567);
}
} // namespace
TEST(IrLowerVop3Integer64Shift, ExistingFormsWriteBothDestinationWords) {
    for (bool left : {false, true}) {
        LoweringFixture t;
        sources(t);
        const auto in = instruction(left);
        const auto old = t.rs.vreg[1];
        ASSERT_TRUE(ir::lower_vop3_integer64_shift(t.b, t.rs, in, t.ok));
        EXPECT_TRUE(t.ok);
        EXPECT_NE(t.rs.vreg[1], old);
        EXPECT_TRUE(t.rs.vreg.contains(2));
        EXPECT_TRUE(t.constant_cached(63));
    }
}
TEST(IrLowerVop3Integer64Shift, LshlKeepsItsExactInlineOneAdmission) {
    for (int value : {0, 2, -1}) {
        LoweringFixture t;
        sources(t);
        auto in = instruction(true);
        in.src[1].value = value;
        const auto old = t.rs.vreg;
        const auto before = t.code_words();
        EXPECT_TRUE(ir::lower_vop3_integer64_shift(t.b, t.rs, in, t.ok));
        EXPECT_FALSE(t.ok);
        EXPECT_EQ(t.rs.vreg, old);
        EXPECT_EQ(t.code_words(), before);
    }
}
TEST(IrLowerVop3Integer64Shift, LshrKeepsItsConsecutiveVgprPairAdmission) {
    for (auto kind : {OperandKind::SGPR, OperandKind::InlineInt, OperandKind::InlineFloat}) {
        LoweringFixture t;
        sources(t);
        auto in = instruction(false);
        in.src[1].kind = kind;
        const auto old = t.rs.vreg;
        EXPECT_TRUE(ir::lower_vop3_integer64_shift(t.b, t.rs, in, t.ok));
        EXPECT_FALSE(t.ok);
        EXPECT_EQ(t.rs.vreg, old);
    }
}
TEST(IrLowerVop3Integer64Shift, ModifiersAndPairOverflowKeepOldDestinations) {
    for (bool left : {false, true})
        for (uint32_t arm = 0; arm < 10; ++arm) {
            LoweringFixture t;
            sources(t);
            auto in = instruction(left);
            if (arm < 3)
                in.src_abs[arm] = true;
            else if (arm < 6)
                in.src_neg[arm - 3] = true;
            else if (arm == 6)
                in.clamp = true;
            else if (arm == 7)
                in.omod = 1;
            else if (arm == 8)
                in.words[0] |= 1u << 11;
            else
                in.dst.value = 255;
            const auto old = t.rs.vreg;
            const auto before = t.code_words();
            EXPECT_TRUE(ir::lower_vop3_integer64_shift(t.b, t.rs, in, t.ok));
            EXPECT_FALSE(t.ok);
            EXPECT_EQ(t.rs.vreg, old);
            EXPECT_EQ(t.code_words(), before);
        }
}
TEST(IrLowerVop3Integer64Shift, UnmatchedOpcodesDoNothing) {
    for (uint32_t op : {0x301u, 0x302u, 0x176u}) {
        LoweringFixture t;
        sources(t);
        auto in = instruction(false);
        in.opcode = op;
        const auto old = t.rs.vreg;
        const auto before = t.code_words();
        EXPECT_FALSE(ir::lower_vop3_integer64_shift(t.b, t.rs, in, t.ok));
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(t.rs.vreg, old);
        EXPECT_EQ(t.code_words(), before);
    }
}
