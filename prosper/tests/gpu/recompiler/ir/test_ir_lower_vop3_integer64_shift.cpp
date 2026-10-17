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
    for (uint32_t op : {0x302u, 0x176u}) {
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

TEST(IrLowerVop3Integer64Shift, RequestedArithmeticRightKeepsCanonicalPairBoundaries) {
    for (uint32_t arm = 0; arm < 13; ++arm) {
        LoweringFixture t;
        sources(t);
        auto in = instruction(false);
        in.opcode = 0x301;
        in.words[0] = 0xd7010001;
        if (arm == 1) in.words[1] |= 1u << 18;
        if (arm == 2) in.has_modifier = true;
        if (arm == 3) in.has_dpp = true;
        if (arm == 4) in.has_sdwa = true;
        if (arm == 5) in.src[1].value = 255;
        if (arm == 6) in.src[1].kind = OperandKind::SGPR;
        if (arm == 7) in.dst.value = 255;
        if (arm == 8) in.src_abs[0] = true;
        if (arm == 9) in.src_neg[1] = true;
        if (arm == 10) in.clamp = true;
        if (arm == 11) in.omod = 1;
        if (arm == 12) in.words[0] |= 1u << 11;
        const auto old = t.rs.vreg;
        const auto before = t.code_words();
        ASSERT_TRUE(ir::lower_vop3_integer64_shift(t.b, t.rs, in, t.ok));
        EXPECT_EQ(t.ok, arm == 0);
        if (arm) {
            EXPECT_EQ(t.rs.vreg, old);
            EXPECT_EQ(t.code_words(), before);
        }
    }
}

TEST(IrLowerVop3Integer64Shift, UncachedOldHighWordIsAllocatedBetweenShiftAndUnpack) {
    for (bool left : {false, true}) {
        LoweringFixture t;
        sources(t);
        auto in = instruction(left);
        in.dst.value = 24; // neither destination word has a prior value
        t.b.uconst_cache.erase(
            0); // begin may have declared zero; force an uncached old_hi request
        ASSERT_FALSE(t.rs.vreg.contains(25));
        ASSERT_TRUE(ir::lower_vop3_integer64_shift(t.b, t.rs, in, t.ok));
        ASSERT_TRUE(t.ok);
        uint32_t shift = 0, low = 0, high = 0;
        for (size_t at = 0; at < t.b.code.size();) {
            const uint32_t size = t.b.code[at] >> 16, op = t.b.code[at] & 0xffff;
            ASSERT_GT(size, 0u);
            if (!low && (op == Op_ShiftLeftLogical || op == Op_ShiftRightLogical))
                shift = t.b.code[at + 2];
            if (op == Op_UConvert && shift && t.b.code[at + 3] == shift) {
                if (!low) low = t.b.code[at + 2];
            }
            if (op == Op_UConvert && low && t.b.code[at + 2] > low) high = t.b.code[at + 2];
            at += size;
        }
        ASSERT_TRUE(t.b.uconst_cache.contains(0));
        const uint32_t old_high_zero = t.b.uconst_cache.at(0);
        if (left)
            EXPECT_LT(old_high_zero, shift); // pack(1,0) already requested zero
        else
            EXPECT_LT(shift, old_high_zero); // pair data did not need a zero constant
        EXPECT_LT(old_high_zero, low);
        EXPECT_LT(low, high);
    }
}
