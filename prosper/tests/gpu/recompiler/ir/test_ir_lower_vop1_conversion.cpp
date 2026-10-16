// Pin the conversion routes' placement, refusals and ownership independently of numeric eval.
#include "gpu/recompiler/ir/lower/vop1_conversion_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include <gtest/gtest.h>
#include <optional>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t opcode) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP1;
    in.opcode = opcode;
    in.dst.value = 1;
    return in;
}
}   // namespace
TEST(IrLowerVop1Conversion, NibbleConversionRetainsItsSignedWidthAndScale) {
    LoweringFixture t;
    auto in = instruction(0x0e);
    const auto a = t.b.uconst(8), old = t.b.uconst(0xabcd1234);
    auto d = old;
    EXPECT_TRUE(ir::lower_vop1_conversion_values(t.b, t.rs, in, a, d, old, t.ok));
    EXPECT_TRUE(t.ok);
    EXPECT_NE(d, old);
    EXPECT_TRUE(t.constant_cached(4));
    EXPECT_TRUE(t.constant_cached(0x3d800000)); // 1/16
}
TEST(IrLowerVop1Conversion, HalfIntegerWordOnePreservesTheLowDestinationHalf) {
    LoweringFixture t;
    auto in = instruction(0x50);
    in.has_sdwa = true;
    in.sdwa_src0_sel = in.sdwa_dst_sel = 5;
    const auto a = t.b.uconst(0x12340000), old = t.b.uconst(0xabcd5678);
    t.rs.vreg[1] = old;
    auto& d = t.rs.vreg[1];
    EXPECT_EQ(ir::lower_vop1_conversion_routes(t.b, t.rs, in, a, d, old, t.ok), true);
    EXPECT_TRUE(t.ok);
    EXPECT_NE(d, old);
    EXPECT_TRUE(t.constant_cached(0xffff));
    EXPECT_TRUE(t.constant_cached(16));
}
TEST(IrLowerVop1Conversion, SdwaByteInsertionPreservesAllOtherBytes) {
    LoweringFixture t;
    auto in = instruction(0x07);
    in.has_sdwa = true;
    in.sdwa_dst_sel = 2;
    const auto a = t.b.uconst(0x3f800000), old = t.b.uconst(0xabcd5678);
    t.rs.vreg[1] = old;
    auto& d = t.rs.vreg[1];
    EXPECT_EQ(ir::lower_vop1_conversion_routes(t.b, t.rs, in, a, d, old, t.ok), true);
    EXPECT_TRUE(t.ok);
    EXPECT_TRUE(t.constant_cached(0xff));
    EXPECT_TRUE(t.constant_cached(0xff00ffff));
    EXPECT_TRUE(t.constant_cached(16));
}
TEST(IrLowerVop1Conversion, RpiUnsupportedModifiersClearOkWithoutComputing) {
    for (unsigned mode = 0; mode < 6; ++mode) {
        LoweringFixture t;
        auto in = instruction(0x0c);
        switch (mode) {
            case 0: in.has_sdwa = true; break;
            case 1: in.has_dpp = true; break;
            case 2: in.src_abs[0] = true; break;
            case 3: in.src_neg[0] = true; break;
            case 4: in.clamp = true; break;
            default: in.omod = 1; break;
        }
        const auto old = t.b.uconst(0x3f800000);
        auto d = old;
        const auto before = t.code_words();
        EXPECT_TRUE(ir::lower_vop1_conversion_values(t.b, t.rs, in, old, d, old, t.ok));
        EXPECT_FALSE(t.ok);
        EXPECT_EQ(d, old);
        EXPECT_EQ(t.code_words(), before);
    }
}
TEST(IrLowerVop1Conversion, OtherOpcodesRemainWithTheCaller) {
    for (uint32_t op : {0x00u, 0x01u, 0x43u, 0x54u, 0xffu}) {
        LoweringFixture t;
        auto in = instruction(op);
        const auto old = t.b.uconst(7);
        auto d = old;
        const auto before = t.code_words();
        EXPECT_EQ(ir::lower_vop1_conversion_routes(t.b, t.rs, in, old, d, old, t.ok), std::nullopt);
        EXPECT_FALSE(ir::lower_vop1_conversion_values(t.b, t.rs, in, old, d, old, t.ok));
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(d, old);
        EXPECT_EQ(t.code_words(), before);
    }
}
