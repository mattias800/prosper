// Pin the moved f16 unary admission and destination placement, independently of its math.
#include "gpu/recompiler/ir/lower/vop1_unary_f16_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <gtest/gtest.h>
#include <cstdint>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction() {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP1;
    in.opcode = 0x54;
    in.dst.value = 1;
    return in;
}
}   // namespace
TEST(IrLowerVop1UnaryF16, PlainRetainsTheOppositeHalfMask) {
    LoweringFixture t;
    auto in = instruction();
    const auto a = t.b.uconst(0x4000), old = t.b.uconst(0xabcd1234);
    t.rs.vreg[1] = old;
    auto& d = t.rs.vreg[1];
    const auto routed = ir::lower_vop1_unary_f16(t.b, t.rs, in, a, d, old, t.ok);
    EXPECT_EQ(routed, std::optional<bool>(true));
    EXPECT_TRUE(t.ok);
    EXPECT_NE(d, old);
    EXPECT_TRUE(t.constant_cached(0xffff0000));
}
TEST(IrLowerVop1UnaryF16, SdwaWordOneRetainsTheLowHalfMask) {
    LoweringFixture t;
    auto in = instruction();
    in.has_sdwa = true;
    in.sdwa_dst_sel = 5;
    in.sdwa_src0_sel = 5;
    const auto a = t.b.uconst(0x40000000), old = t.b.uconst(0xabcd1234);
    t.rs.vreg[1] = old;
    auto& d = t.rs.vreg[1];
    EXPECT_EQ(ir::lower_vop1_unary_f16(t.b, t.rs, in, a, d, old, t.ok), std::optional<bool>(true));
    EXPECT_TRUE(t.ok);
    EXPECT_TRUE(t.constant_cached(0x0000ffff));
    EXPECT_TRUE(t.constant_cached(16));
}
TEST(IrLowerVop1UnaryF16, UnsupportedDwordSdwaAndPlainModifiersFallBack) {
    for (bool sdwa : {false, true}) {
        SCOPED_TRACE(sdwa);
        LoweringFixture t;
        auto in = instruction();
        in.has_sdwa = sdwa;
        in.clamp = !sdwa;
        const auto old = t.b.uconst(7);
        t.rs.vreg[1] = old;
        auto& d = t.rs.vreg[1];
        const auto words = t.code_words();
        EXPECT_EQ(ir::lower_vop1_unary_f16(t.b, t.rs, in, old, d, old, t.ok), std::nullopt);
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(d, old);
        EXPECT_EQ(t.code_words(), words);
    }
}
TEST(IrLowerVop1UnaryF16, OtherFamiliesAndFormatsRemainUntouched) {
    for (auto fmt : {Rdna2Format::VOP1, Rdna2Format::VOP3}) {
        SCOPED_TRACE(int(fmt));
        LoweringFixture t;
        auto in = instruction();
        in.fmt = fmt;
        if (fmt == Rdna2Format::VOP1) in.opcode = 0x03;
        const auto old = t.b.uconst(7);
        t.rs.vreg[1] = old;
        auto& d = t.rs.vreg[1];
        const auto words = t.code_words();
        EXPECT_EQ(ir::lower_vop1_unary_f16(t.b, t.rs, in, old, d, old, t.ok), std::nullopt);
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(d, old);
        EXPECT_EQ(t.code_words(), words);
    }
}
