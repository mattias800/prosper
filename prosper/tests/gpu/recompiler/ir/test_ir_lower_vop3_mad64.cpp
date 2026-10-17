// MAD64 preserves both destination words and the existing mask writer's domain.
#include "gpu/recompiler/ir/lower/vop3_mad64_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <cstdint>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t opcode, uint32_t sdst = 106) {
    return LoweringFixture::decode(
        {0xd4000001u | (opcode << 16) | (sdst << 8), 258u | (259u << 9) | (261u << 18)});
}
void sources(LoweringFixture& t) {
    t.rs.vreg[1] = t.b.uconst(0xaabbccdd);
    t.rs.vreg[2] = t.b.uconst(0xffffffff);
    t.rs.vreg[3] = t.b.uconst(2);
    t.rs.vreg[5] = t.b.uconst(0xffffffff);
    t.rs.vreg[6] = t.b.uconst(0xffffffff);
}
}   // namespace
TEST(IrLowerVop3Mad64, BothWordsAndTheExistingWave32VccLoLifetimeArePublished) {
    for (uint32_t op : {0x176u, 0x177u})
        for (uint32_t wave : {32u, 64u}) {
            LoweringFixture t;
            sources(t);
            t.b.wave_size = wave;
            t.b.allow_b32_masks = true;
            const auto old_low = t.rs.vreg.at(1), old_high = t.rs.vreg.at(2);
            ASSERT_TRUE(ir::lower_vop3_mad64(t.b, t.rs, instruction(op), t.ok, false));
            ASSERT_TRUE(t.ok);
            EXPECT_NE(t.rs.vreg.at(1), old_low);
            EXPECT_NE(t.rs.vreg.at(2), old_high);
            EXPECT_NE(t.rs.vcc, 0u);
            EXPECT_EQ(t.rs.sreg_bool_b32.contains(106), wave == 32);
            if (wave == 32) {
                ASSERT_TRUE(t.rs.sreg_bool.contains(106));
                EXPECT_EQ(t.rs.sreg_bool.at(106), t.rs.vcc);
            }
        }
}
TEST(IrLowerVop3Mad64, ExistingSgprSdstRemainsAPairInEitherWave) {
    for (uint32_t wave : {32u, 64u}) {
        LoweringFixture t;
        sources(t);
        t.b.wave_size = wave;
        t.b.allow_b32_masks = true;
        const auto old_vcc = t.rs.vcc;
        ASSERT_TRUE(ir::lower_vop3_mad64(t.b, t.rs, instruction(0x176, 20), t.ok, false));
        ASSERT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.sreg_bool.contains(20));
        EXPECT_FALSE(t.rs.sreg_bool_b32.contains(20));
        EXPECT_EQ(t.rs.vcc, old_vcc);
    }
}
TEST(IrLowerVop3Mad64, DestinationPairOverflowRefusesBeforeReadingOrPublishing) {
    LoweringFixture t;
    sources(t);
    auto in = instruction(0x176);
    in.dst.value = 255;
    const auto old = t.rs.vreg;
    const auto before = t.code_words();
    EXPECT_TRUE(ir::lower_vop3_mad64(t.b, t.rs, in, t.ok, false));
    EXPECT_FALSE(t.ok);
    EXPECT_EQ(t.rs.vreg, old);
    EXPECT_EQ(t.code_words(), before);
}
TEST(IrLowerVop3Mad64, UnmatchedDefinitionsDoNothing) {
    LoweringFixture t;
    sources(t);
    auto in = instruction(0x176);
    in.opcode = 0x175;
    const auto old = t.rs.vreg;
    const auto before = t.code_words();
    EXPECT_FALSE(ir::lower_vop3_mad64(t.b, t.rs, in, t.ok, false));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.rs.vreg, old);
    EXPECT_EQ(t.code_words(), before);
}
