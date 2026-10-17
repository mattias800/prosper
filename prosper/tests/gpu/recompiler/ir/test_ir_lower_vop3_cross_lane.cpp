// The native dynamic WRITELANE value changes; all route decisions and side state stay owned here.
#include "gpu/recompiler/ir/lower/vop3_cross_lane_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/ops/vector_cross_lane.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t selector = 21u) {
    return LoweringFixture::decode({0xd7610001u, 20u | (selector << 9)});
}
void native(LoweringFixture& t, uint32_t wave) {
    t.b.wave_size = wave;
    t.b.native_subgroup_size = wave;
    t.set_sgpr(20, 0x5370abcd);
    t.set_sgpr(21, wave + 5u);
}
TEST(IrLowerVop3CrossLane, NativeDynamicWriteKeepsMasksAndRetiresOnlyTheOldLaneSlot) {
    const auto& effects = rdna2_instruction_effects(Rdna2Format::VOP3, 0x361);
    EXPECT_EQ(effects.lane_flow, Rdna2LaneFlow::WriteLane);
    EXPECT_EQ(rdna2_valu_source_count(Rdna2Format::VOP3, 0x361), 2u);
    for (uint32_t wave : {32u, 64u}) {
        LoweringFixture t;
        native(t, wave);
        t.rs.exec = t.b.bfalse();
        t.rs.exec_narrowed = true;
        t.rs.vcc = t.b.btrue();
        t.rs.scc = t.b.bfalse();
        t.rs.invalidated_vgpr_lane_slots.insert(1);
        t.rs.invalidated_vgpr_lane_slots.insert(7);
        const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
        const auto routed = lower_vop3_cross_lane(t.b, t.rs, instruction(), t.ok, true);
        ASSERT_TRUE(routed.has_value());
        EXPECT_TRUE(*routed);
        ASSERT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.contains(1));
        EXPECT_FALSE(t.rs.invalidated_vgpr_lane_slots.contains(1));
        EXPECT_TRUE(t.rs.invalidated_vgpr_lane_slots.contains(7));
        EXPECT_TRUE(t.rs.vgpr_lane_slots.empty());
        EXPECT_TRUE(t.rs.vgpr_lane_mask_slots.empty());
        EXPECT_EQ(t.rs.exec, exec);
        EXPECT_EQ(t.rs.vcc, vcc);
        EXPECT_EQ(t.rs.scc, scc);
        EXPECT_TRUE(t.rs.exec_narrowed);
    }
}
TEST(IrLowerVop3CrossLane, OldDestinationIsResolvedAfterTheLaneComparison) {
    LoweringFixture t;
    native(t, 64);
    t.b.uconst_cache.erase(0);
    const auto start = t.b.code.size();
    ASSERT_TRUE(lower_vop3_cross_lane(t.b, t.rs, instruction(), t.ok, true).has_value());
    ASSERT_TRUE(t.ok);
    size_t comparison = 0, select = 0;
    uint32_t compare_id = 0, select_id = 0;
    for (size_t i = start; i < t.b.code.size();) {
        const uint32_t count = t.b.code[i] >> 16;
        ASSERT_GT(count, 0u);
        if ((t.b.code[i] & 0xffffu) == Op_IEqual) {
            comparison = i;
            compare_id = t.b.code[i + 2];
        }
        if ((t.b.code[i] & 0xffffu) == Op_Select) {
            select = i;
            select_id = t.b.code[i + 2];
        }
        i += count;
    }
    EXPECT_GT(select, comparison);
    ASSERT_TRUE(t.constant_cached(0));
    EXPECT_GT(t.b.uconst_cache.at(0), compare_id);
    EXPECT_LT(t.b.uconst_cache.at(0), select_id);
}
TEST(IrLowerVop3CrossLane, VectorSelectorAndExistingDataOrMaskSlotStillRefuse) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t arm = 0; arm < 3; ++arm) {
            LoweringFixture t;
            native(t, wave);
            t.rs.vreg[1] = t.b.uconst(0x12345678);
            t.rs.vreg[2] = t.b.uconst(5);
            if (arm == 1) t.rs.vgpr_lane_slots[1][2] = t.b.uconst(7);
            if (arm == 2) t.rs.vgpr_lane_mask_slots[1][2] = t.b.btrue();
            const auto old = t.rs.vreg.at(1);
            const auto before = t.b.code.size();
            ASSERT_TRUE(lower_vop3_cross_lane(t.b, t.rs, instruction(arm ? 21u : 258u), t.ok, true)
                            .has_value());
            EXPECT_FALSE(t.ok);
            EXPECT_EQ(t.rs.vreg.at(1), old);
            EXPECT_EQ(t.b.code.size(), before);
        }
}
TEST(IrLowerVop3CrossLane, DynamicSelectorStillNeedsAnExactNativeComputeWave) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t native_width : {0u, wave == 32 ? 64u : 32u}) {
            LoweringFixture t;
            native(t, wave);
            t.b.native_subgroup_size = native_width;
            const auto before = t.b.code.size();
            ASSERT_TRUE(lower_vop3_cross_lane(t.b, t.rs, instruction(), t.ok, true).has_value());
            EXPECT_FALSE(t.ok);
            EXPECT_EQ(t.b.code.size(), before);
            EXPECT_TRUE(t.rs.vreg.empty());
        }
}
TEST(IrLowerVop3CrossLane, InlineLaneSlotPathAndForeignOpcodeDoNotUseTheNewValue) {
    LoweringFixture t;
    native(t, 64);
    ASSERT_TRUE(lower_vop3_cross_lane(t.b, t.rs, instruction(133), t.ok, true).has_value());
    ASSERT_TRUE(t.ok);
    EXPECT_EQ(t.rs.vgpr_lane_slots.at(1).at(5), t.rs.sreg.at(20));
    EXPECT_FALSE(t.rs.vreg.contains(1));
    auto foreign = instruction();
    foreign.opcode = 0x362;
    const auto before = t.b.code.size();
    EXPECT_FALSE(lower_vop3_cross_lane(t.b, t.rs, foreign, t.ok, true).has_value());
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.b.code.size(), before);
}
} // namespace
