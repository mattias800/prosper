// DS shuffle relocation must keep stage/native guards, destination predication and mask state.
// In particular, a native-only BPERMUTE must not absorb the portable dispatcher's event route.
#include "gpu/recompiler/ir/lower/ds_cross_lane_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using prosper::test::ir::LoweringFixture;

namespace {
Rdna2Inst bpermute() {
    return LoweringFixture::decode({0xdacc0000u, 0x01000302u});
}
Rdna2Inst swizzle(uint32_t offset) {
    return LoweringFixture::decode({0xd8d40000u | offset, 0x01000002u});
}

TEST(IrLowerDsCrossLane, OtherFormatsAndDsOperationsRemainWithTheirOwner) {
    LoweringFixture t;
    for (auto in : {LoweringFixture::decode({0xd8d80000u, 0x01000002u}), bpermute()}) {
        if (in.opcode == bpermute().opcode) in.fmt = Rdna2Format::VOP2;
        const auto before = t.b.code.size();
        EXPECT_EQ(lower_ds_cross_lane(t.b, t.rs, in, t.ok), std::nullopt);
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(t.b.code.size(), before);
        EXPECT_TRUE(t.rs.vreg.empty());
    }
}

TEST(IrLowerDsCrossLane, BpermuteRequiresAnExactNativeComputeWave) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t arm = 0; arm < 4; ++arm) {
            LoweringFixture t;
            t.b.wave_size = wave;
            t.b.native_subgroup_size = arm == 0 ? 0 : arm == 1 ? 96u - wave : wave;
            if (arm == 2) t.b.is_compute = false;
            auto in = bpermute();
            if (arm == 3) in.ds_gds = true;
            const auto before = t.b.code.size();
            EXPECT_EQ(lower_ds_cross_lane(t.b, t.rs, in, t.ok), std::optional<bool>{true});
            EXPECT_FALSE(t.ok);
            EXPECT_EQ(t.b.code.size(), before);
            EXPECT_TRUE(t.rs.vreg.empty());
        }
}

TEST(IrLowerDsCrossLane, NativeBpermutePreservesMasksAndExistingOtherRegisters) {
    for (uint32_t wave : {32u, 64u}) {
        LoweringFixture t;
        t.b.wave_size = t.b.native_subgroup_size = wave;
        t.rs.exec = t.b.bfalse();
        t.rs.exec_narrowed = true;
        t.rs.vcc = t.b.btrue();
        t.rs.scc = t.b.bfalse();
        t.rs.vreg[1] = t.b.uconst(0xcafebabe);
        t.rs.vreg[2] = t.b.uconst(4);
        t.rs.vreg[3] = t.b.uconst(0x12345678);
        const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
        const auto address = t.rs.vreg.at(2), source = t.rs.vreg.at(3);
        EXPECT_EQ(lower_ds_cross_lane(t.b, t.rs, bpermute(), t.ok), std::optional<bool>{true});
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.contains(1));
        EXPECT_EQ(t.rs.vreg.at(2), address);
        EXPECT_EQ(t.rs.vreg.at(3), source);
        EXPECT_EQ(t.rs.exec, exec);
        EXPECT_EQ(t.rs.vcc, vcc);
        EXPECT_EQ(t.rs.scc, scc);
        EXPECT_TRUE(t.rs.exec_narrowed);
    }
}

TEST(IrLowerDsCrossLane, MissingPhysicalWordsKeepTheExistingZeroFallback) {
    for (auto in : {bpermute(), swizzle(0x80b1u)}) {
        LoweringFixture t;
        t.b.native_subgroup_size = 64;
        ASSERT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(lower_ds_cross_lane(t.b, t.rs, in, t.ok), std::optional<bool>{true});
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.contains(1));
        EXPECT_FALSE(t.rs.vreg.contains(2));
        EXPECT_FALSE(t.rs.vreg.contains(3));
    }
}

TEST(IrLowerDsCrossLane, SwizzleRetainsComputeAndFragmentStageAdmission) {
    for (uint32_t stage = 0; stage < 3; ++stage) {
        LoweringFixture t;
        t.b.is_compute = stage == 0;
        t.b.is_fragment = stage == 1;
        t.rs.vreg[2] = t.b.uconst(0x11223344);
        const auto before = t.b.code.size();
        EXPECT_EQ(lower_ds_cross_lane(t.b, t.rs, swizzle(0x020fu), t.ok),
                  std::optional<bool>{true});
        EXPECT_EQ(t.ok, stage != 2);
        EXPECT_EQ(t.rs.vreg.contains(1), stage != 2);
        if (stage == 2) EXPECT_EQ(t.b.code.size(), before);
    }
}

TEST(IrLowerDsCrossLane, RotateAndFftSwizzlesRemainRefusedBeforePublication) {
    for (uint32_t offset : {0xc000u, 0xe000u, 0xffffu}) {
        LoweringFixture t;
        t.rs.vreg[1] = t.b.uconst(0xabcdef01);
        const auto old = t.rs.vreg.at(1);
        const auto before = t.b.code.size();
        EXPECT_EQ(lower_ds_cross_lane(t.b, t.rs, swizzle(offset), t.ok), std::optional<bool>{true});
        EXPECT_FALSE(t.ok);
        EXPECT_EQ(t.rs.vreg.at(1), old);
        EXPECT_EQ(t.b.code.size(), before);
    }
}
TEST(IrLowerDsCrossLane, ComputeSubgroupContractsRetainTheirOriginalWidths) {
    for (uint32_t control : {0x80e4u, 0x020fu}) {
        LoweringFixture t;
        t.rs.vreg[2] = t.b.uconst(0x11223344);
        ASSERT_EQ(t.b.compute_min_subgroup_size, 0u);
        ASSERT_EQ(lower_ds_cross_lane(t.b, t.rs, swizzle(control), t.ok),
                  std::optional<bool>{true});
        ASSERT_TRUE(t.ok);
        EXPECT_EQ(t.b.compute_min_subgroup_size, control & 0x8000u ? 4u : 32u);
    }
}
TEST(IrLowerDsCrossLane, MissingSwizzleSourceIsResolvedAfterThePeerArithmetic) {
    LoweringFixture t;
    t.b.uconst_cache.erase(0);
    const auto before = t.b.code.size();
    // AND=15, OR=2, XOR=1: lane arithmetic itself does not intern zero.
    ASSERT_EQ(lower_ds_cross_lane(t.b, t.rs, swizzle(0x044fu), t.ok), std::optional<bool>{true});
    ASSERT_TRUE(t.ok);
    uint32_t peer = 0, shuffled_source = 0;
    for (size_t i = before; i < t.b.code.size();) {
        const auto words = t.b.code[i] >> 16;
        ASSERT_GT(words, 0u);
        if ((t.b.code[i] & 0xffffu) == Op_GroupNonUniformShuffle) {
            shuffled_source = t.b.code[i + 4];
            peer = t.b.code[i + 5];
            break;
        }
        i += words;
    }
    ASSERT_NE(peer, 0u);
    ASSERT_TRUE(t.constant_cached(0));
    EXPECT_EQ(shuffled_source, t.b.uconst_cache.at(0));
    EXPECT_GT(shuffled_source, peer);
}
} // namespace
