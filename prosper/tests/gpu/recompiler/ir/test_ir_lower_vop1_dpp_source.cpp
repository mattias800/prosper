// VOP1 DPP row-XOR stays a native-compute SRC0 transform; caller state remains caller-owned.
#include "gpu/recompiler/ir/lower/vop1_dpp_source_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using prosper::test::ir::LoweringFixture;

namespace {
Rdna2Inst mov_dpp(uint32_t control) {
    const uint32_t control_word =
        2u | (control << 8u) | (1u << 19u) | (0xfu << 24u) | (0xfu << 28u);
    return LoweringFixture::decode({0x7e0802fau, control_word});
}

void expect_mov_facts(const Rdna2Inst& in) {
    const auto& effects = rdna2_instruction_effects(in);
    ASSERT_EQ(in.fmt, Rdna2Format::VOP1);
    ASSERT_EQ(in.opcode, 0x01u);
    ASSERT_TRUE(effects.describes_vector_result());
    EXPECT_STREQ(effects.name, "v_mov_b32");
    EXPECT_EQ(effects.lane_flow, Rdna2LaneFlow::LaneWise);
    EXPECT_EQ(effects.vdst_dwords, 1u);
}

uint32_t count_shuffle(const std::vector<uint32_t>& code, size_t begin) {
    uint32_t count = 0;
    for (size_t at = begin; at < code.size();) {
        const uint32_t words = code[at] >> 16u;
        if (!words || words > code.size() - at) return 0;
        count += (code[at] & 0xffffu) == Op_GroupNonUniformShuffle;
        at += words;
    }
    return count;
}

TEST(IrLowerVop1DppSource, NativeComputeRowRorAndXmaskUseTwoShufflesAndPreserveFlags) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t control : {0x128u, 0x160u, 0x164u, 0x16fu}) {
            LoweringFixture t;
            t.b.is_compute = true;
            t.b.wave_size = t.b.native_subgroup_size = wave;
            const Rdna2Inst in = mov_dpp(control);
            expect_mov_facts(in);
            ASSERT_TRUE(in.has_dpp);
            ASSERT_TRUE(in.dpp_bound_ctrl);
            ASSERT_EQ(in.dpp_ctrl, control);
            uint32_t source = t.b.load_input(0);
            const uint32_t input_source = source;
            t.rs.vreg[in.src[0].value] = source;
            t.rs.exec = t.b.btrue();
            t.rs.vcc = t.b.bfalse();
            t.rs.scc = t.b.btrue();
            const uint32_t exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
            const uint32_t dpp_active = t.b.bfalse();
            uint32_t active = dpp_active;
            const size_t before = t.b.code.size();

            EXPECT_EQ(lower_vop1_dpp_source(t.b, t.rs, in, t.ok, source, active), std::nullopt);
            EXPECT_TRUE(t.ok);
            EXPECT_EQ(t.b.compute_min_subgroup_size, 16u);
            EXPECT_EQ(count_shuffle(t.b.code, before), 2u);
            EXPECT_EQ(t.rs.exec, exec);
            EXPECT_EQ(t.rs.vcc, vcc);
            EXPECT_EQ(t.rs.scc, scc);
            EXPECT_EQ(active, dpp_active);
            EXPECT_EQ(t.rs.vreg.at(in.src[0].value), input_source);
        }
}

TEST(IrLowerVop1DppSource, PortableAndNonComputeRowXorRemainRefusedBeforeEmission) {
    for (uint32_t wave : {32u, 64u})
        for (bool compute : {false, true}) {
            LoweringFixture t;
            t.b.is_compute = compute;
            t.b.wave_size = wave;
            t.b.native_subgroup_size = compute ? 0u : wave;
            const Rdna2Inst in = mov_dpp(0x164u);
            expect_mov_facts(in);
            uint32_t source = t.b.load_input(0);
            const uint32_t input_source = source;
            t.rs.vreg[in.src[0].value] = source;
            t.rs.exec = t.b.btrue();
            t.rs.vcc = t.b.bfalse();
            t.rs.scc = t.b.btrue();
            const uint32_t exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
            const uint32_t active = t.b.bfalse();
            uint32_t dpp_active = active;
            const size_t before = t.b.code.size();

            EXPECT_EQ(lower_vop1_dpp_source(t.b, t.rs, in, t.ok, source, dpp_active),
                      std::optional<bool>{true});
            EXPECT_FALSE(t.ok);
            EXPECT_EQ(t.b.code.size(), before);
            EXPECT_EQ(t.b.compute_min_subgroup_size, 0u);
            EXPECT_EQ(source, input_source);
            EXPECT_EQ(dpp_active, active);
            EXPECT_EQ(t.rs.exec, exec);
            EXPECT_EQ(t.rs.vcc, vcc);
            EXPECT_EQ(t.rs.scc, scc);
        }
}

TEST(IrLowerVop1DppSource, NonDppMovIsAnUnclaimedNoOp) {
    LoweringFixture t;
    t.b.is_compute = true;
    t.b.wave_size = t.b.native_subgroup_size = 64;
    const Rdna2Inst in = LoweringFixture::decode({0x7e080280u});
    expect_mov_facts(in);
    ASSERT_FALSE(in.has_dpp);
    uint32_t source = t.b.load_input(0);
    uint32_t dpp_active = t.b.bfalse();
    const uint32_t before_source = source, before_active = dpp_active;
    const size_t before = t.b.code.size();

    EXPECT_EQ(lower_vop1_dpp_source(t.b, t.rs, in, t.ok, source, dpp_active), std::nullopt);
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.b.code.size(), before);
    EXPECT_EQ(t.b.compute_min_subgroup_size, 0u);
    EXPECT_EQ(source, before_source);
    EXPECT_EQ(dpp_active, before_active);
}
}  // namespace
