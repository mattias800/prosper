// Migration must retain source guards, native-row contracts, late inputs and caller-owned masks.
#include "gpu/recompiler/ir/permlane_cases.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/lower/vop3_cross_lane_lowering.hpp"
#include "gpu/recompiler/ir/ops/vector_permlane.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <gtest/gtest.h>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(bool across) {
    return LoweringFixture::decode(
        {across ? 0xd7780001u : 0xd7770001u, 258u | (20u << 9) | (21u << 18)});
}
void seed(LoweringFixture& t) {
    t.rs.vreg[2] = t.b.uconst(0x12345678);
    t.rs.vreg[1] = t.b.uconst(0x98765432);
    t.set_sgpr(20, 0x87654321);
    t.set_sgpr(21, 0x0fedcba9);
}
} // namespace
TEST(IrLowerVop3Permlane, ExistingCentralEffectsAndSubgroupContractsRemain) {
    for (bool across : {false, true}) {
        LoweringFixture t;
        seed(t);
        const auto in = instruction(across);
        const auto& e = rdna2_instruction_effects(in.fmt, in.opcode);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::CrossLane);
        EXPECT_EQ(rdna2_valu_source_count(in.fmt, in.opcode), 3u);
        EXPECT_EQ(e.vdst_dwords, 1u);
        ASSERT_EQ(lower_vop3_cross_lane(t.b, t.rs, in, t.ok, false), std::optional(true));
        ASSERT_TRUE(t.ok);
        EXPECT_EQ(compute_spirv_min_subgroup_size(t.b.finish()), across ? 32u : 16u);
    }
}
TEST(IrLowerVop3Permlane, RegisterKindsAndReservedModifiersStillRefuseBeforeReading) {
    for (bool across : {false, true})
        for (uint32_t arm = 0; arm < 14; ++arm) {
            LoweringFixture t;
            seed(t);
            auto in = instruction(across);
            if (arm == 0) in.src[0] = decode_src_field(128);
            if (arm == 1) in.src[1] = decode_src_field(258);
            if (arm == 2) in.src[2] = decode_src_field(258);
            if (arm == 3) in.words[0] |= 1u << 13;
            if (arm == 4) in.words[0] |= 1u << 14;
            if (arm == 5) in.clamp = true;
            if (arm == 6) in.omod = 1;
            if (arm == 7) in.src_abs[0] = true;
            if (arm == 8) in.src_abs[1] = true;
            if (arm == 9) in.src_abs[2] = true;
            if (arm == 10) in.src_neg[0] = true;
            if (arm == 11) in.src_neg[1] = true;
            if (arm == 12) in.src_neg[2] = true;
            if (arm == 13) {
                t.b.is_compute = false;
                t.b.is_fragment = false;
            }
            const auto code = t.b.code;
            const auto old = t.rs.vreg.at(1);
            EXPECT_EQ(lower_vop3_cross_lane(t.b, t.rs, in, t.ok, false), std::optional(true));
            EXPECT_FALSE(t.ok);
            EXPECT_EQ(t.b.code, code);
            EXPECT_EQ(t.rs.vreg.at(1), old);
        }
}
TEST(IrLowerVop3Permlane, UnavailableSavedSelectorRemainsARefusedDataRead) {
    for (bool across : {false, true})
        for (bool unavailable : {false, true}) {
            LoweringFixture t;
            seed(t);
            if (unavailable) t.rs.unavailable_saved_mask_words.insert(20);
            const auto in = instruction(across);
            ASSERT_EQ(lower_vop3_cross_lane(t.b, t.rs, in, t.ok, false), std::optional(true));
            EXPECT_EQ(t.ok, !unavailable);
            EXPECT_EQ(t.rs.unavailable_saved_mask_words.contains(20), unavailable);
        }
}
TEST(IrLowerVop3Permlane, ActiveAndOldInputsResolveAfterTheFirstAndSecondShuffle) {
    for (bool fetch : {false, true})
        for (bool bound : {false, true}) {
            LoweringFixture t;
            const auto value = t.b.uconst(37), lo = t.b.uconst(0), hi = t.b.uconst(UINT32_MAX);
            std::vector<std::string> order;
            const auto count_shuffles = [&] {
                size_t found = 0;
                for (size_t i = 0; i < t.b.code.size();) {
                    const auto n = t.b.code[i] >> 16;
                    if (!n || n > t.b.code.size() - i) break;
                    found += (t.b.code[i] & 0xffffu) == 345;
                    i += n;
                }
                return found;
            };
            emit_permlane_b32(
                t.b, value, lo, hi, false, fetch, bound,
                [&] {
                    order.push_back("active");
                    EXPECT_EQ(count_shuffles(), 1u);
                    return t.b.btrue();
                },
                [&] {
                    order.push_back("old");
                    EXPECT_EQ(count_shuffles(), 2u);
                    return t.b.uconst(91);
                });
            const std::vector<std::string> expected =
                fetch   ? std::vector<std::string>{}
                : bound ? std::vector<std::string>{"active"}
                        : std::vector<std::string>{"active", "old"};
            EXPECT_EQ(order, expected);
        }
}
TEST(IrLowerVop3Permlane, AllValueChoicesLeaveMaskOwnershipWithTheCaller) {
    for (bool fetch : {false, true})
        for (bool bound : {false, true}) {
            LoweringFixture t;
            seed(t);
            t.rs.exec = t.b.bfalse();
            t.rs.exec_narrowed = true;
            t.rs.vcc = t.b.btrue();
            t.rs.scc = t.b.bfalse();
            const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
            auto in = instruction(false);
            in.permlane_fetch_inactive = fetch;
            in.permlane_bound_ctrl = bound;
            ASSERT_EQ(lower_vop3_cross_lane(t.b, t.rs, in, t.ok, false), std::optional(true));
            EXPECT_TRUE(t.ok);
            EXPECT_EQ(t.rs.exec, exec);
            EXPECT_EQ(t.rs.vcc, vcc);
            EXPECT_EQ(t.rs.scc, scc);
            EXPECT_TRUE(t.rs.exec_narrowed);
        }
}
