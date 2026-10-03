// #4268: skipped unsigned DPP row maxima leave a compute producer's output unwritten.
#include "gpu/recompiler/rdna2_dpp_row_shr.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "fixtures/dpp_row_max.hpp"
#include <gtest/gtest.h>
#include <array>

using namespace prosper::gpu;

namespace {
uint32_t count(const std::vector<uint32_t>& words, uint32_t opcode,
               uint32_t extension = UINT32_MAX) {
    uint32_t result = 0;
    for (size_t i = 5; i < words.size();) {
        const uint32_t size = words[i] >> 16;
        if (!size || i + size > words.size()) return 0;
        result += (words[i] & 0xffffu) == opcode &&
                  (extension == UINT32_MAX || (size >= 6 && words[i + 4] == extension));
        i += size;
    }
    return result;
}

std::vector<uint32_t> compile(const std::vector<uint32_t>& code, bool cfg = false) {
    return recompile_valu(code.data(), code.size(), 3, 1, nullptr, 0, kDefaultComputePgmRsrc1, cfg);
}
}  // namespace

TEST(ComputeDppRowMax, RecognizesEveryLegalShiftAndRejectsUnownedForms) {
    for (uint32_t shift = 1; shift <= 15; ++shift) {
        const auto code = prosper::test::dpp_row_max_program({shift});
        const auto in = rdna2_decode_one(code.data(), code.size());
        ASSERT_TRUE(is_inplace_vmax_u32_dpp_row_shr(in)) << "shift " << shift;
        for (uint32_t mutation = 0; mutation < 12; ++mutation) {
            auto altered = in;
            switch (mutation) {
                case 0: altered.dst.value = 2; break;
                case 1: altered.src[0].value = 2; break;
                case 2: altered.src[1].value = 2; break;
                case 3: altered.dpp_bound_ctrl = true; break;
                case 4: altered.dpp_row_mask = 7; break;
                case 5: altered.dpp_bank_mask = 7; break;
                case 6: altered.has_modifier = true; break;
                case 7: altered.src_neg[0] = true; break;
                case 8: altered.src_abs[1] = true; break;
                case 9: altered.dpp_ctrl = 0x110; break;
                case 10: altered.dpp_ctrl = 0x120; break;
                case 11: altered.opcode = 0x13; break;
            }
            EXPECT_FALSE(is_inplace_vmax_u32_dpp_row_shr(altered))
                << "shift " << shift << ", mutation " << mutation;
        }
    }
}

TEST(ComputeDppRowMax, LinearLoweringUsesUnsignedMaxAndSourceExecShuffle) {
    const auto code = prosper::test::dpp_row_max_program({1, 2, 4, 8});
    const auto module = compile(code);
    ASSERT_FALSE(module.empty()) << "the complete synthetic ladder must compile";
    EXPECT_EQ(count(module, Op_ExtInst, Glsl_UMax), 4u);
    EXPECT_EQ(count(module, Op_GroupNonUniformShuffle), 8u)
        << "each static site shuffles the source value and its EXEC bit";
    EXPECT_EQ(count(module, Op_Switch), 0u) << "this checkpoint owns the linear route";
}

TEST(ComputeDppRowMax, RefusesUnresolvedSpillSourcesBeforeWriting) {
    const auto code = prosper::test::dpp_row_max_program({3});
    const auto in = rdna2_decode_one(code.data(), code.size());
    SpirvCompute b;
    b.begin(3);
    b.is_compute = true;
    RegState rs;
    rs.exec = b.bfalse();
    rs.exec_narrowed = true;
    rs.vreg[1] = b.uconst(0x80000001u);
    rs.vgpr_lane_slots[1][0] = b.uconst(77);
    rs.vgpr_lane_mask_slots[1][0] = b.btrue();
    bool ok = true;
    ASSERT_TRUE(emit_alu(b, rs, in, ok, true, nullptr, false, nullptr, true));
    EXPECT_FALSE(ok) << "the per-lane placeholder cannot stand for a scalar-spill VGPR";
    EXPECT_TRUE(rs.vgpr_lane_slots.contains(1));
    EXPECT_TRUE(rs.vgpr_lane_mask_slots.contains(1));
    EXPECT_FALSE(rs.invalidated_vgpr_lane_slots.contains(1));
    EXPECT_EQ(b.compute_min_subgroup_size, 0u) << "refusal emitted no cross-lane maximum";
}

TEST(ComputeDppRowMax, DispatcherRemainsRefusedUntilEventPhaseIsIntegrated) {
    const auto maximum = prosper::test::dpp_row_max_program({1, 2, 4, 8});
    const auto addition = prosper::test::dpp_row_max_program({1, 2, 4, 8}, false, true);
    EXPECT_TRUE(compile(maximum, true).empty())
        << "linear shuffles must never bypass the CFG static-event contract";
    const auto control = compile(addition, true);
    ASSERT_FALSE(control.empty())
        << "same CFG and sources are supported for the existing ADD phase";
    EXPECT_GT(count(control, Op_Switch), 0u);
}

TEST(ComputeDppRowMax, ProductionComputeCompilesDefaultAndExactNativeProfiles) {
    auto code = prosper::test::dpp_row_max_program({1, 2, 4, 8});
    code.insert(code.begin(), 0x7e020300u);   // v_mov_b32 v1,v0: genuine production local-id input
    for (uint32_t native : {0u, 64u}) {
        ComputeShaderConfig config;
        config.wave_size = 64;
        config.native_subgroup_size = native;
        const auto module = recompile_compute(code.data(), code.size(), nullptr, config);
        ASSERT_FALSE(module.empty()) << "native width " << native;
        EXPECT_EQ(count(module, Op_ExtInst, Glsl_UMax), 4u);
        EXPECT_EQ(count(module, Op_GroupNonUniformShuffle), 8u);
    }
}
