// #4268: skipped unsigned DPP row maxima leave a compute producer's output unwritten.
#include "gpu/recompiler/rdna2_dpp_row_shr.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "fixtures/dpp_row_max.hpp"
#include <gtest/gtest.h>
#include <array>
#include <tuple>

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
        auto inactive_fetch = code;
        inactive_fetch[1] |= 1u << 18;
        EXPECT_FALSE(is_inplace_vmax_u32_dpp_row_shr(
            rdna2_decode_one(inactive_fetch.data(), inactive_fetch.size())))
            << "FI1 must not acquire the encoded BC0/FI0 service";
        for (uint32_t mutation = 0; mutation < 17; ++mutation) {
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
                case 12: altered.clamp = true; break;
                case 13: altered.omod = 1; break;
                case 14: altered.has_sdwa = true; break;
                case 15: altered.src_neg[1] = true; break;
                case 16: altered.src_abs[0] = true; break;
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

TEST(ComputeDppRowMax, DispatcherUsesTaggedPortablePhaseInsteadOfHostShuffles) {
    for (bool add : {false, true}) {
        const auto code = prosper::test::dpp_row_max_program({1, 2, 4, 8}, false, add);
        const auto module = compile(code, true);
        ASSERT_FALSE(module.empty()) << "MAX and the established ADD control both compile";
        EXPECT_GT(count(module, Op_Switch), 0u);
        EXPECT_GE(count(module, Op_ControlBarrier), 2u);
        EXPECT_EQ(count(module, Op_GroupNonUniformShuffle), 0u)
            << "portable CFG peers are addressed through event/EXEC scratch";
        EXPECT_EQ(count(module, Op_ExtInst, Glsl_UMax), add ? 0u : 1u)
            << "one common phase services every static site";
    }
}

TEST(ComputeDppRowMax, MixedBranchesLoopsAndLaterBarrierCompileBothProfiles) {
    using prosper::test::DppRowCfgCase;
    for (const auto shape : {DppRowCfgCase::Mixed, DppRowCfgCase::DivergentSites,
                             DppRowCfgCase::LoopAndCompletedPeer,
                             DppRowCfgCase::LaterBarrierPhase}) {
        const auto code = prosper::test::dpp_row_cfg_export_program(shape);
        for (bool native : {false, true}) {
            const auto module = recompile_ngg_exports_for_test(
                code.data(), code.size(), 10, 0, nullptr, 4, 0, {}, true, true, native);
            ASSERT_FALSE(module.empty()) << "shape " << static_cast<int>(shape)
                                         << ", exact native " << native;
            EXPECT_GT(count(module, Op_Switch), 0u);
            EXPECT_GT(count(module, Op_ExtInst, Glsl_UMax), 0u);
            EXPECT_EQ(count(module, Op_GroupNonUniformShuffle) != 0, native);
        }
    }
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

namespace {
// The matched positive writes a different physical VGPR. Both streams use genuine input v1;
// only the negative replaces it with WRITELANE's scalar numeric/Bool spill representation.
class ComputeDppRowMaxCfgSpill : public testing::TestWithParam<std::tuple<bool, bool, bool>> {};
}

TEST_P(ComputeDppRowMaxCfgSpill, RejectsScalarSpillBeforePublishingRowSource) {
    const auto [mask, restored, native] = GetParam();
    for (bool spill_source : {false, true}) {
        std::vector<uint32_t> code{0xbf820000u}; // explicit CFG entry before WRITELANE
        code.push_back(0xd7610000u | (spill_source ? 1u : 8u));
        code.push_back(mask ? 0x0001007eu : 0x00010081u); // EXEC_LO or numeric 1, inline lane 0
        const auto write = rdna2_decode_one(code.data() + 1, code.size() - 1);
        ASSERT_EQ(write.opcode, 0x361u);
        ASSERT_EQ(write.dst.value, spill_source ? 1 : 8);
        ASSERT_EQ(write.src[1].kind, OperandKind::InlineInt);
        ASSERT_EQ(write.src[1].value, 0);
        if (restored) code.push_back(0xbf820000u); // save/load CFG state before the row event
        const uint32_t max_pc = static_cast<uint32_t>(code.size());
        const auto row = prosper::test::dpp_row_max_program({1});
        code.insert(code.end(), row.begin(), row.end() - 1);
        code.insert(code.end(), {0xf8000941u, 0x00000001u, 0xbf810000u}); // raw v1 sink
        const uint64_t address = 0x42680000u + (mask << 8) + (restored << 6) +
                                 (native << 4) + spill_source;
        const auto module = recompile_ngg_exports_for_test(
            code.data(), code.size(), 10, 0, nullptr, 4, 0,
            {RecompileDiagnosticStage::Compute, address}, true, true, native);
        if (spill_source) {
            EXPECT_TRUE(module.empty());
            const auto reason = last_terminal_reject_reason(address);
            EXPECT_NE(reason.find("reason=dpp-row-max-source-unresolved"), std::string::npos)
                << reason;
            EXPECT_NE(reason.find("pc=" + std::to_string(max_pc)), std::string::npos)
                << reason;
        } else {
            ASSERT_FALSE(module.empty()) << "unrelated spill must not erase genuine v1 input";
            EXPECT_GT(count(module, Op_Switch), 0u);
            EXPECT_GT(count(module, Op_ExtInst, Glsl_UMax), 0u);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Profiles, ComputeDppRowMaxCfgSpill,
    testing::Combine(testing::Bool(), testing::Bool(), testing::Bool()),
    ([](const testing::TestParamInfo<ComputeDppRowMaxCfgSpill::ParamType>& test) {
        const auto [mask, restored, native] = test.param;
        return std::string(mask ? "Mask" : "Numeric") + (restored ? "Restored" : "Adjacent") +
               (native ? "Native64" : "Portable");
    }));
