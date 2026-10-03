// #4300: portable floating row rotations must write their distinct destination, not skip a producer.
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "fixtures/dpp_row_fadd.hpp"
#include <gtest/gtest.h>
#include <tuple>

using namespace prosper::gpu;
namespace {
uint32_t count(const std::vector<uint32_t>& module, uint32_t opcode) {
    uint32_t result = 0;
    for (size_t i = 5; i < module.size();) {
        const uint32_t words = module[i] >> 16;
        if (!words || words > module.size() - i) return 0;
        result += (module[i] & 0xffffu) == opcode;
        i += words;
    }
    return result;
}
}   // namespace

TEST(ComputeDppRowFadd, AdmitsOnlyUnmodifiedBoundedRotateEight) {
    const auto code = prosper::test::dpp_row_fadd_program(prosper::test::DppRowFaddCase::Linear);
    const auto in = rdna2_decode_one(code.data(), code.size());
    ASSERT_EQ(dpp_row_ror8_op(in), DppRowRor8Op::AddF32);
    ASSERT_NE(in.dst.value, in.src[0].value);
    ASSERT_NE(in.src[0].value, in.src[1].value);
    for (uint32_t stride = 0; stride < 16; ++stride) {
        auto other = in;
        other.dpp_ctrl = 0x160u + stride;
        EXPECT_EQ(dpp_row_ror8_op(other), DppRowRor8Op::None)
            << "the portable phase must not silently rotate another XOR stride by eight";
    }
    for (uint32_t change = 0; change < 12; ++change) {
        auto other = in;
        switch (change) {
            case 0: other.dpp_bound_ctrl = false; break;
            case 1: other.dpp_row_mask = 7; break;
            case 2: other.dpp_bank_mask = 7; break;
            case 3: other.has_modifier = true; break;
            case 4: other.has_sdwa = true; break;
            case 5: other.src_neg[0] = true; break;
            case 6: other.src_neg[1] = true; break;
            case 7: other.src_abs[0] = true; break;
            case 8: other.src_abs[1] = true; break;
            case 9: other.clamp = true; break;
            case 10: other.omod = 1; break;
            case 11: other.n_src = 1; break;
        }
        EXPECT_EQ(dpp_row_ror8_op(other), DppRowRor8Op::None) << change;
    }
    auto fi1 = code;
    fi1[1] |= 1u << 18;
    EXPECT_EQ(dpp_row_ror8_op(rdna2_decode_one(fi1.data(), fi1.size())), DppRowRor8Op::None);
}

TEST(ComputeDppRowFadd, LinearBranchesLoopsAndLaterPhasesCompileBothProfiles) {
    using prosper::test::DppRowFaddCase;
    for (auto shape : {DppRowFaddCase::Linear, DppRowFaddCase::DivergentSites,
                       DppRowFaddCase::LoopAndCompletedPeer, DppRowFaddCase::LaterBarrierPhase}) {
        const auto code = prosper::test::dpp_row_fadd_program(shape);
        for (bool native : {false, true}) {
            const auto module = recompile_ngg_exports_for_test(
                code.data(), code.size(), 10, 0, nullptr, 4, 0, {}, true, true, native);
            ASSERT_FALSE(module.empty()) << static_cast<int>(shape) << ", native " << native;
            EXPECT_GT(count(module, Op_FAdd), 0u);
            if (!native) {
                EXPECT_GT(count(module, Op_Switch), 0u);
                EXPECT_GE(count(module, Op_ControlBarrier), 2u);
                EXPECT_EQ(count(module, Op_GroupNonUniformShuffle), 0u);
            }
        }
    }
}

namespace {
class ComputeDppRowFaddSpill : public testing::TestWithParam<std::tuple<bool, bool, bool>> {};
}   // namespace
TEST_P(ComputeDppRowFaddSpill, GuardsEachOperandBeforePublishing) {
    const auto [mask, second, native] = GetParam();
    for (bool spill_source : {false, true}) {
        std::vector<uint32_t> code{0xbf820000u,
                                   0xd7610000u | (spill_source ? (second ? 3u : 2u) : 8u),
                                   mask ? 0x0001007eu : 0x00010081u,
                                   (0x14u << 25) | (7u << 17) | (7u << 9) | 0xfau, 0xff011107u};
        // The already-supported MAX service selects synchronized native/portable dispatch.
        // Export independent genuine v4 to isolate source refusal at the new floating event.
        const uint32_t pc = static_cast<uint32_t>(code.size());
        code.insert(code.end(), {(3u << 25) | (1u << 17) | (3u << 9) | 0xfau, 0xff092802u,
                                 0xf8000941u, 0x00000004u, 0xbf810000u});
        const uint64_t address =
            0x43000000u + (mask << 8) + (second << 6) + (native << 4) + spill_source;
        const auto module = recompile_ngg_exports_for_test(
            code.data(), code.size(), 10, 0, nullptr, 4, 0,
            {RecompileDiagnosticStage::Compute, address}, true, true, native);
        if (spill_source) {
            EXPECT_TRUE(module.empty());
            const auto reason = last_terminal_reject_reason(address);
            EXPECT_NE(reason.find("reason=dpp-row-fadd-source-unresolved"), std::string::npos)
                << reason;
            EXPECT_NE(reason.find("pc=" + std::to_string(pc)), std::string::npos) << reason;
        } else {
            ASSERT_FALSE(module.empty()) << "unrelated scalar spill must retain both real operands";
            EXPECT_GT(count(module, Op_FAdd), 0u);
        }
    }
}
INSTANTIATE_TEST_SUITE_P(
    Profiles, ComputeDppRowFaddSpill,
    testing::Combine(testing::Bool(), testing::Bool(), testing::Bool()),
    ([](const testing::TestParamInfo<ComputeDppRowFaddSpill::ParamType>& test) {
        const auto [mask, second, native] = test.param;
        return std::string(mask ? "Mask" : "Numeric") + (second ? "Src1" : "Src0") +
               (native ? "Native64" : "Portable");
    }));
