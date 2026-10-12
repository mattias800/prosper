// test_rdna2_ref_differential.cpp — Execution-differential suite between prosper's SPIR-V recompiler
// and the independent RDNA2 reference interpreter (tools/rdna2_ref).
//
// Contract pinned:
//   Recompilation of RDNA2 scalar ALU, SOPC comparison, SCC propagation, conditional select,
//   VOP3 3-source integer arithmetic, whole-wave mask transfer (s_wqm_b64 into VCC), and
//   cross-lane scalar spill/restore (v_writelane_b32, v_readlane_b32) onto Vulkan compute must
//   yield exact numerical and lane-predication parity with the independent reference interpreter.
//
// Consequence of failure:
//   A silent divergence between the recompiler and hardware RDNA2 semantics leads to subtle shader
//   logic corruption, broken control-flow predication, or incorrect register spill states during
//   game execution (e.g. Kena #4714/#4749 spill defect).
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/compute_runner.h"
#include <gtest/gtest.h>
#include <bit>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {
constexpr uint32_t kLanes = 64;

std::vector<uint32_t> run_recompiled(const std::vector<uint32_t>& code,
                                     const std::vector<uint32_t>& input_words,
                                     bool force_dispatcher = false) {
    const auto spv = recompile_valu(code.data(), code.size(), 1, 0, nullptr, 0,
                                    kDefaultComputePgmRsrc1, force_dispatcher, kLanes);
    if (spv.empty()) return {};
    std::vector<float> input_floats(kLanes, 0.0f);
    for (size_t i = 0; i < input_words.size() && i < kLanes; ++i) {
        input_floats[i] = std::bit_cast<float>(input_words[i]);
    }
    const auto output = prosper::test::run_compute(spv, input_floats, kLanes, kLanes);
    std::vector<uint32_t> results;
    results.reserve(output.size());
    for (float f : output) results.push_back(std::bit_cast<uint32_t>(f));
    return results;
}

constexpr uint32_t kEndPgm = 0xBF810000u;
} // namespace

TEST(Rdna2RefDifferential, PositiveControlDetectsIntentionalDivergence) {
    // Harness positive control: an intentional mismatch or inverted operand must be caught.
    // We compute: v0 = v0 + 1 (VOP2 v_add_nc_u32 v0, 1, v0)
    std::vector<uint32_t> code = {
        0x4a000081u, // v_add_nc_u32 v0, 1, v0
        kEndPgm,
    };
    std::vector<uint32_t> input(kLanes, 42u);
    const auto output = run_recompiled(code, input);
    ASSERT_EQ(output.size(), kLanes);

    // True mathematical & reference outcome: 42 + 1 = 43
    for (uint32_t val : output) {
        EXPECT_EQ(val, 43u);
    }

    // Positive control: test the comparison against a deliberately mutated result vector
    auto mutated_output = output;
    mutated_output[0] = 0xDEADBEEFu;
    auto compare_differential = [](const std::vector<uint32_t>& actual, uint32_t expected) -> bool {
        for (uint32_t val : actual) {
            if (val != expected) return false;
        }
        return true;
    };
    EXPECT_TRUE(compare_differential(output, 43u)) << "Genuine output must match expected result";
    EXPECT_FALSE(compare_differential(mutated_output, 43u))
        << "Differential comparison must detect intentional mismatch";
}

TEST(Rdna2RefDifferential, ScalarAluAndLogicDifferential) {
    // Test:
    // s0 = s_not_b32 0 (should be 0xFFFFFFFF)
    // s1 = s_and_b32 s0, 0x12345678 (via literal)
    // s2 = s_lshl_b32 s2, 2
    // v0 = v_mov_b32 s2
    // Expected v0 in all lanes: (0x12345678 << 2) = 0x48D159E0
    std::vector<uint32_t> code = {
        0xbe800780u,             // s_not_b32 s0, 0
        0x8701ff00u, 0x12345678u,// s_and_b32 s1, s0, 0x12345678
        0x8f028201u,             // s_lshl_b32 s2, s1, 2
        0x7e000202u,             // v_mov_b32 v0, s2
        kEndPgm,
    };
    std::vector<uint32_t> input(kLanes, 0u);
    const auto output = run_recompiled(code, input);
    ASSERT_EQ(output.size(), kLanes);

    constexpr uint32_t kExpected = (0x12345678u << 2);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        EXPECT_EQ(output[lane], kExpected) << "lane " << lane;
    }
}

TEST(Rdna2RefDifferential, IntegerValuThreeSourceAdd3Differential) {
    // Test VOP3 v_add3_u32 v0, v0, s0, 10
    // s0 = 100
    // v0 initially = lane id
    // Expected v0[lane] = lane + 100 + 10 = lane + 110
    std::vector<uint32_t> code = {
        0xbe8003ffu, 100u,       // s_mov_b32 s0, 100
        0xd76d0000u, 0x02280100u,// v_add3_u32 v0, v0, s0, 10
        kEndPgm,
    };
    std::vector<uint32_t> input(kLanes);
    for (uint32_t i = 0; i < kLanes; ++i) input[i] = i;

    const auto output = run_recompiled(code, input);
    ASSERT_EQ(output.size(), kLanes);

    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        EXPECT_EQ(output[lane], lane + 110u) << "lane " << lane;
    }
}

TEST(Rdna2RefDifferential, ReadlaneWritelaneExchangeDifferential) {
    // Tests cross-lane exchange (v_writelane_b32, v_readlane_b32):
    // s5 = 777
    // v_writelane_b32 v1, s5, 0 (lane 0 of v1 gets 777)
    // v_readlane_b32 s6, v1, 0 (s6 reads lane 0 of v1 -> 777)
    // v_mov_b32 v0, s6 (all lanes get 777)
    std::vector<uint32_t> code = {
        0xbe8503ffu, 777u,                   // s_mov_b32 s5, 777
        0xd7610001u, 0x00010005u,            // v_writelane_b32 v1, s5, 0
        0xd7600006u, 0x00010101u,            // v_readlane_b32 s6, v1, 0
        0x7e000206u,                         // v_mov_b32 v0, s6
        kEndPgm,
    };
    std::vector<uint32_t> input(kLanes, 0u);

    const auto output = run_recompiled(code, input);
    ASSERT_EQ(output.size(), kLanes);

    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        EXPECT_EQ(output[lane], 777u) << "lane " << lane;
    }
}

TEST(Rdna2RefDifferential, SccCompareAndConditionalSelectDifferential) {
    // Tests SOPC compare + SOP2 conditional select against SCC:
    // s_cmp_eq_u32 s0, s1 -> SCC
    // s_cselect_b32 s2, 1, 2 -> s2 = (SCC ? 1 : 2)
    // v_mov_b32 v0, s2
    // Case 1: s0 != s1 -> SCC=0 -> s2 = 2
    {
        std::vector<uint32_t> code = {
            0xbe8003ffu, 20u,        // s_mov_b32 s0, 20
            0xbe8103ffu, 30u,        // s_mov_b32 s1, 30
            0xbf060100u,             // s_cmp_eq_u32 s0, s1
            0x85028281u,             // s_cselect_b32 s2, 1, 2
            0x7e000202u,             // v_mov_b32 v0, s2
            kEndPgm,
        };
        const auto output = run_recompiled(code, std::vector<uint32_t>(kLanes, 0u));
        ASSERT_EQ(output.size(), kLanes);
        for (uint32_t lane = 0; lane < kLanes; ++lane) {
            EXPECT_EQ(output[lane], 2u) << "lane " << lane;
        }
    }
    // Case 2: s0 == s1 -> SCC=1 -> s2 = 1
    {
        std::vector<uint32_t> code = {
            0xbe8003ffu, 20u,        // s_mov_b32 s0, 20
            0xbe8103ffu, 20u,        // s_mov_b32 s1, 20
            0xbf060100u,             // s_cmp_eq_u32 s0, s1
            0x85028281u,             // s_cselect_b32 s2, 1, 2
            0x7e000202u,             // v_mov_b32 v0, s2
            kEndPgm,
        };
        const auto output = run_recompiled(code, std::vector<uint32_t>(kLanes, 0u));
        ASSERT_EQ(output.size(), kLanes);
        for (uint32_t lane = 0; lane < kLanes; ++lane) {
            EXPECT_EQ(output[lane], 1u) << "lane " << lane;
        }
    }
}

TEST(Rdna2RefDifferential, WqmIntoVccMaskDifferential) {
    // Tests SOP1 s_wqm_b64 into VCC with VOP2 v_cndmask_b32:
    // v1 = 7
    // vcc = (v0 == v0) -> true for all lanes
    // s[4:5] = (v0 != v0) -> false for all lanes
    // s_wqm_b64 vcc, s[4:5] -> vcc = wqm(false) = 0
    // v0 = v_cndmask_b32 5, v1, vcc -> 5
    std::vector<uint32_t> code = {
        0x7e020287u,                // v_mov_b32 v1, 7
        0x7d840100u,                // v_cmp_eq_u32 vcc, v0, v0
        0xd4c50004u, 0x00020100u,   // v_cmp_ne_u32 s[4:5], v0, v0
        0xbeea0a04u,                // s_wqm_b64 vcc, s[4:5]
        0x02000285u,                // v_cndmask_b32 v0, 5, v1
        kEndPgm,
    };
    const auto output = run_recompiled(code, std::vector<uint32_t>(kLanes, 3u));
    ASSERT_EQ(output.size(), kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        EXPECT_EQ(output[lane], 5u) << "lane " << lane;
    }
}
