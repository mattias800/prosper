// #4268: exact unsigned row scans, independent EXEC writes, and 16-lane row boundaries.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/compute_runner.h"
#include "fixtures/dpp_row_max.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <vector>

namespace {
constexpr uint32_t kLanes = 128;
using Words = std::array<uint32_t, kLanes>;

Words inputs() {
    Words words{};
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        // Finite raw float bit patterns, with unsigned ordering opposed to signed/float ordering.
        const uint32_t row = lane / 16;
        words[lane] = lane % 4 == 0   ? 0xff7fffffu - row
                      : lane % 4 == 1 ? 0x80000000u + lane
                      : lane % 4 == 2 ? 0x7f7fffffu - lane
                                      : 17u + lane;
    }
    return words;
}

Words oracle(Words words, std::initializer_list<uint32_t> shifts, bool masked) {
    for (uint32_t shift : shifts) {
        const Words previous = words;
        for (uint32_t lane = 0; lane < kLanes; ++lane) {
            const bool destination_active = !masked || lane % 5 != 0;
            const bool in_bounds = lane % 16 >= shift;
            const bool source_active = in_bounds && (!masked || (lane - shift) % 5 != 0);
            if (destination_active && in_bounds && source_active)
                words[lane] = std::max(previous[lane], previous[lane - shift]);
        }
    }
    return words;
}

void execute(std::initializer_list<uint32_t> shifts, bool masked) {
    const auto properties = prosper::test::default_compute_subgroup_properties();
    uint32_t required = 0;
    if (properties.size < 16) {
        if (!prosper::test::default_compute_required_subgroup_supported(16, 64))
            GTEST_SKIP() << "DPP16 needs a supported subgroup of at least 16 lanes";
        required = 16;
    }
    const auto code = prosper::test::dpp_row_max_program(shifts, masked);
    const auto module = prosper::gpu::recompile_valu(code.data(), code.size(), 3, 1);
    ASSERT_FALSE(module.empty());
    const auto source = inputs();
    std::vector<float> input(kLanes * 3);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        input[lane * 3] = std::bit_cast<float>(masked && lane % 5 == 0 ? 0u : 1u);
        input[lane * 3 + 1] = std::bit_cast<float>(source[lane]);
        input[lane * 3 + 2] = std::bit_cast<float>(1u);
    }
    const auto output = prosper::test::run_compute(module, input, kLanes, kLanes, {}, {}, nullptr,
                                                   64, nullptr, nullptr, nullptr, required);
    ASSERT_EQ(output.size(), kLanes) << "dispatch and host readback must complete";
    const auto expected = oracle(source, shifts, masked);
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_EQ(std::bit_cast<uint32_t>(output[lane]), expected[lane]) << "lane " << lane;
}
}   // namespace

TEST(ComputeDppRowMaxExecution, EveryShiftKeepsItsOwnRow) {
    for (uint32_t shift = 1; shift <= 15; ++shift) {
        SCOPED_TRACE(shift);
        execute({shift}, false);
    }
}

TEST(ComputeDppRowMaxExecution, LadderUsesInstructionOrderSnapshots) {
    execute({1, 2, 4, 8}, false);
}

TEST(ComputeDppRowMaxExecution, InactiveSourcesAndDestinationsKeepTheirValues) {
    execute({1, 2, 4, 8}, true);
}

TEST(ComputeDppRowMaxExecution, OracleDistinguishesSignedMaximumAndCrossRowFetch) {
    const auto source = inputs();
    const auto expected = oracle(source, {1}, false);
    EXPECT_NE(expected[2], static_cast<uint32_t>(std::max(static_cast<int32_t>(source[2]),
                                                          static_cast<int32_t>(source[1]))));
    EXPECT_EQ(expected[16], source[16]);
    // Use a separately constructed row-boundary counterexample, not the producer's input order.
    Words boundary{};
    boundary[15] = 0xff7fffff;
    boundary[16] = 7;
    const auto isolated = oracle(boundary, {1}, false);
    EXPECT_NE(isolated[16], std::max(boundary[15], boundary[16]));
}

namespace {
using prosper::test::DppRowCfgCase;
constexpr uint32_t kCfgLanes = 256;
using CfgWords = std::array<uint32_t, kCfgLanes>;

CfgWords cfg_inputs() {
    CfgWords words{};
    for (uint32_t lane = 0; lane < kCfgLanes; ++lane)
        words[lane] = lane % 4 == 0 ? 0xff7fffffu - lane / 16
                      : lane % 4 == 1 ? 0x80000000u + lane
                      : lane % 4 == 2 ? 0x7f7fffffu - lane : 17u + lane;
    return words;
}

// Independent instruction-order oracle: take a full snapshot before each row operation and
// choose each wave's real branch/trip count. A completed peer performs no later operations.
CfgWords cfg_oracle(CfgWords words, DppRowCfgCase shape) {
    const auto row = [&](uint32_t wave, uint32_t shift, bool maximum, bool masked) {
        const CfgWords previous = words;
        for (uint32_t lane = wave * 64; lane < (wave + 1) * 64; ++lane) {
            if (lane % 16 < shift || (masked && (lane % 5 == 0 || (lane - shift) % 5 == 0)))
                continue;
            words[lane] = maximum ? std::max(previous[lane], previous[lane - shift])
                                  : previous[lane] + previous[lane - shift];
        }
    };
    constexpr uint32_t trips[] = {0, 3, 1, 4};
    for (uint32_t wave = 0; wave < 4; ++wave) {
        if (shape == DppRowCfgCase::Mixed) {
            row(wave, 1, false, false); row(wave, 2, true, false);
            row(wave, 4, false, false); row(wave, 8, true, false);
        } else if (shape == DppRowCfgCase::LoopAndCompletedPeer) {
            for (uint32_t iteration = 0; iteration < trips[wave]; ++iteration) {
                row(wave, 1, false, true); row(wave, 2, true, true);
            }
        } else if (wave % 2 == 0) {
            row(wave, 1, true, false); row(wave, 2, true, false);
        } else {
            row(wave, 1, false, false); row(wave, 4, true, false);
            row(wave, 8, false, false);
        }
    }
    return words;
}

class ComputeDppRowMaxCfgExecution : public testing::TestWithParam<bool> {};

void execute_cfg(DppRowCfgCase shape, bool native) {
    if (native && !prosper::test::default_compute_required_subgroup_supported(64, kCfgLanes))
        GTEST_SKIP() << "the native profile requires enabled exact Wave64 full subgroups";
    const auto code = prosper::test::dpp_row_cfg_export_program(shape);
    const auto module = prosper::gpu::recompile_ngg_exports_for_test(
        code.data(), code.size(), 10, 0, nullptr, 4, 0, {}, true, true, native);
    ASSERT_FALSE(module.empty());
    const auto source = cfg_inputs();
    std::vector<float> input(kCfgLanes * 10, 0.0f);
    constexpr uint32_t trips[] = {0, 3, 1, 4};
    for (uint32_t lane = 0; lane < kCfgLanes; ++lane) {
        const uint32_t wave = lane / 64;
        input[lane * 10] = std::bit_cast<float>(wave % 2);
        input[lane * 10 + 1] = std::bit_cast<float>(source[lane]);
        input[lane * 10 + 2] = std::bit_cast<float>(1u);
        input[lane * 10 + 3] = std::bit_cast<float>(trips[wave]);
        input[lane * 10 + 4] = std::bit_cast<float>(lane % 5 == 0 ? 0u : 1u);
        input[lane * 10 + 5] = std::bit_cast<float>(1u);
    }
    constexpr uint32_t stride = prosper::gpu::kNggExportProbeWords;
    const auto output = prosper::test::run_compute(
        module, input, kCfgLanes, kCfgLanes * stride, {}, {}, nullptr, kCfgLanes,
        nullptr, nullptr, nullptr, native ? 64u : 0u);
    ASSERT_EQ(output.size(), kCfgLanes * stride) << "the whole synchronized dispatch must complete";
    const auto expected = cfg_oracle(source, shape);
    for (uint32_t lane = 0; lane < kCfgLanes; ++lane)
        EXPECT_EQ(std::bit_cast<uint32_t>(output[lane * stride]), expected[lane])
            << "lane " << lane << ", native " << native;
}
} // namespace

TEST_P(ComputeDppRowMaxCfgExecution, MixedAddAndMaxPreserveUnsignedOrdering) {
    execute_cfg(DppRowCfgCase::Mixed, GetParam());
}
TEST_P(ComputeDppRowMaxCfgExecution, DifferentWavesUseTheirOwnStaticSites) {
    execute_cfg(DppRowCfgCase::DivergentSites, GetParam());
}
TEST_P(ComputeDppRowMaxCfgExecution, LoopsKeepEndedPeersAndIndependentExecSafe) {
    execute_cfg(DppRowCfgCase::LoopAndCompletedPeer, GetParam());
}
TEST_P(ComputeDppRowMaxCfgExecution, LaterBarrierPhaseHasBothScratchPlanes) {
    execute_cfg(DppRowCfgCase::LaterBarrierPhase, GetParam());
}
INSTANTIATE_TEST_SUITE_P(Profiles, ComputeDppRowMaxCfgExecution, testing::Values(false, true),
    [](const testing::TestParamInfo<bool>& profile) {
        return profile.param ? "Native64" : "Portable";
    });
