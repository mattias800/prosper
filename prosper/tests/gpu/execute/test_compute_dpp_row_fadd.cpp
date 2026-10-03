// #4300: an inactive rotated source contributes zero; its active distinct destination still writes.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/compute_runner.h"
#include "fixtures/dpp_row_fadd.hpp"
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <tuple>

namespace {
using prosper::test::DppRowFaddCase;
constexpr uint32_t kLanes = 256;
using Values = std::array<float, kLanes>;
constexpr uint32_t kTrips[] = {0, 3, 1, 4};
Values data() {
    Values value{};
    for (uint32_t lane = 0; lane < kLanes; ++lane) value[lane] = 16.0f + lane * 0.25f;
    return value;
}
Values bias() {
    Values value{};
    for (uint32_t lane = 0; lane < kLanes; ++lane) value[lane] = (lane + 1) * 0.25f;
    return value;
}
Values initial_destination() {
    Values value{};
    for (uint32_t lane = 0; lane < kLanes; ++lane) value[lane] = 1000.0f + lane;
    return value;
}

// Independent instruction-order model using complete snapshots and genuine per-wave control.
Values oracle(DppRowFaddCase shape) {
    auto source = data();
    auto destination = initial_destination();
    const auto own = bias();
    const auto add = [&](uint32_t wave, bool masked) {
        const Values previous = source;
        for (uint32_t lane = wave * 64; lane < (wave + 1) * 64; ++lane) {
            if (masked && lane % 5 == 0) continue;
            const uint32_t peer = (lane / 16) * 16 + ((lane % 16 + 8) % 16);
            const float rotated = masked && peer % 5 == 0 ? 0.0f : previous[peer];
            destination[lane] = rotated + own[lane];
        }
        // The fixture's following MOV copies only active destinations into the next SRC0.
        for (uint32_t lane = wave * 64; lane < (wave + 1) * 64; ++lane)
            if (!masked || lane % 5 != 0) source[lane] = destination[lane];
    };
    for (uint32_t wave = 0; wave < 4; ++wave) {
        if (shape == DppRowFaddCase::Linear) {
            add(wave, false);
        } else if (shape == DppRowFaddCase::LoopAndCompletedPeer) {
            for (uint32_t trip = 0; trip < kTrips[wave]; ++trip) add(wave, true);
        } else if (wave % 2 == 0) {
            add(wave, false);
            add(wave, false);
        } else {
            const Values previous = source;
            for (uint32_t lane = wave * 64; lane < (wave + 1) * 64; ++lane) {
                const uint32_t peer = (lane / 16) * 16 + ((lane % 16 + 8) % 16);
                source[lane] = previous[peer];   // the other branch's MOV row event
            }
            add(wave, false);
        }
    }
    return destination;
}

class ComputeDppRowFaddExecution : public testing::TestWithParam<std::tuple<DppRowFaddCase, bool>> {
};
}   // namespace

TEST_P(ComputeDppRowFaddExecution, KeepsOwnAddendDistinctDestinationAndEventHistory) {
    const auto [shape, native] = GetParam();
    if (native && !prosper::test::default_compute_required_subgroup_supported(64, kLanes))
        GTEST_SKIP() << "native profile requires enabled exact Wave64 full subgroups";
    const auto code = prosper::test::dpp_row_fadd_program(shape);
    const auto module = prosper::gpu::recompile_ngg_exports_for_test(
        code.data(), code.size(), 10, 0, nullptr, 4, 0, {}, true, true, native);
    ASSERT_FALSE(module.empty());
    const auto source = data(), own = bias(), original = initial_destination();
    std::vector<float> input(size_t(kLanes) * 10, 0.0f);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        input[size_t(lane) * 10] = std::bit_cast<float>(lane / 64 % 2);
        input[size_t(lane) * 10 + 1] = original[lane];
        input[size_t(lane) * 10 + 2] = source[lane];
        input[size_t(lane) * 10 + 3] = own[lane];
        input[size_t(lane) * 10 + 4] = std::bit_cast<float>(kTrips[lane / 64]);
        input[size_t(lane) * 10 + 5] = std::bit_cast<float>(lane % 5 == 0 ? 0u : 1u);
        input[size_t(lane) * 10 + 6] = std::bit_cast<float>(1u);
        input[size_t(lane) * 10 + 7] = std::bit_cast<float>(1u);
    }
    constexpr uint32_t stride = prosper::gpu::kNggExportProbeWords;
    const auto output =
        prosper::test::run_compute(module, input, kLanes, kLanes * stride, {}, {}, nullptr, kLanes,
                                   nullptr, nullptr, nullptr, native ? 64u : 0u);
    ASSERT_EQ(output.size(), size_t(kLanes) * stride);
    const auto expected = oracle(shape);
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_EQ(std::bit_cast<uint32_t>(output[size_t(lane) * stride]),
                  std::bit_cast<uint32_t>(expected[lane]))
            << "lane " << lane;
}

INSTANTIATE_TEST_SUITE_P(
    Profiles, ComputeDppRowFaddExecution,
    testing::Combine(testing::Values(DppRowFaddCase::Linear, DppRowFaddCase::DivergentSites,
                                     DppRowFaddCase::LoopAndCompletedPeer,
                                     DppRowFaddCase::LaterBarrierPhase),
                     testing::Bool()),
    ([](const testing::TestParamInfo<ComputeDppRowFaddExecution::ParamType>& test) {
        const auto [shape, native] = test.param;
        return std::string("Shape") + std::to_string(static_cast<int>(shape)) +
               (native ? "Native64" : "Portable");
    }));

TEST(ComputeDppRowFaddOracle, DetectsRotatedAddendAndSuppressedBoundedWrite) {
    const auto source = data(), own = bias();
    const auto linear = oracle(DppRowFaddCase::Linear);
    EXPECT_NE(linear[0], source[8] + own[8]) << "SRC1 must remain at the destination lane";
    const auto loop = oracle(DppRowFaddCase::LoopAndCompletedPeer);
    constexpr uint32_t lane = 67;   // active destination; rotated lane 75 has EXEC off
    EXPECT_EQ(loop[lane], own[lane]);
    EXPECT_NE(loop[lane], initial_destination()[lane])
        << "encoded BC1 zero-source arithmetic must still replace distinct VDST";
    EXPECT_EQ(loop[0], initial_destination()[0]) << "completed wave must publish no later event";
}
