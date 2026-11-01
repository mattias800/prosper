// Execute the same declared-bank boundary on hardware, including both halves of Wave64.
#include "fixtures/allocated_movrels.hpp"
#include "fixtures/compute_runner.h"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstddef>
#include <vector>
namespace fixture = prosper::test::allocated_movrels;
TEST(AllocatedMovrelsGpu, CurrentVgprZeroAndInRangeSourcesExecuteOnNativeWave64) {
    if (!prosper::test::default_compute_required_subgroup_supported(64, 64))
        GTEST_SKIP() << "native Wave64 compute unavailable";
    for (uint32_t offset : {0u, 6u, 22u, 23u, UINT32_MAX}) {
        for (bool rewrite : {false, true}) {
            SCOPED_TRACE(offset);
            SCOPED_TRACE(rewrite);
            const auto module = fixture::compile(offset, 64, rewrite);
            ASSERT_FALSE(module.empty());
            std::vector<uint32_t> output;
            prosper::test::run_compute(
                module, {0.0f}, 64, 1, {}, std::vector<uint32_t>(64, 0xdeadbeefu), &output, 64,
                nullptr, nullptr, nullptr, 64, nullptr, fixture::config(offset).user_sgprs);
            EXPECT_EQ(output, fixture::expected(offset, 64, rewrite));
        }
    }
}

TEST(AllocatedMovrelsGpu, RelativePaddingWritesSurviveTheNativeDispatcher) {
    if (!prosper::test::default_compute_required_subgroup_supported(64, 64))
        GTEST_SKIP() << "native Wave64 compute unavailable";
    for (uint32_t offset : {20u, 22u, 23u}) {
        const auto module = fixture::compile(offset, 64, false, true);
        ASSERT_FALSE(module.empty());
        std::vector<uint32_t> output;
        prosper::test::run_compute(
            module, {0.0f}, 64, 1, {}, std::vector<uint32_t>(64, 0xdeadbeefu), &output, 64, nullptr,
            nullptr, nullptr, 64, nullptr, fixture::config(offset).user_sgprs);
        EXPECT_EQ(output,
                  offset < 23 ? std::vector<uint32_t>(64, 105u) : fixture::expected(offset, 64));
    }
}

TEST(AllocatedMovrelsGpu, DsAddressAndWideDataFitTheDeclaredBank) {
    if (!prosper::test::default_compute_required_subgroup_supported(64, 64))
        GTEST_SKIP() << "native Wave64 compute unavailable";
    auto c = fixture::config(0);
    c.lds_bytes = 512;
    for (bool wide : {false, true}) {
        const auto resources = fixture::ds_table(wide);
        const auto words = fixture::ds_boundary(wide);
        const auto module =
            prosper::gpu::recompile_compute(words.data(), words.size(), &resources, c);
        ASSERT_FALSE(module.empty()) << wide;
        std::vector<uint32_t> expected(std::size_t{64} * (wide ? 2u : 1u), 7u);
        if (wide)
            for (std::size_t i = 1; i < expected.size(); i += 2) expected[i] = 13;
        std::vector<uint32_t> output;
        prosper::test::run_compute(module, {0.0f}, 64, 1, {},
                                   std::vector<uint32_t>(expected.size(), 0xdeadbeefu), &output, 64,
                                   nullptr, nullptr, nullptr, 64, nullptr, c.user_sgprs);
        EXPECT_EQ(output, expected);
    }
}
