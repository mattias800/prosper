// Execute the same declared-bank boundary on hardware, including both halves of Wave64.
#include "fixtures/allocated_movrels.hpp"
#include "fixtures/compute_runner.h"
#include <gtest/gtest.h>
#include <cstdint>
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
