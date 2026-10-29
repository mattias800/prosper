// The same structured scalar-DATA exit must survive real native Wave64 execution.
#include "fixtures/compute_runner.h"
#include "fixtures/loop_vcc_data_exit.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

TEST(LoopVccDataExitGpu, BothScalarBranchesObserveTheExitIterationOnRequiredWave64) {
    if (!prosper::test::default_compute_required_subgroup_supported(64, 64))
        GTEST_SKIP() << "native Wave64 compute unavailable";
    namespace fixture = prosper::test::loop_vcc_data_exit;
    for (bool shift : {false, true}) {
        for (uint32_t trips : {1u, 4u}) {
            for (uint32_t condition : {0u, 1u}) {
                const fixture::Options o{.trips = trips, .condition = condition, .shift = shift};
                const auto module = fixture::compile(o);
                ASSERT_FALSE(module.empty());
                ASSERT_FALSE(fixture::has_switch(module));
                std::vector<uint32_t> output;
                prosper::test::run_compute(
                    module, {0.0f}, 64, 1, {}, std::vector<uint32_t>(64, 0xdeadbeefu), &output, 64,
                    nullptr, nullptr, nullptr, 64, nullptr, fixture::config(o).user_sgprs);
                EXPECT_EQ(output, std::vector<uint32_t>(64, fixture::expected(o)));
            }
        }
    }
}
