// An entry mask must not force a fictitious Bool at an exit where both paths hold scalar DATA.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/loop_vcc_data_exit.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <vector>

namespace {
namespace fixture = prosper::test::loop_vcc_data_exit;
constexpr uint32_t untouched = 0xdeadbeefu;
void execute(fixture::Options o) {
    SCOPED_TRACE(o.trips);
    SCOPED_TRACE(o.condition);
    SCOPED_TRACE(o.entry_mask);
    const auto module = fixture::compile(o);
    ASSERT_FALSE(module.empty());
    ASSERT_FALSE(fixture::has_switch(module)); // exercise the structured loop, not a fallback
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.push_constants = fixture::config(o).user_sgprs;
    const auto output = vm.run_buffers(64, {{3, std::vector<uint32_t>(80, untouched)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(output.size(), 80u);
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(output[lane], fixture::expected(o)) << lane;
    for (uint32_t lane = 64; lane < output.size(); ++lane) EXPECT_EQ(output[lane], untouched);
}
} // namespace

TEST(LoopVccDataExit, CurrentScalarHalvesSurviveBothNestedBranchOutcomes) {
    for (bool shift : {false, true})
        for (uint32_t trips : {1u, 4u})
            for (uint32_t condition : {0u, 1u})
                execute({.trips = trips, .condition = condition, .shift = shift});
}
TEST(LoopVccDataExit, OmittingTheEntryCompareKeepsTheNumericResult) {
    for (uint32_t condition : {0u, 1u}) execute({.condition = condition, .entry_mask = false});
}
TEST(LoopVccDataExit, AnUnknownLaunchRootRemainsHard) {
    constexpr uint64_t address = 0x4808120100ull;
    EXPECT_TRUE(fixture::compile({.missing_high = true, .known_launch = false}, address).empty());
    EXPECT_NE(prosper::gpu::last_terminal_reject_reason(address).find("origin=unknown-launch"),
              std::string::npos);
}
TEST(LoopVccDataExit, AMissingSpillRootRemainsHard) {
    constexpr uint64_t address = 0x4808120200ull;
    EXPECT_TRUE(fixture::compile({.spill_high = true}, address).empty());
    EXPECT_NE(prosper::gpu::last_terminal_reject_reason(address).find("origin=spill"),
              std::string::npos);
}
