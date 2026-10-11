// #4907: use the production live compute device and complete buffer writeback. Four distinct
// guest waves and real LDS across phases expose wrong peers and divergent host barrier schedules.
#include "fixtures/readlane_barrier.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "shared/live/live_compute.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

namespace fixture = prosper::test::readlane_barrier;
namespace live = prosper::frontend;
using namespace prosper::gpu;
namespace {
class ReadlaneBarrierExecution : public testing::Test {
protected:
    void SetUp() override {
        live::execute_live_compute_items({});
        ASSERT_TRUE(live::live_compute_storage_image_device().initialized)
            << "production Vulkan initialization failed";
    }
};

void execute(fixture::Options options = {}) {
    auto bytes = fixture::input();
    auto resources = fixture::table();
    resources.resources.front().gpu_addr = reinterpret_cast<uint64_t>(bytes.data());
    ComputeItem item;
    item.code_addr = fixture::address();
    item.spirv = fixture::compile(fixture::program(options), resources, item.code_addr);
    ASSERT_FALSE(item.spirv.empty()) << last_terminal_reject_reason(item.code_addr);
    item.resources = std::make_shared<ShaderResourceTable>(resources);
    item.launch.local_x = item.launch.threads_x = fixture::lanes;
    item.launch.local_y = item.launch.local_z = item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    const auto cpu_fills = live::live_compute_cpu_fill_dispatches();
    ASSERT_TRUE(live::execute_live_compute_items({item}));
    EXPECT_EQ(live::live_compute_cpu_fill_dispatches(), cpu_fills);
    EXPECT_EQ(bytes, fixture::expected(options))
        << "actual device values plus every untouched input and padding word";
}
} // namespace

TEST_F(ReadlaneBarrierExecution, BalancedTwoPhaseLdsAndScalarValues) {
    execute({.loop = false, .differing_peers = false});
}
TEST_F(ReadlaneBarrierExecution, FourGuestWavesUseDifferentStaticPeerSites) {
    execute({.loop = false});
}
TEST_F(ReadlaneBarrierExecution, DivergentTripCountsKeepCompletedWavesParticipating) {
    execute();
}
TEST_F(ReadlaneBarrierExecution, ExecOffSelectedPeerKeepsItsRealValue) {
    execute({.narrowed_readlane = true});
}
