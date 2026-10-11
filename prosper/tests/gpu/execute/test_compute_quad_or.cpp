// Execute actual ordinary-compute modules through the shipping backend. Distinct per-lane
// sources expose wrong peers, reordered events, and incorrectly retained inactive destinations.
#include "fixtures/compute_quad_or.hpp"
#include "fixtures/render_runner.h"
#include "shared/live/live_compute.hpp"
#include <gtest/gtest.h>
#include <memory>

namespace fixture = prosper::test::compute_quad_or;
namespace live = prosper::frontend;
using namespace prosper::gpu;
class ComputeQuadOrExecution : public testing::TestWithParam<uint32_t> {
protected:
    ComputeShaderConfig config() const {
        auto c = fixture::config(GetParam() == 64 ? 64 : 0);
        c.wave_size = GetParam() == 32 ? 32 : 64;
        return c;
    }
    void SetUp() override {
        ASSERT_TRUE(prosper::test::render_vk_ctx().ok) << "production Vulkan initialization failed";
        const auto shared = shared_vulkan_context();
        ASSERT_TRUE(shared.valid());
        // This synthetic local128 arm explicitly tests the existing opt-in native multiwave
        // contract. It does not change the shipping executor's single-wave default selector.
        if (GetParam() == 64 &&
            select_native_compute_subgroup_size(shared, config(), true, false) != 64)
            GTEST_SKIP() << "enabled exact full native subgroup contract unavailable";
        live::execute_live_compute_items({});
        const auto device = live::live_compute_storage_image_device();
        ASSERT_TRUE(device.initialized);
        ASSERT_TRUE(device.adopted);
    }
    void execute(fixture::Shape shape) {
        std::vector<uint32_t> bytes(fixture::lanes, fixture::sentinel);
        auto resources = fixture::table();
        resources.resources.front().gpu_addr = reinterpret_cast<uint64_t>(bytes.data());
        const auto code = fixture::program(shape);
        ComputeItem item;
        item.code_addr = fixture::address();
        item.recompile_config = config();
        item.spirv = fixture::compile(code, resources, item.recompile_config, item.code_addr);
        ASSERT_FALSE(item.spirv.empty()) << last_terminal_reject_reason(item.code_addr);
        item.required_subgroup_size = compute_spirv_min_subgroup_size(item.spirv);
        item.resources = std::make_shared<ShaderResourceTable>(resources);
        item.user_sgprs = item.recompile_config.user_sgprs;
        item.recompile_config_available = true;
        item.launch.local_x = item.launch.threads_x = fixture::lanes;
        item.launch.local_y = item.launch.local_z = item.launch.threads_y = item.launch.threads_z =
            1;
        item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
        const auto cpu = live::live_compute_cpu_fill_dispatches();
        ASSERT_TRUE(live::execute_live_compute_items({item}));
        EXPECT_EQ(live::live_compute_cpu_fill_dispatches(), cpu);
        EXPECT_EQ(bytes, fixture::expected(shape));
        const auto expected = fixture::expected(shape);
        for (size_t lane = 0; lane < bytes.size(); ++lane)
            EXPECT_EQ(bytes[lane], expected[lane]) << "logical lane " << lane;
    }
};
TEST_P(ComputeQuadOrExecution, DistinctPeerAndUnpermutedLocalSource) {
    execute(fixture::Shape::Plain);
}
TEST_P(ComputeQuadOrExecution, ComplementaryCurrentGenerationDefinitions) {
    execute(fixture::Shape::Complement);
}
TEST_P(ComputeQuadOrExecution, ExecInactiveDestinationAndPeer) {
    execute(fixture::Shape::Masked);
}
TEST_P(ComputeQuadOrExecution, DifferentWavePcsUseDifferentEvents) {
    execute(fixture::Shape::DistinctSites);
}
// 0: portable guest Wave64, 32: portable guest Wave32, 64: explicit native guest Wave64.
INSTANTIATE_TEST_SUITE_P(PortableAndNative64, ComputeQuadOrExecution,
                         testing::Values(0u, 32u, 64u));
