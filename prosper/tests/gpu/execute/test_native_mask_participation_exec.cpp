#include "fixtures/native_mask_participation.hpp"
#include "fixtures/render_runner.h"
#include "shared/live/live_compute.hpp"

#include <gtest/gtest.h>
#include <memory>

namespace fixture = prosper::test::native_mask_participation;
namespace live = prosper::frontend;
using namespace prosper::gpu;
namespace {
class NativeMaskParticipationExecution : public testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(prosper::test::render_vk_ctx().ok)
            << "production renderer Vulkan initialization failed";
        const auto shared = shared_vulkan_context();
        ASSERT_TRUE(shared.valid());
        if (select_native_compute_subgroup_size(shared, fixture::config(), false, false) != 64)
            GTEST_SKIP() << "enabled exact full Wave64 subgroup contract unavailable";
        live::execute_live_compute_items({});
        const auto device = live::live_compute_storage_image_device();
        ASSERT_TRUE(device.initialized);
        ASSERT_TRUE(device.adopted)
            << "native fixture must use the actual shared production device";
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
    ASSERT_EQ(compute_spirv_min_subgroup_size(item.spirv), 64u);
    item.required_subgroup_size = 64;
    item.resources = std::make_shared<ShaderResourceTable>(resources);
    item.user_sgprs = fixture::config().user_sgprs;
    item.recompile_config_available = true;
    item.recompile_config = fixture::config();
    item.launch.local_x = item.launch.threads_x = 64;
    item.launch.local_y = item.launch.local_z = item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.groups_x = fixture::trips.size();
    item.launch.groups_y = item.launch.groups_z = 1;
    const auto cpu = live::live_compute_cpu_fill_dispatches();
    ASSERT_TRUE(live::execute_live_compute_items({item}));
    EXPECT_EQ(live::live_compute_cpu_fill_dispatches(), cpu);
    EXPECT_EQ(bytes, fixture::expected(options))
        << "all current LO/HI/FFBH words, differing loop epochs, untouched inputs and padding";
}
} // namespace
TEST_F(NativeMaskParticipationExecution, DifferentGroupsAndIterationsKeepFreshPhysicalWords) {
    execute();
}
TEST_F(NativeMaskParticipationExecution, ExecOffPeerRemainsARealSelectedSource) {
    execute({.exec_off_peer = true});
}
TEST_F(NativeMaskParticipationExecution, OpaqueRoutingPrefixDoesNotReplaceGenuineProducer) {
    execute({.opaque_prefix = true});
}
TEST_F(NativeMaskParticipationExecution, RealMaskedLoadsPreserveInitializedPredicateLanes) {
    execute({.masked_load = true});
}
