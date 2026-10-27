// A real exact64 dispatch must preserve the initialized low half and arbitrary old high half.
#include "fixtures/arbitrary_placeholder.hpp"
#include "fixtures/render_runner.h"
#include "shared/live/live_compute.hpp"
#include <gtest/gtest.h>
#include <memory>

namespace fixture = prosper::test::arbitrary_placeholder;
namespace live = prosper::frontend;
using namespace prosper::gpu;

TEST(ArbitraryPlaceholderGpu, TheActualWaveKeepsBothHalvesOfTheRetainedValue) {
    ASSERT_TRUE(prosper::test::render_vk_ctx().ok);
    const auto shared = shared_vulkan_context();
    ASSERT_TRUE(shared.valid());
    if (select_native_compute_subgroup_size(shared, fixture::config(), false, false) != 64)
        GTEST_SKIP() << "enabled exact64 subgroup contract unavailable";
    live::execute_live_compute_items({});
    const auto device = live::live_compute_storage_image_device();
    ASSERT_TRUE(device.initialized);
    ASSERT_TRUE(device.adopted);
    fixture::Words output(64, 0xdeadbeefu);
    auto resources = fixture::table(reinterpret_cast<uint64_t>(output.data()));
    const auto code = fixture::narrowed_writer();
    ComputeItem item;
    item.code_addr = 0x5497b00200ull;
    item.spirv = recompile_compute(code.data(), code.size(), &resources, fixture::config(),
                                   {RecompileDiagnosticStage::Compute, item.code_addr});
    ASSERT_FALSE(item.spirv.empty()) << last_terminal_reject_reason(item.code_addr);
    item.required_subgroup_size = 64;
    item.resources = std::make_shared<ShaderResourceTable>(resources);
    item.user_sgprs = fixture::config().user_sgprs;
    item.recompile_config_available = true;
    item.recompile_config = fixture::config();
    item.launch.local_x = item.launch.threads_x = 64;
    item.launch.local_y = item.launch.local_z = item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    const auto cpu = live::live_compute_cpu_fill_dispatches();
    ASSERT_TRUE(live::execute_live_compute_items({item}));
    EXPECT_EQ(live::live_compute_cpu_fill_dispatches(), cpu);
    EXPECT_EQ(output, fixture::expected(true));
}
