// ADR 0028: the frontend glue that names a Wave64 route on a refusal line. A stand-in context
// supplies the fields live_compute.cpp's real one has, so nothing here needs a Vulkan device.
#include <gtest/gtest.h>

#include <memory>

#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/compute_wave_route.hpp"
#include "shared/live/compute_wave_admission.hpp"

using namespace prosper::frontend;
using namespace prosper::gpu;

namespace {

struct FakeContext {
    uint32_t min_native_subgroup_size = 32, max_native_subgroup_size = 32, subgroup_size = 32;
    bool native_subgroup_contract = true;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
};

ComputeItem item_with(ComputeCrossLaneKind kind, ComputeWaveContext context) {
    ComputeItem item;
    item.recompile_config_available = true;
    item.recompile_config.local_x = 64;
    item.recompile_config.wave_size = 64;
    item.recompile_config.threads_x = 64;
    ComputeWaveOpFacts facts;
    facts.analyzed = true;
    facts.ops.push_back({3, kind, context, true, 0});
    item.wave_ops = std::make_shared<ComputeWaveOpFacts>(facts);
    return item;
}

}  // namespace

TEST(ComputeWaveAdmission, AnItemWithoutFactsSaysUnanalyzedNotNoCrossLaneOp) {
    ComputeItem item;
    EXPECT_STREQ(compute_wave_route_text(FakeContext{}, item, {32768, 1024}).text, "unanalyzed");
}

TEST(ComputeWaveAdmission, TheRouteTextNamesTheRouteAndTheReason) {
    const auto loop = item_with(ComputeCrossLaneKind::ReadLane, ComputeWaveContext::Loop);
    EXPECT_STREQ(compute_wave_route_text(FakeContext{}, loop, {32768, 1024}).text,
                 "needs-n-lanes:cross-lane-in-loop");
    const auto top = item_with(ComputeCrossLaneKind::Ballot, ComputeWaveContext::TopLevel);
    EXPECT_STREQ(compute_wave_route_text(FakeContext{}, top, {32768, 1024}).text,
                 "workgroup-exchange:cross-lane-in-uniform-flow");
    // Unknown device limits are refused, not assumed.
    EXPECT_STREQ(compute_wave_route_text(FakeContext{}, top, {0, 0}).text,
                 "refused:shared-memory-limit-unknown");
}

TEST(ComputeWaveAdmission, ANativeContractRoutesNative) {
    FakeContext ctx;
    ctx.min_native_subgroup_size = ctx.max_native_subgroup_size = 64;
    auto item = item_with(ComputeCrossLaneKind::ReadLane, ComputeWaveContext::Loop);
    item.required_subgroup_size = 64;
    EXPECT_STREQ(compute_wave_route_text(ctx, item, {32768, 1024}).text,
                 "native:host-subgroup-covers-guest-wave");
}

TEST(ComputeWaveAdmission, APartialWorkgroupIsRefused) {
    auto item = item_with(ComputeCrossLaneKind::Ballot, ComputeWaveContext::TopLevel);
    item.recompile_config.exact_thread_extent = true;
    item.recompile_config.threads_x = 40;
    item.recompile_config.threads_y = item.recompile_config.threads_z = 1;
    EXPECT_STREQ(compute_wave_route_text(FakeContext{}, item, {32768, 1024}).text,
                 "refused:partial-workgroup-barrier");
}
