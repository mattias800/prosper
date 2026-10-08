// ADR 0028: the frontend glue that names a Wave64 route on a refusal line and checks an exchange
// module against the device limits. A stand-in context supplies the fields live_compute.cpp's real
// one has, so nothing here needs a Vulkan device.
#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <vector>

#include "gpu/recompiler/compute_wave_route.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "shared/live/compute_wave_admission.hpp"

#include "fixtures/wave64_exchange_fixture.hpp"

namespace fx = prosper::test::wave64_exchange;
using namespace prosper::frontend;
using namespace prosper::gpu;

namespace {

struct FakeContext {
    uint32_t min_native_subgroup_size = 32, max_native_subgroup_size = 32, subgroup_size = 32;
    bool native_subgroup_contract = true;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
};

ComputeItem item_for(const fx::Case& c, uint32_t exchange_width) {
    ComputeItem item;
    item.spirv = fx::compile(c, exchange_width);
    item.recompile_config_available = true;
    item.recompile_config.local_x = c.local;
    item.recompile_config.wave_size = 64;
    item.recompile_config.threads_x = c.local;
    item.code_addr = 0x5028;
    const auto code = fx::program(c);
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    auto facts = std::make_shared<ComputeWaveOpFacts>(
        analyze_compute_wave_ops(ins, code.data(), code.size()));
    item.wave_ops = facts;
    return item;
}

}  // namespace

TEST(ComputeWaveAdmission, AnItemWithoutFactsSaysUnanalyzedNotNoCrossLaneOp) {
    ComputeItem item;
    EXPECT_STREQ(compute_wave_route_text(FakeContext{}, item, {32768, 1024}).text, "unanalyzed");
}

TEST(ComputeWaveAdmission, TheRouteTextNamesTheRouteAndTheReason) {
    const fx::Case c{128, fx::Trips::Constant3};
    const auto item = item_for(c, 0);
    // The loop's readlane is a cross-lane operation in a loop: route 4 territory for the analysis.
    EXPECT_STREQ(compute_wave_route_text(FakeContext{}, item, {32768, 1024}).text,
                 "needs-n-lanes:cross-lane-in-loop");
    // Unknown device limits are refused, not assumed: this one is about the budget, so use a
    // top-level program for it.
    ComputeItem plain = item;
    ComputeWaveOpFacts top;
    top.analyzed = true;
    top.ops.push_back({3, ComputeCrossLaneKind::Ballot, ComputeWaveContext::TopLevel, true, 0});
    plain.wave_ops = std::make_shared<ComputeWaveOpFacts>(top);
    EXPECT_STREQ(compute_wave_route_text(FakeContext{}, plain, {0, 0}).text,
                 "refused:shared-memory-limit-unknown");
    EXPECT_STREQ(compute_wave_route_text(FakeContext{}, plain, {32768, 1024}).text,
                 "workgroup-exchange:cross-lane-in-uniform-flow");
}

TEST(ComputeWaveAdmission, ANativeContractRoutesNative) {
    FakeContext ctx;
    ctx.min_native_subgroup_size = 64;
    ctx.max_native_subgroup_size = 64;
    ComputeItem item = item_for({64, fx::Trips::Constant3}, 0);
    item.required_subgroup_size = 64;
    EXPECT_STREQ(compute_wave_route_text(ctx, item, {32768, 1024}).text,
                 "native:host-subgroup-covers-guest-wave");
}

TEST(ComputeWaveAdmission, AnExchangeModuleIsCheckedAgainstTheDeviceLimits) {
    const fx::Case c{128, fx::Trips::Constant3};
    const auto item = item_for(c, 32);
    ASSERT_TRUE(compute_spirv_wave64_exchange(item.spirv));
    EXPECT_EQ(exchange_limit(FakeContext{}, item, {32768, 1024}), nullptr);
    // Not enough workgroup memory for the exchange's scratch: refused, visibly.
    const char* why = exchange_limit(FakeContext{}, item, {512, 1024});
    ASSERT_NE(why, nullptr);
    EXPECT_STREQ(why, "wave64-exchange-limit");
    // A workgroup larger than the device allows.
    EXPECT_NE(exchange_limit(FakeContext{}, item, {32768, 64}), nullptr);
    // Unknown limits are not assumed to be large.
    EXPECT_NE(exchange_limit(FakeContext{}, item, {0, 0}), nullptr);
}

TEST(ComputeWaveAdmission, AnOrdinaryModuleIsNeverHeldToTheExchangeLimits) {
    const fx::Case c{128, fx::Trips::Constant3};
    const auto item = item_for(c, 0);
    ASSERT_FALSE(compute_spirv_wave64_exchange(item.spirv));
    EXPECT_EQ(exchange_limit(FakeContext{}, item, {0, 0}), nullptr)
        << "the limits belong to the exchange module only";
}

TEST(ComputeWaveAdmission, APartialWorkgroupIsRefusedEvenWhenTheModuleIsAnExchange) {
    fx::Case c{128, fx::Trips::Constant3};
    ComputeItem item = item_for(c, 32);
    ASSERT_TRUE(compute_spirv_wave64_exchange(item.spirv));
    item.recompile_config.exact_thread_extent = true;
    item.recompile_config.threads_x = 100;
    item.recompile_config.threads_y = item.recompile_config.threads_z = 1;
    EXPECT_NE(exchange_limit(FakeContext{}, item, {32768, 1024}), nullptr);
}
