// ADR 0028 route 3, execution half: a Wave64 compute program whose v_readlane sits in a loop is
// compiled through the exact exchange dispatcher and DISPATCHED on this machine's GPU. Expected
// values come from the closed-form oracle in the fixture (a sum of integers), not from a native
// 64-lane run: no 64-lane reference device exists on a 32-lane host, and the limit of this test is
// stated in the PR that adds it.
//
// What it does and does not show: it shows the exchange result equals the mathematically expected
// value on whichever subgroup width the local device has (32 on NVIDIA, so each guest wave spans
// two host subgroups). It does not show bit equality with RADV's native Wave64.
#include <gtest/gtest.h>

#include <cstdio>
#include <vector>

#include "fixtures/compute_runner.h"
#include "fixtures/wave64_exchange_fixture.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

namespace fx = prosper::test::wave64_exchange;
using namespace prosper::gpu;

namespace {

std::vector<uint32_t> run(const fx::Case& c, const std::vector<uint32_t>& module) {
    std::vector<uint32_t> output;
    const auto ran = prosper::test::run_compute(module, std::vector<float>(c.local), c.local,
                                                c.local, {}, fx::input(c), &output, c.local);
    if (ran.empty() && output.empty())
        ADD_FAILURE() << "the dispatch did not run (no Vulkan device?)";
    return output;
}

void expect_exact(const fx::Case& c, const std::vector<uint32_t>& got) {
    const auto want = fx::expected(c);
    ASSERT_EQ(got.size(), want.size());
    for (uint32_t lane = 0; lane < c.local; ++lane)
        EXPECT_EQ(got[c.local * 3 + lane], want[c.local * 3 + lane]) << "lane " << lane;
    for (uint32_t i = 0; i < c.local * 3; ++i)
        EXPECT_EQ(got[i], want[i]) << "the inputs must not be disturbed, word " << i;
}

}  // namespace

TEST(Wave64ExchangeExecution, ReadLaneInAUniformLoopOverTwoGuestWaves) {
    const fx::Case c{128, fx::Trips::Constant3};
    const auto module = fx::compile(c, /*exchange_width=*/32);
    ASSERT_FALSE(module.empty());
    ASSERT_TRUE(compute_spirv_wave64_exchange(module)) << "the arm must exercise the exchange";
    expect_exact(c, run(c, module));
}

TEST(Wave64ExchangeExecution, ReadLaneInALoopWhoseTripCountDiffersPerGuestWave) {
    // Wave 0 iterates once, wave 1 twice, wave 2 three times, wave 3 four times. Peers that finish
    // early must keep servicing the exchange for the waves that have not.
    const fx::Case c{256, fx::Trips::PerWave};
    const auto module = fx::compile(c, /*exchange_width=*/32);
    ASSERT_FALSE(module.empty());
    ASSERT_TRUE(compute_spirv_wave64_exchange(module));
    expect_exact(c, run(c, module));
}

TEST(Wave64ExchangeExecution, OneGuestWaveIsExactToo) {
    const fx::Case c{64, fx::Trips::Constant3};
    const auto module = fx::compile(c, /*exchange_width=*/32);
    ASSERT_FALSE(module.empty());
    ASSERT_TRUE(compute_spirv_wave64_exchange(module));
    expect_exact(c, run(c, module));
}

TEST(Wave64ExchangeExecution, AGuestBarrierInsideTheLoopStillComputesExactly) {
    // The dispatcher accepted this shape (it hoists the barrier into its common phase); if its
    // result were not exact this arm would have to become a refusal instead.
    fx::Case c{128, fx::Trips::Constant3};
    c.barrier_in_loop = true;
    const auto module = fx::compile(c, /*exchange_width=*/32);
    ASSERT_FALSE(module.empty());
    ASSERT_TRUE(compute_spirv_wave64_exchange(module));
    expect_exact(c, run(c, module));
}

TEST(Wave64ExchangeExecution, TheOracleRejectsAWrongLane) {
    // Positive control for the oracle itself, built by hand: reading lane 6 instead of lane 5 of
    // the same wave must NOT equal the expectation, so a lowering that picked the wrong source
    // lane cannot pass by accident.
    const fx::Case c{64, fx::Trips::Constant3};
    auto want = fx::expected(c);
    uint32_t wrong = 0;
    for (uint32_t i = 0; i < 3; ++i) wrong += fx::data(6) + i;
    EXPECT_NE(want[c.local * 3], wrong);
    EXPECT_EQ(want[c.local * 3], 3u * fx::data(5) + 3u) << "3 * data(5) + (0 + 1 + 2)";
}

TEST(Wave64ExchangeExecution, ControlArmTheOrdinaryLoweringIsWrongOnANarrowHost) {
    // The route-OFF module needs a 64-lane subgroup. On a host with narrower subgroups, running it
    // anyway reads lane 5 of the HOST subgroup, not of the guest wave -- which is why the backend
    // declines it. This arm shows the exchange arm's equality is not a property of the fixture.
    const auto host = prosper::test::default_compute_subgroup_properties();
    if (!host.size || host.size >= 64) GTEST_SKIP() << "host subgroup is " << host.size << " lanes";
    const fx::Case c{128, fx::Trips::Constant3};
    const auto module = fx::compile(c, /*exchange_width=*/0);
    ASSERT_FALSE(module.empty());
    ASSERT_EQ(compute_spirv_min_subgroup_size(module), 64u);
    EXPECT_NE(run(c, module), fx::expected(c));
}
