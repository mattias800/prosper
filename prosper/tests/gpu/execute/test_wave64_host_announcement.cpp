// The renderer says at start-up whether this device can run guest Wave64 fragment programs at
// their own width, and the line agrees with the facts the admission test reads.
//
// This is the PRODUCTION route of fragment_wave64_host_line (test_wave64_host_line pins only its
// wording). It passes on any device: a 64-lane one must be reported NATIVE, a narrower one --
// lavapipe in CI is fixed at 8 lanes -- must be warned.
#include "diagnostics/perf/wave64_refusal.hpp"
#include "fixtures/render_runner.h"
#include <gtest/gtest.h>
#include <string>

TEST(Wave64HostAnnouncement, RendererStartUpPrintsTheLineForThisDevice) {
    testing::internal::CaptureStderr();
    const auto& ctx = prosper::test::render_vk_ctx();   // first use in this process creates it
    const std::string log = testing::internal::GetCapturedStderr();
    if (!ctx.dev) GTEST_SKIP() << "no Vulkan device";

    const prosper::diagnostics::perf::FragmentWave64Host host{
        ctx.subgroup_size_control,
        (ctx.required_subgroup_size_stages & VK_SHADER_STAGE_FRAGMENT_BIT) != 0,
        (ctx.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) != 0, ctx.min_subgroup_size,
        ctx.max_subgroup_size};
    const std::string expected = prosper::diagnostics::perf::fragment_wave64_host_line(host);
    const size_t at = log.find(expected);
    EXPECT_NE(at, std::string::npos) << "start-up log lacks:\n" << expected << "\ngot:\n" << log;
    if (at != std::string::npos)
        EXPECT_EQ(log.find(expected, at + 1), std::string::npos) << "printed more than once";
    // Exactly one direction, and the one the device's own facts select.
    const bool native_line =
        log.find("guest Wave64 fragment programs: NATIVE") != std::string::npos;
    const bool warning_line =
        log.find("WARNING: this GPU cannot run PS5 Wave64 fragment programs") != std::string::npos;
    EXPECT_NE(native_line, warning_line) << log;
    EXPECT_EQ(native_line, host.native()) << log;
}
