// #4230: actual AGC registration -> draw realization -> shipping backend -> original guest MOV/EXP.
// Observe native RGBA32F bytes against independently derived varying-W planes, not host input records.
#include "fixtures/ps_pull_model_fixture.hpp"
#include "fixtures/render_runner.h"
#include <gtest/gtest.h>
#include <cstdio>
#include <cstring>

namespace e = prosper::test::ps_pull;
TEST(PsPullModelExec, OriginalGuestRawExportsHaveVaryingWPlanes) {
    const auto& ctx = prosper::test::render_vk_ctx();
    ASSERT_TRUE(ctx.ok) << "no runtime is not execution evidence";
    ASSERT_TRUE(ctx.geometry_shader_enabled) << "the producing geometry contract must execute";
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(ctx.phys, VK_FORMAT_R32G32B32A32_SFLOAT, &properties);
    ASSERT_TRUE(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT);
    uint32_t attempts = 0, pixels = 0;
    for (const auto& c : e::cases) {
        SCOPED_TRACE(c.name);
        prosper::gpu::DrawItem draw;
        ASSERT_TRUE(e::realize(c, draw));
        ASSERT_FALSE(draw.gs.empty());
        prosper::test::BackendDraw backend;
        backend.vs_shared = draw.vs_shared;
        backend.fs_shared = draw.fs_shared;
        backend.vs = draw.vs;
        backend.gs = draw.gs;
        backend.fs = draw.fs;
        backend.ps = &draw.ps;
        backend.vcount = draw.vertex_count;
        prosper::test::BackendColorTarget target;
        target.load_existing = false;
        target.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        ++attempts;
        std::fprintf(stderr, "[ps-pull-gpu] draw_attempt=%u case=%s native_rgba32f=true\n",
                     attempts, c.name);
        const std::vector draws{backend};
        const auto raw = prosper::test::render_draws_rgba(draws, e::width, e::height, nullptr,
                                                          nullptr, false, &target);
        ASSERT_EQ(raw.size(), size_t(e::width) * e::height * sizeof(float) * 4)
            << "completed native-format raw readback";
        // All sampled centers are strictly interior, away from edge/top-left/clip/sample qualifiers.
        for (uint32_t y = 3; y <= 7; ++y)
            for (uint32_t x = 3; x <= 7; ++x) {
                if (x + y > 11) continue;
                std::array<float, 4> actual{};
                std::memcpy(actual.data(), raw.data() + (size_t(y) * e::width + x) * sizeof(actual),
                            sizeof(actual));
                ++pixels;
                const auto want = e::expected(c, x, y);
                EXPECT_TRUE(e::matches(actual, want))
                    << "pixel " << x << ',' << y << " actual=" << actual[0] << ',' << actual[1]
                    << ',' << actual[2] << ',' << actual[3] << " expected=" << want[0] << ','
                    << want[1] << ',' << want[2] << ',' << want[3];
            }
    }
    EXPECT_EQ(attempts, e::cases.size());
    EXPECT_EQ(pixels, 114u);   // 19 independent interior centers per actual draw
    std::fprintf(stderr,
                 "[ps-pull-gpu] draw_attempts=%u expected=6 raw_pixel_checks=%u expected=114\n",
                 attempts, pixels);
}
