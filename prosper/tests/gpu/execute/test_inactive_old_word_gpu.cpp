// Mutation: refusing or zeroing a pure masked overwrite loses its independently expected pixels.
#include "fixtures/inactive_old_word.hpp"
#include "fixtures/native_pixel_parameter.hpp"
#include "fixtures/render_runner.h"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {
std::vector<uint32_t> triangle(float red) {
    using namespace prosper::gpu;
    SpirvCompute b;
    b.begin_vertex();
    const auto vertex = b.vertex_invocation_id();
    const auto coordinate = [&](uint32_t index) {
        return b.sel(b.ucmp(Op_IEqual, vertex, b.uconst(index)), b.uconst(fbits(3.0f)),
                     b.uconst(fbits(-1.0f)));
    };
    b.export_position(coordinate(1), coordinate(2), b.uconst(0), b.uconst(fbits(1.0f)));
    b.export_param(0, b.uconst(fbits(red)), b.uconst(0), b.uconst(0), b.uconst(fbits(1.0f)));
    return b.finish();
}
void witness(prosper::test::inactive_old_word::Writer writer, bool wave32) {
    namespace fixture = prosper::test::inactive_old_word;
    const auto& context = prosper::test::render_vk_ctx();
    const uint32_t width = wave32 ? 32u : 64u;
    if (!context.ok || !context.subgroup_size_control ||
        !(context.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) ||
        !(context.required_subgroup_size_stages & VK_SHADER_STAGE_FRAGMENT_BIT) ||
        !(context.subgroup_operations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) ||
        context.min_subgroup_size > width || context.max_subgroup_size < width)
        GTEST_SKIP() << "device cannot require the fixture's native fragment width";
    for (float threshold : {0.0f, 2.5f, 9.5f}) {
        const auto result = prosper::test::native_pixel_parameter::compile(
            fixture::program(writer, false, false, threshold), fixture::mapping(), wave32,
            context.float_transport);
        ASSERT_FALSE(result.module.empty()) << result.reason;
        for (float red : {0.125f, 0.625f}) {
            const auto pixels =
                prosper::test::render_triangle_rgba(triangle(red), result.module, 8, 8);
            ASSERT_EQ(pixels.size(), 8u * 8u * 4u);
            for (size_t pixel = 0; pixel < 64; ++pixel) {
                const float x = static_cast<float>(pixel % 8) + 0.5f;
                const float written = writer == fixture::Writer::Interpolate ? red : x * 0.125f;
                const float wanted = x >= threshold ? written : 0.25f;
                for (size_t channel = 0; channel < 3; ++channel)
                    EXPECT_NEAR(pixels[pixel * 4 + channel], wanted * 255.0f, 1.0f);
                EXPECT_EQ(pixels[pixel * 4 + 3], 255u);
            }
        }
    }
}
} // namespace

TEST(InactiveOldWordGpu, Native32InterpolationOverwritesOnlyItsCurrentLanes) {
    witness(prosper::test::inactive_old_word::Writer::Interpolate, true);
}
TEST(InactiveOldWordGpu, Native64InterpolationOverwritesOnlyItsCurrentLanes) {
    witness(prosper::test::inactive_old_word::Writer::Interpolate, false);
}
TEST(InactiveOldWordGpu, Native32MultiplyOverwritesOnlyItsCurrentLanes) {
    witness(prosper::test::inactive_old_word::Writer::Multiply, true);
}
TEST(InactiveOldWordGpu, Native64MultiplyOverwritesOnlyItsCurrentLanes) {
    witness(prosper::test::inactive_old_word::Writer::Multiply, false);
}
