// Mutation: retaining lane-local READFIRST produces a per-pixel x ramp instead of one
// scalar colour. Compare an independent divergent BroadcastFirst/reduction on the same raster.
#include "fixtures/native_pixel_parameter.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstddef>
#include "fixtures/native_fragment_readfirst.hpp"
#include "fixtures/native_readfirst_reference.hpp"
#include "fixtures/render_runner.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

namespace {
std::vector<uint32_t> triangle(bool clipped = false) {
    using namespace prosper::gpu;
    SpirvCompute b;
    b.begin_vertex();
    const auto vertex = b.vertex_invocation_id();
    const auto coordinate = [&](uint32_t index, float low) {
        return b.sel(b.ucmp(Op_IEqual, vertex, b.uconst(index)), b.uconst(fbits(3.0f)),
                     b.uconst(fbits(low)));
    };
    b.export_position(coordinate(1, clipped ? -0.25f : -1.0f), coordinate(2, -1.0f), b.uconst(0),
                      b.uconst(fbits(1.0f)));
    return b.finish();
}
void witness(bool wave32, bool high, bool clipped = false, bool whole_quad = false) {
    namespace fixture = prosper::test::native_fragment_readfirst;
    const auto& context = prosper::test::render_vk_ctx();
    const uint32_t width = wave32 ? 32 : 64;
    if (!context.ok ||
        (!context.fragment_reconvergence.maximal || !context.fragment_reconvergence.full_quads) ||
        !context.subgroup_size_control ||
        !(context.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) ||
        !(context.required_subgroup_size_stages & VK_SHADER_STAGE_FRAGMENT_BIT) ||
        !(context.subgroup_operations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) ||
        !(context.subgroup_operations & VK_SUBGROUP_FEATURE_BALLOT_BIT) ||
        !(context.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) ||
        context.min_subgroup_size > width || context.max_subgroup_size < width)
        GTEST_SKIP() << "device cannot execute the native reconverged fragment reference";
    for (float threshold : {0.0f, 2.5f, 7.5f, 9.5f}) {
        SCOPED_TRACE(threshold);
        const auto result = prosper::test::native_pixel_parameter::compile(
            fixture::program(true, high, threshold, true, whole_quad, wave32), fixture::mapping(),
            wave32, context.float_transport);
        ASSERT_FALSE(result.module.empty()) << result.reason;
        const auto reference = prosper::test::native_readfirst_reference(
            width, threshold, context.float_transport, whole_quad);
        const auto expected =
            prosper::test::render_triangle_rgba(triangle(clipped), reference, 8, 8);
        ASSERT_EQ(expected.size(), 8u * 8u * 4u);
        for (size_t pixel = 0; pixel < 64; ++pixel)
            ASSERT_EQ(expected[pixel * 4 + 3], 255u) << "lane zero must exist for the empty arm";
        const auto actual =
            prosper::test::render_triangle_rgba(triangle(clipped), result.module, 8, 8);
        EXPECT_EQ(actual, expected);
    }
}
} // namespace

TEST(NativeFragmentReadfirstGpu, Native32BroadcastsTheCurrentExecSource) {
    witness(true, false);
}
TEST(NativeFragmentReadfirstGpu, Native64BroadcastsTheCurrentExecSource) {
    witness(false, false);
}
TEST(NativeFragmentReadfirstGpu, Native32PublishesTheHighVccWord) {
    witness(true, true);
}
TEST(NativeFragmentReadfirstGpu, Native64PublishesTheHighVccWord) {
    witness(false, true);
}

TEST(NativeFragmentReadfirstGpu, Native32EdgeKeepsHelpersOutOfInitialExec) {
    witness(true, false, true);
}
TEST(NativeFragmentReadfirstGpu, Native64EdgeKeepsHelpersOutOfInitialExec) {
    witness(false, false, true);
}
TEST(NativeFragmentReadfirstGpu, Native32EdgeWqmIncludesTheHelperSource) {
    witness(true, false, true, true);
}
TEST(NativeFragmentReadfirstGpu, Native64EdgeWqmIncludesTheHelperSource) {
    witness(false, false, true, true);
}
