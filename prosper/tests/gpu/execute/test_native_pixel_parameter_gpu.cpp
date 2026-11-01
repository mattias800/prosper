// The automatic parameter-base service must read genuine rasterized native Input values.
// A constant M0/SGPR stand-in cannot supply the two distinct PARAM0 colours observed here.
#include "fixtures/native_pixel_parameter.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "fixtures/render_runner.h"
#include <gtest/gtest.h>
#include <bit>

namespace {
using namespace prosper::gpu;
std::vector<uint32_t> triangle(float red) {
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
} // namespace

TEST(NativePixelParameterGpu, NativeInterpolationAndTheReusedScalarReachTheAttachment) {
    namespace fixture = prosper::test::native_pixel_parameter;
    for (const bool wave32 : {false, true})
        for (const bool smooth : {false, true}) {
            const auto result = fixture::compile(fixture::program(30, smooth),
                                                 fixture::mapping(30, smooth), wave32);
            ASSERT_FALSE(result.module.empty()) << result.reason;
            for (const float red : {0.125f, 0.625f}) {
                const auto pixels =
                    prosper::test::render_triangle_rgba(triangle(red), result.module, 8, 8);
                ASSERT_EQ(pixels.size(), 8u * 8u * 4u);
                for (size_t pixel = 0; pixel < 64; ++pixel) {
                    EXPECT_NEAR(pixels[pixel * 4], red * 255.0f, 1.0f);
                    EXPECT_NEAR(pixels[pixel * 4 + 1], 127.5f, 1.0f);
                    EXPECT_EQ(pixels[pixel * 4 + 2], pixels[pixel * 4]);
                    EXPECT_EQ(pixels[pixel * 4 + 3], 255u);
                }
            }
        }
}
