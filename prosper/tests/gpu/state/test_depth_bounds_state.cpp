// test_depth_bounds_state -- DB_DEPTH_CONTROL.DEPTH_BOUNDS_ENABLE and DB_DEPTH_BOUNDS_MIN/MAX survive
// the PM4 fold into RenderState and resolve into the pipeline state the backend consumes. UE4 limits
// each shadow cascade's projection and each deferred light to its depth slice with this test; before
// it was decoded, Kena's three cascade projections each covered every pixel (KENA_STATUS.md).
#include "gpu/pm4/command_processor.hpp"
#include "gpu/state/render_state.hpp"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>

using namespace prosper::gpu;
namespace {
// Literal context offsets (AMD gc_10_3: DB_DEPTH_BOUNDS_MIN 0xA008, MAX 0xA009, DB_DEPTH_CONTROL
// 0xA200), independent of the producer's pm4_registers.hpp constants.
constexpr uint32_t kDbDepthBoundsMin = 0x008, kDbDepthBoundsMax = 0x009, kDbDepthControl = 0x200;
constexpr uint32_t kDepthBoundsEnable = 1u << 3, kZEnable = 1u << 1;

uint32_t bits(float value) {
    uint32_t out = 0;
    std::memcpy(&out, &value, sizeof out);
    return out;
}

void write_context(GpuState& state, uint32_t offset, uint32_t value) {
    const uint32_t words[]{0xc0016900u, offset, value};   // SET_CONTEXT_REG, one register
    size_t consumed = 0;
    EXPECT_EQ(run_command_buffer(words, std::size(words), state, &consumed), 1u);
    EXPECT_EQ(consumed, std::size(words));
}

ResolvedPipelineState resolve(uint32_t depth_control, float min, float max) {
    GpuState state;
    write_context(state, kDbDepthControl, depth_control);
    write_context(state, kDbDepthBoundsMin, bits(min));
    write_context(state, kDbDepthBoundsMax, bits(max));
    return resolve_pipeline_state(extract_render_state(state));
}
}   // namespace

TEST(DepthBoundsState, EnableAndBoundsDecodeWithoutTheDepthTest) {
    // Kena's far-cascade projection: DB_DEPTH_CONTROL=0x8 (bounds only, Z_ENABLE clear).
    GpuState state;
    write_context(state, kDbDepthControl, kDepthBoundsEnable);
    write_context(state, kDbDepthBoundsMin, bits(0.25f));
    write_context(state, kDbDepthBoundsMax, bits(0.75f));
    const RenderState rs = extract_render_state(state);
    EXPECT_TRUE(rs.depth_bounds_enable);
    EXPECT_FALSE(rs.z_enable);
    EXPECT_EQ(rs.depth_bounds_min, 0.25f);
    EXPECT_EQ(rs.depth_bounds_max, 0.75f);

    const ResolvedPipelineState ps = resolve_pipeline_state(rs);
    EXPECT_TRUE(ps.depth_bounds_enable);
    EXPECT_FALSE(ps.depth_test_enable);
    EXPECT_EQ(ps.depth_bounds_min, 0.25f);
    EXPECT_EQ(ps.depth_bounds_max, 0.75f);
    // The test reads the stored depth, so the pass must bind the attachment although no depth
    // test, write or clear is enabled.
    EXPECT_TRUE(uses_depth_stencil_attachment(ps));
}

TEST(DepthBoundsState, ClearEnableBitLeavesTheTestOff) {
    // Programmed bounds without DEPTH_BOUNDS_ENABLE (a depth test only) must not enable the test.
    const ResolvedPipelineState ps = resolve(kZEnable, 0.25f, 0.75f);
    EXPECT_FALSE(ps.depth_bounds_enable);
    EXPECT_TRUE(ps.depth_test_enable);
    EXPECT_EQ(ps.depth_bounds_min, 0.0f);
    EXPECT_EQ(ps.depth_bounds_max, 1.0f);

    ResolvedPipelineState none = resolve(0u, 0.25f, 0.75f);
    EXPECT_FALSE(none.depth_bounds_enable);
    EXPECT_FALSE(uses_depth_stencil_attachment(none));
}

TEST(DepthBoundsState, BoundsClampIntoTheVulkanRangeAndNanIsEmpty) {
    const ResolvedPipelineState wide = resolve(kDepthBoundsEnable, -2.0f, 3.0f);
    EXPECT_EQ(wide.depth_bounds_min, 0.0f);
    EXPECT_EQ(wide.depth_bounds_max, 1.0f);

    // A NaN bound compares false on hardware: no stored depth passes. It must become an empty
    // range, never the full [0, 1] that a naive clamp of NaN could produce.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const ResolvedPipelineState empty_min = resolve(kDepthBoundsEnable, nan, 0.5f);
    EXPECT_TRUE(empty_min.depth_bounds_enable);
    EXPECT_GT(empty_min.depth_bounds_min, empty_min.depth_bounds_max);
    const ResolvedPipelineState empty_max = resolve(kDepthBoundsEnable, 0.5f, nan);
    EXPECT_GT(empty_max.depth_bounds_min, empty_max.depth_bounds_max);
    EXPECT_FALSE(std::isnan(empty_max.depth_bounds_min) || std::isnan(empty_max.depth_bounds_max));
}
