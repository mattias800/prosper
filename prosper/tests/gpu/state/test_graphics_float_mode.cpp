// Actual hardware launch banks must remain distinct: using a compute/default/other-stage mode
// silently changes shader arithmetic. AMD GC10.3 register offsets and masks are independent oracles.
#include "gpu/state/graphics_float_mode.hpp"
#include "gpu/state/render_state.hpp"
#include "gpu/pm4/command_processor.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
using namespace prosper::gpu;
TEST(GraphicsFloatMode, EachPhysicalBankRetainsEveryModeAndIeeeBit) {
    const std::array stages{GraphicsHardwareStage::Vertex, GraphicsHardwareStage::Geometry,
                            GraphicsHardwareStage::Hull};
    const std::array<uint32_t, 3> registers{0x4a, 0x8a, 0x10a};
    for (size_t stage = 0; stage < stages.size(); ++stage)
        for (uint32_t mode = 0; mode < 256; ++mode)
            for (bool ieee : {false, true}) {
                GpuState state;
                state.sh[registers[stage]] = (mode << 12) | (ieee ? 0x00800000u : 0);
                const auto got = graphics_float_mode(state, stages[stage]);
                EXPECT_TRUE(got.available);
                EXPECT_EQ(got.value, mode);
                EXPECT_EQ(got.ieee_mode, ieee);
                for (size_t other = 0; other < stages.size(); ++other)
                    if (other != stage)
                        EXPECT_EQ(graphics_float_mode(state, stages[other]), GraphicsFloatMode{});
            }
}
TEST(GraphicsFloatMode, AbsentAndObservedZeroAreDifferent) {
    GpuState state;
    EXPECT_EQ(graphics_float_mode(state, GraphicsHardwareStage::Geometry), GraphicsFloatMode{});
    state.sh[0x8a] = 0;
    EXPECT_EQ(graphics_float_mode(state, GraphicsHardwareStage::Geometry),
              (GraphicsFloatMode{true, 0, false}));
    EXPECT_TRUE(GraphicsFloatMode{}.canonical());
    EXPECT_FALSE((GraphicsFloatMode{false, 1, false}.canonical()));
    EXPECT_FALSE((GraphicsFloatMode{false, 0, true}.canonical()));
}
TEST(GraphicsFloatMode, NggAndMergedEsGsUseGsWhileLegacyVertexUsesVs) {
    GpuState state;
    state.sh[0x4a] = 0x00031000;   // VS mode0x31, IEEE off
    state.sh[0x8a] = 0x00882000;   // GS mode0x82, IEEE on
    state.sh[0x10a] = 0x000c3000;   // HS mode0xc3 cannot substitute for either
    // The stage selector itself is unavailable; even two observed banks do not prove a choice.
    EXPECT_EQ(vertex_graphics_float_mode(state), GraphicsFloatMode{});
    for (uint32_t stages : {0u, 0x20u, 0x2000u, 0x2020u}) {
        state.cx[0x2d5] = stages;   // VGT_SHADER_STAGES_EN
        const GraphicsFloatMode expected =
            stages ? GraphicsFloatMode{true, 0x82, true} : GraphicsFloatMode{true, 0x31, false};
        EXPECT_EQ(vertex_graphics_float_mode(state), expected);
        EXPECT_EQ(extract_render_state_uncached(state).vertex_float_mode, expected);
    }
    state.cx[0x2d5] = 0;
    state.sh.erase(0x4a);
    EXPECT_EQ(vertex_graphics_float_mode(state), GraphicsFloatMode{});
}
