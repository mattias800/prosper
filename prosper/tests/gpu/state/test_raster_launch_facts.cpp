// Raw producing raster controls must survive the PM4 fold and draw snapshots. Dropping presence
// makes unwritten state indistinguishable from a programmed zero and cannot authorize guest launch.
#include "gpu/pm4/command_processor.hpp"
#include "gpu/state/render_state.hpp"
#include <array>
#include <gtest/gtest.h>

using namespace prosper::gpu;
namespace {
struct Control {
    uint32_t offset;
    bool RasterLaunchFacts::* available;
    uint32_t RasterLaunchFacts::* value;
};
// Literal context offsets and raw-word expectations are independent of PM4_FIELD/producer constants.
constexpr std::array<Control, 5> controls{{
    {0x310, &RasterLaunchFacts::sc_shader_control_available, &RasterLaunchFacts::sc_shader_control},
    {0x292, &RasterLaunchFacts::sc_mode_cntl_0_available, &RasterLaunchFacts::sc_mode_cntl_0},
    {0x293, &RasterLaunchFacts::sc_mode_cntl_1_available, &RasterLaunchFacts::sc_mode_cntl_1},
    {0x2f8, &RasterLaunchFacts::sc_aa_config_available, &RasterLaunchFacts::sc_aa_config},
    {0x203, &RasterLaunchFacts::db_shader_control_available, &RasterLaunchFacts::db_shader_control},
}};
void write_control(GpuState& state, uint32_t offset, uint32_t value, uint32_t op = 0x69) {
    const uint32_t words[]{0xc0010000u | (op << 8), offset, value};
    size_t consumed = 0;
    EXPECT_EQ(run_command_buffer(words, std::size(words), state, &consumed), 1u);
    EXPECT_EQ(consumed, std::size(words));
}
}   // namespace

TEST(RasterLaunchFacts, MissingControlsStayUnavailableWithoutRegisterInsertion) {
    GpuState state;
    const auto facts = extract_render_state(state).ps_raster_launch;
    EXPECT_EQ(facts, RasterLaunchFacts{});
    EXPECT_TRUE(facts.canonical());
    EXPECT_TRUE(state.cx.empty());
}

TEST(RasterLaunchFacts, EachPresentZeroHasIndependentKnownness) {
    for (const auto& written : controls) {
        GpuState state;
        write_control(state, written.offset, 0);
        const auto facts = extract_render_state(state).ps_raster_launch;
        for (const auto& control : controls) {
            EXPECT_EQ(facts.*control.available, control.offset == written.offset);
            EXPECT_EQ(facts.*control.value, 0u);
        }
        EXPECT_TRUE(facts.canonical());
        EXPECT_NE(facts, RasterLaunchFacts{});
        EXPECT_EQ(state.cx.size(), 1u);
    }
}

TEST(RasterLaunchFacts, EveryRawBitSurvivesTheProducingPm4Fold) {
    for (uint32_t bit = 0; bit < 32; ++bit) {
        GpuState state;
        for (size_t i = 0; i < controls.size(); ++i)
            write_control(state, controls[i].offset,
                          (uint32_t{1} << bit) ^ uint32_t(i * 0x13579bdu));
        const auto facts = extract_render_state(state).ps_raster_launch;
        for (size_t i = 0; i < controls.size(); ++i) {
            EXPECT_TRUE(facts.*controls[i].available);
            EXPECT_EQ(facts.*controls[i].value, (uint32_t{1} << bit) ^ uint32_t(i * 0x13579bdu));
        }
        EXPECT_TRUE(facts.canonical());
    }
}

TEST(RasterLaunchFacts, ShaderFileWritesCannotBecomeContextLaunchFacts) {
    GpuState state;
    for (const auto& control : controls) write_control(state, control.offset, UINT32_MAX, 0x76);
    EXPECT_EQ(extract_render_state(state).ps_raster_launch, RasterLaunchFacts{});
    EXPECT_TRUE(state.cx.empty());
    EXPECT_EQ(state.sh.size(), controls.size());
}

TEST(RasterLaunchFacts, DrawSnapshotsKeepTheirOwnProducingWords) {
    GpuState state;
    for (const auto& control : controls) write_control(state, control.offset, 0x81234567u);
    const uint32_t draw[]{0xc0011010u, 3u, 0u};   // actual three-word DrawIndexAuto packet
    ASSERT_EQ(run_command_buffer(draw, std::size(draw), state), 1u);
    for (const auto& control : controls) write_control(state, control.offset, 0u);
    ASSERT_EQ(run_command_buffer(draw, std::size(draw), state), 1u);
    ASSERT_EQ(state.draws.size(), 2u);
    ASSERT_TRUE(state.draws[0].state);
    ASSERT_TRUE(state.draws[1].state);
    const auto first = extract_render_state(*state.draws[0].state).ps_raster_launch;
    const auto second = extract_render_state(*state.draws[1].state).ps_raster_launch;
    for (const auto& control : controls) {
        EXPECT_TRUE(first.*control.available);
        EXPECT_EQ(first.*control.value, 0x81234567u);
        EXPECT_TRUE(second.*control.available);
        EXPECT_EQ(second.*control.value, 0u);
    }
    EXPECT_NE(first, second);
}

TEST(RasterLaunchFacts, UnobservedPayloadIsNotCanonical) {
    for (const auto& control : controls) {
        RasterLaunchFacts facts;
        facts.*control.value = 1u;
        EXPECT_FALSE(facts.canonical());
        facts.*control.available = true;
        EXPECT_TRUE(facts.canonical());
        facts.*control.value = 0u;
        EXPECT_TRUE(facts.canonical());
    }
}
