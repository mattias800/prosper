// Normal registered generated-GS observations must keep code identity without sharing draw entry.
// Empty-GS warm tests cannot constrain this recurring geometry/helper-plan compilation defect.
#include "fixtures/fragment_raster_fixture.hpp"
#include "gpu/execute/derived_interpolation_geometry.hpp"
#include <gtest/gtest.h>

namespace {
namespace g = prosper::gpu;
namespace f = prosper::test::fragment_draw;
namespace helper = prosper::test::fragment_raster;
namespace p = prosper::agc::Pm4;
class DerivedInterpolationGeometry : public ::testing::Test {
protected:
    void SetUp() override { publish(g::FloatTransportProfile::ExplicitNonFinite32); }
    void TearDown() override { g::reset_float_transport_config_for_test(); }
    static void publish(g::FloatTransportProfile profile) {
        g::reset_float_transport_config_for_test();
        g::publish_float_transport_config({profile});
        g::publish_float_controls_support(true, true);
    }
    static std::vector<uint32_t> original(uint32_t hint) {
        auto code = helper::original(2);
        code.insert(code.begin(),
                    0xbf800000u | hint); // Genuine original S_NOP, not specialization.
        return code;
    }
    static std::vector<std::pair<uint32_t, uint32_t>> context(uint32_t barycentric = 1) {
        auto words = helper::context();
        for (auto& [reg, value] : words)
            if (reg == p::SPI_PS_INPUT_ENA || reg == p::SPI_PS_INPUT_ADDR)
                value = (1u << barycentric) | (1u << 8); // Actual center and POS_X/Y words.
        return words;
    }
    static g::FragmentPacketDeviceContract source_device() {
        // Explicit offline compiler profile: no VkDevice/query/queue or enabled-device proof.
        return {1, true, false,
                g::FragmentPacketRasterDeviceContract{true, 128, 128, 128, 1024, 3, 1, 128}};
    }
};

TEST_F(DerivedInterpolationGeometry, SeventeenRegisteredGeneratedGsObservationsStayWarm) {
    const auto code = original(4399);
    const auto* programs = f::register_programs(code);
    ASSERT_NE(programs, nullptr);
    const auto physical = context();
    std::array<g::DrawItem, 17> draws;   // Retain actual old draws and their private launch owners.
    std::array<std::shared_ptr<const g::FragmentDrawProgramPlan>, 17> plans;
    const auto geometry_before = g::derived_interpolation_geometry_stats();
    const auto plan_before = g::fragment_draw_cache_stats();
    for (uint32_t observation = 0; observation < draws.size(); ++observation) {
        auto color = f::color_a;
        color[0] = float(observation + 1) / 64.f;
        ASSERT_TRUE(f::realize_registered(draws[observation], *programs, color, f::ieee_rsrc1, 15,
                                          physical));
        const auto& inputs = draws[observation].fragment_draw_inputs;
        ASSERT_TRUE(inputs && inputs->launch_source && inputs->source_gs);
        ASSERT_FALSE(draws[observation].gs.empty()) << "the normal realizer must generate GS";
        EXPECT_EQ(*inputs->source_gs, draws[observation].gs);
        EXPECT_EQ(*inputs->raw_code, code);
        EXPECT_TRUE(inputs->generated_interpolation_geometry);
        EXPECT_TRUE(inputs->launch_source->matches(*inputs));
        const auto prepared = g::prepare_fragment_packet_inputs(inputs, true);
        ASSERT_TRUE(prepared && prepared->launch_source);
        plans[observation] =
            g::cached_fragment_draw_program(*inputs, *prepared, source_device(), 16);
        ASSERT_TRUE(plans[observation]);
        if (observation) {
            const auto& first = *draws[0].fragment_draw_inputs;
            EXPECT_EQ(inputs->raw_code, first.raw_code);
            EXPECT_EQ(inputs->source_gs, first.source_gs);
            EXPECT_FALSE(inputs->source_gs.owner_before(first.source_gs));
            EXPECT_FALSE(first.source_gs.owner_before(inputs->source_gs));
            EXPECT_EQ(plans[observation], plans[0]);
            EXPECT_NE(inputs->entry, first.entry);
            EXPECT_NE(inputs->launch_source, first.launch_source);
            EXPECT_FALSE(first.launch_source->matches(*inputs));
            EXPECT_FALSE(inputs->launch_source->matches(first));
        }
    }
    const auto geometry_after = g::derived_interpolation_geometry_stats();
    const auto plan_after = g::fragment_draw_cache_stats();
    EXPECT_EQ(geometry_after.compile_calls - geometry_before.compile_calls, 1u);
    EXPECT_EQ(geometry_after.cache_hits - geometry_before.cache_hits, 16u);
    EXPECT_EQ(plan_after.program_compile_calls - plan_before.program_compile_calls, 1u);
    EXPECT_EQ(plan_after.program_hits - plan_before.program_hits, 16u);
    // Named-refused profiles must remain warm too, without turning cache residence into readiness.
    EXPECT_EQ(plans.back()->rejection_reason(), plans.front()->rejection_reason());
}

TEST_F(DerivedInterpolationGeometry, ChangedOriginalLayoutAndTransportKeepDistinctGsGenerations) {
    const auto* first_programs = f::register_programs(original(4400));
    const auto* changed_programs = f::register_programs(original(4401));
    ASSERT_NE(first_programs, nullptr);
    ASSERT_NE(changed_programs, nullptr);
    std::array<g::DrawItem, 4> draws;
    const auto center = context(), linear_center = context(4);
    const auto before = g::derived_interpolation_geometry_stats();
    ASSERT_TRUE(
        f::realize_registered(draws[0], *first_programs, f::color_a, f::ieee_rsrc1, 15, center));
    ASSERT_TRUE(
        f::realize_registered(draws[1], *changed_programs, f::color_a, f::ieee_rsrc1, 15, center));
    ASSERT_TRUE(f::realize_registered(draws[2], *first_programs, f::color_a, f::ieee_rsrc1, 15,
                                      linear_center));
    publish(g::FloatTransportProfile::Implicit);
    ASSERT_TRUE(
        f::realize_registered(draws[3], *first_programs, f::color_a, f::ieee_rsrc1, 15, center));
    for (uint32_t index = 0; index < draws.size(); ++index) {
        const auto& inputs = draws[index].fragment_draw_inputs;
        ASSERT_TRUE(inputs && inputs->launch_source && inputs->source_gs);
        ASSERT_FALSE(draws[index].gs.empty());
        EXPECT_TRUE(inputs->launch_source->matches(*inputs));
        for (uint32_t prior = 0; prior < index; ++prior) {
            const auto& old = *draws[prior].fragment_draw_inputs;
            EXPECT_NE(inputs->source_gs, old.source_gs);
            EXPECT_FALSE(old.launch_source->matches(*inputs));
            EXPECT_FALSE(inputs->launch_source->matches(old));
        }
    }
    EXPECT_EQ(g::derived_interpolation_geometry_stats().compile_calls - before.compile_calls, 4u);
    const auto& a = *draws[0].fragment_draw_inputs;
    const auto& b = *draws[2].fragment_draw_inputs;
    EXPECT_EQ(a.raw_code, b.raw_code);
    EXPECT_NE(a.interpolation.system_locations, b.interpolation.system_locations);
    EXPECT_NE(a.float_transport.profile, draws[3].fragment_draw_inputs->float_transport.profile);
    // Isolate profile fields from any native VS key change. Both arguments come from actual
    // normal observations of this same registration; the service returns SOURCE, not permission.
    const auto analysis = g::acquire_shader_analysis(
        reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(draws[0].fs_guest_addr)),
        a.raw_code->size());
    ASSERT_TRUE(analysis && draws[0].vs_shared);
    ASSERT_EQ(g::shader_analysis_owned_words(analysis), a.raw_code);
    const auto isolated_before = g::derived_interpolation_geometry_stats();
    const auto same = g::acquire_derived_interpolation_geometry(
        analysis, draws[0].vs_shared, a.interpolation, false, false, a.float_transport);
    EXPECT_EQ(same, a.source_gs);
    const auto changed_layout = g::acquire_derived_interpolation_geometry(
        analysis, draws[0].vs_shared, b.interpolation, false, false, a.float_transport);
    const auto changed_transport = g::acquire_derived_interpolation_geometry(
        analysis, draws[0].vs_shared, a.interpolation, false, false,
        draws[3].fragment_draw_inputs->float_transport);
    const auto diagnostic = g::acquire_derived_interpolation_geometry(
        analysis, draws[0].vs_shared, a.interpolation, true, false, a.float_transport);
    const auto rect = g::acquire_derived_interpolation_geometry(
        analysis, draws[0].vs_shared, a.interpolation, false, true, a.float_transport);
    ASSERT_TRUE(changed_layout && changed_transport);
    ASSERT_TRUE(diagnostic && rect);
    EXPECT_FALSE(changed_layout->empty());
    EXPECT_FALSE(changed_transport->empty());
    EXPECT_FALSE(diagnostic->empty());
    EXPECT_FALSE(rect->empty());
    EXPECT_NE(changed_layout, same);
    EXPECT_NE(changed_transport, same);
    EXPECT_NE(diagnostic, same);
    EXPECT_NE(rect, same);
    EXPECT_NE(diagnostic, rect);
    EXPECT_EQ(g::derived_interpolation_geometry_stats().compile_calls -
                  isolated_before.compile_calls,
              4u);
    EXPECT_EQ(g::derived_interpolation_geometry_stats().cache_hits - isolated_before.cache_hits,
              1u);
}
}   // namespace
