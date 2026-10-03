// SOURCE-only original producing joins. The explicit offline feature profile below is a test
// compiler input, NOT evidence that a real VkDevice enabled it or an attachment was committed.
#include "fixtures/fragment_draw_fixture.hpp"
#include "fixtures/fragment_draw_source.hpp"
#include <gtest/gtest.h>

namespace {
namespace g = prosper::gpu;
namespace f = prosper::test::fragment_draw;
void retain_plan(const g::FragmentDrawProgramPlan& plan) {
    f::retain_source(plan.collect_words(), "collector");
    f::retain_source(plan.count_words(), "count");
    f::retain_source(plan.assembly_words(), "assembly");
    if (plan.capacity_owner())
        f::retain_source(plan.capacity_owner()->kernel()->program.packet.spirv, "original_ps");
    f::retain_source(plan.validation_words(), "validation");
    f::retain_source(plan.replay_words(), "replay");
}
class FragmentDrawPlan : public ::testing::Test {
protected:
    void SetUp() override {
        g::reset_float_transport_config_for_test();
        g::publish_float_transport_config({g::FloatTransportProfile::ExplicitNonFinite32});
        g::publish_float_controls_support(true, true);
    }
    void TearDown() override { g::reset_float_transport_config_for_test(); }
    const g::FragmentPacketDeviceContract source_device{0x1234, true, false};
};
TEST_F(FragmentDrawPlan, SameOriginalCachedCodeConsumesDistinctCurrentUserWords) {
    g::DrawItem a, b;
    ASSERT_TRUE(f::realize(a, f::color_a));
    ASSERT_TRUE(f::realize(b, f::color_b));
    const auto pa = f::prepare(a), pb = f::prepare(b);
    ASSERT_TRUE(pa && pb && a.fragment_draw_inputs && b.fragment_draw_inputs);
    ASSERT_EQ(*a.fragment_draw_inputs->raw_code, f::fragment_words());
    for (const auto& input : {a.fragment_draw_inputs, b.fragment_draw_inputs}) {
        ASSERT_TRUE(input->launch_rsrc1.available && input->float_flags.available);
        EXPECT_EQ(input->launch_rsrc1.value, f::ieee_rsrc1);
        EXPECT_TRUE(input->float_flags.ieee_mode);
        EXPECT_EQ((input->launch_rsrc1.value >> 23) & 1u, uint32_t(input->float_flags.ieee_mode));
    }
    EXPECT_FALSE(pa->ready) << "generic launch preparation still does not grant a transaction";
    const auto code_a =
        g::cached_fragment_draw_program(*a.fragment_draw_inputs, *pa, source_device, 48);
    const auto code_b =
        g::cached_fragment_draw_program(*b.fragment_draw_inputs, *pb, source_device, 48);
    retain_plan(*code_a);
    ASSERT_TRUE(code_a->rejection_reason().empty()) << code_a->rejection_reason();
    ASSERT_TRUE(code_b->rejection_reason().empty()) << code_b->rejection_reason();
    EXPECT_EQ(code_a, code_b) << "scalar VALUES are not compilation/profile keys";
    const auto ta = g::instantiate_fragment_draw_transaction(
        code_a, a.fragment_draw_inputs, *pa, f::width, f::height, 1, source_device.device_identity);
    const auto tb = g::instantiate_fragment_draw_transaction(
        code_b, b.fragment_draw_inputs, *pb, f::width, f::height, 1, source_device.device_identity);
    ASSERT_TRUE(ta.rejection().empty()) << ta.rejection();
    ASSERT_TRUE(tb.rejection().empty()) << tb.rejection();
    EXPECT_NE(ta.entry_words(), tb.entry_words());
    EXPECT_EQ(code_a->capacity_owner()->kernel()->guest_code, f::fragment_words());
    EXPECT_EQ(code_a->capacity_owner()->kernel()->program.packet.export_observation,
              g::FragmentPacketExportObservation::Architectural);
    ASSERT_EQ(code_a->capacity_owner()->export_sites().size(), 1u);
    EXPECT_EQ(code_a->capacity_owner()->export_sites().front(),
              (g::FragmentPacketExportSite{5, 0, 15, 0, 1, 1}));
    EXPECT_EQ(code_a->collector_shape().fields.size(), 0u);
    EXPECT_FALSE(code_a->replay_words().empty());
    EXPECT_EQ(ta.producing_inputs(), a.fragment_draw_inputs);
    EXPECT_EQ(tb.producing_inputs(), b.fragment_draw_inputs);

    g::DrawItem mode0;
    ASSERT_TRUE(f::realize(mode0, f::color_a, f::fragment_words(), 0u));
    const auto mode0_prepared = f::prepare(mode0);
    ASSERT_TRUE(mode0_prepared && mode0.fragment_draw_inputs);
    ASSERT_EQ(*mode0.fragment_draw_inputs->raw_code, f::fragment_words());
    ASSERT_TRUE(mode0.fragment_draw_inputs->launch_rsrc1.available &&
                mode0.fragment_draw_inputs->float_flags.available);
    EXPECT_EQ(mode0.fragment_draw_inputs->launch_rsrc1.value, 0u);
    EXPECT_FALSE(mode0.fragment_draw_inputs->float_flags.ieee_mode);
    EXPECT_EQ((mode0.fragment_draw_inputs->launch_rsrc1.value >> 23) & 1u,
              uint32_t(mode0.fragment_draw_inputs->float_flags.ieee_mode));
    const auto refused = g::compile_fragment_draw_program(*mode0.fragment_draw_inputs,
                                                          *mode0_prepared, source_device, 48);
    f::retain_source(refused.collect_words(), "non_ieee_collector");
    EXPECT_EQ(refused.rejection_reason(), "packet-f32-non-ieee-mode-unimplemented");
    EXPECT_FALSE(refused.capacity_owner());
    EXPECT_TRUE(refused.validation_words().empty());
    EXPECT_TRUE(refused.replay_words().empty());
}
TEST_F(FragmentDrawPlan, ForeignProducerDeviceAndDynamicPrefixCannotBorrowAuthority) {
    g::DrawItem draw;
    ASSERT_TRUE(f::realize(draw));
    const auto prepared = f::prepare(draw);
    ASSERT_TRUE(prepared && draw.fragment_draw_inputs);
    const auto plan =
        g::cached_fragment_draw_program(*draw.fragment_draw_inputs, *prepared, source_device, 48);
    retain_plan(*plan);
    ASSERT_TRUE(plan->rejection_reason().empty()) << plan->rejection_reason();
    auto foreign = std::make_shared<g::RasterQuadInputs>(*draw.fragment_draw_inputs);
    EXPECT_EQ(g::instantiate_fragment_draw_transaction(plan, foreign, *prepared, f::width,
                                                       f::height, 1, source_device.device_identity)
                  .rejection(),
              "fragment-draw-producing-profile-mismatch");
    EXPECT_EQ(g::instantiate_fragment_draw_transaction(plan, draw.fragment_draw_inputs, *prepared,
                                                       f::width, f::height, 1, 0x5678)
                  .rejection(),
              "fragment-draw-executing-device-mismatch");
    auto changed = *prepared;
    ASSERT_FALSE(changed.initial_user_sgprs.empty());
    changed.initial_user_sgprs[0].second ^= 1;
    EXPECT_EQ(g::instantiate_fragment_draw_transaction(plan, draw.fragment_draw_inputs, changed,
                                                       f::width, f::height, 1,
                                                       source_device.device_identity)
                  .rejection(),
              "fragment-draw-user-prefix-owner-mismatch");
    EXPECT_EQ(g::instantiate_fragment_draw_transaction(plan, draw.fragment_draw_inputs, *prepared,
                                                       0, f::height, 1,
                                                       source_device.device_identity)
                  .rejection(),
              "fragment-draw-raster-extent-invalid");
}
TEST_F(FragmentDrawPlan, UnsupportedAttachmentControlsNameOriginalSiteWithoutInventingOutput) {
    g::DrawItem good;
    ASSERT_TRUE(f::realize(good));
    const auto prepared = f::prepare(good);
    ASSERT_TRUE(prepared && good.fragment_draw_inputs);
    const auto plan =
        g::compile_fragment_draw_program(*good.fragment_draw_inputs, *prepared, source_device, 48);
    retain_plan(plan);
    ASSERT_TRUE(plan.rejection_reason().empty()) << plan.rejection_reason();
    for (uint32_t control : {1u << 4, 1u << 12}) { // MRT1 and VM0, respectively
        auto original = f::fragment_words();
        ASSERT_EQ(original[5], 0xf800180fu);
        original[5] ^= control;
        g::DrawItem unsupported;
        ASSERT_TRUE(f::realize(unsupported, f::color_a, original));
        const auto input = f::prepare(unsupported);
        ASSERT_TRUE(input && unsupported.fragment_draw_inputs);
        ASSERT_EQ(*unsupported.fragment_draw_inputs->raw_code, original);
        const auto refused = g::compile_fragment_draw_program(*unsupported.fragment_draw_inputs,
                                                              *input, source_device, 48);
        EXPECT_EQ(refused.rejection_reason(),
                  "fragment-draw-attachment-export-recipe-unimplemented:pc=5");
        EXPECT_FALSE(refused.capacity_owner());
        EXPECT_TRUE(refused.validation_words().empty());
        EXPECT_TRUE(refused.replay_words().empty());
    }
}
TEST_F(FragmentDrawPlan, SeventeenLiveOriginalProgramsNeverRecompileAfterWarmup) {
    std::array<g::DrawItem, 17> draws;
    std::array<std::shared_ptr<const g::FragmentPacketPreparation>, 17> prepared;
    std::array<std::shared_ptr<const g::FragmentDrawProgramPlan>, 17> plans;
    const auto before = g::fragment_draw_cache_stats().program_compile_calls;
    for (uint32_t index = 0; index < draws.size(); ++index) {
        ASSERT_TRUE(f::realize(draws[index], f::color_a, f::distinct_fragment_words(index)));
        prepared[index] = f::prepare(draws[index]);
        ASSERT_TRUE(prepared[index] && draws[index].fragment_draw_inputs);
        ASSERT_EQ(*draws[index].fragment_draw_inputs->raw_code, f::distinct_fragment_words(index));
        plans[index] = g::cached_fragment_draw_program(*draws[index].fragment_draw_inputs,
                                                       *prepared[index], source_device, 49);
        ASSERT_TRUE(plans[index]);
        ASSERT_TRUE(plans[index]->rejection_reason().empty()) << plans[index]->rejection_reason();
        ASSERT_TRUE(plans[index]->capacity_owner() && plans[index]->capacity_owner()->kernel());
        ASSERT_TRUE(plans[index]->source_live());
        EXPECT_EQ(plans[index]->capacity_owner()->kernel()->program.packet.export_observation,
                  g::FragmentPacketExportObservation::Architectural);
        // Every actual emitted form is retained separately, not overwritten by the next key.
        const auto prefix = "warm17_" + std::to_string(index) + '_';
        f::retain_source(plans[index]->collect_words(), (prefix + "collector").c_str());
        f::retain_source(plans[index]->count_words(), (prefix + "count").c_str());
        f::retain_source(plans[index]->assembly_words(), (prefix + "assembly").c_str());
        f::retain_source(plans[index]->capacity_owner()->kernel()->program.packet.spirv,
                         (prefix + "original_ps").c_str());
        f::retain_source(plans[index]->validation_words(), (prefix + "validation").c_str());
        f::retain_source(plans[index]->replay_words(), (prefix + "replay").c_str());
    }
    EXPECT_EQ(g::fragment_draw_cache_stats().program_compile_calls - before, 17u);
    for (uint32_t frame = 0; frame < 4; ++frame)
        for (uint32_t index = 0; index < draws.size(); ++index)
            EXPECT_EQ(g::cached_fragment_draw_program(*draws[index].fragment_draw_inputs,
                                                      *prepared[index], source_device, 49),
                      plans[index]);
    EXPECT_EQ(g::fragment_draw_cache_stats().program_compile_calls - before, 17u)
        << "all-clear16 would recompile on every repeated frame";
}
TEST_F(FragmentDrawPlan, DeadOriginalGenerationRetiresWithoutRetainingItsOwnAnalysis) {
    std::shared_ptr<const g::FragmentDrawProgramPlan> completion_plan;
    std::shared_ptr<const g::RasterQuadInputs> completion_inputs;
    std::weak_ptr<const std::vector<uint32_t>> source;
    {
        g::DrawItem draw;
        ASSERT_TRUE(f::realize(draw, f::color_a, f::distinct_fragment_words(100)));
        const auto prepared = f::prepare(draw);
        ASSERT_TRUE(prepared && draw.fragment_draw_inputs);
        source = draw.fragment_draw_inputs->raw_code;
        completion_plan = g::cached_fragment_draw_program(*draw.fragment_draw_inputs, *prepared,
                                                          source_device, 50);
        ASSERT_TRUE(completion_plan && completion_plan->rejection_reason().empty());
        retain_plan(*completion_plan);
        completion_inputs = draw.fragment_draw_inputs;   // genuine draw/completion producer lease
        g::clear_shader_analysis_cache();
        EXPECT_FALSE(source.expired());
    }
    EXPECT_TRUE(completion_plan->source_live());
    completion_inputs.reset();
    EXPECT_TRUE(source.expired()) << "cached copied plan must not retain the retired analysis";
    EXPECT_FALSE(completion_plan->source_live());
    const auto before = g::fragment_draw_cache_stats().program_retired;
    g::DrawItem next;
    ASSERT_TRUE(f::realize(next, f::color_a, f::distinct_fragment_words(101)));
    const auto prepared = f::prepare(next);
    ASSERT_TRUE(prepared && next.fragment_draw_inputs);
    const auto next_plan =
        g::cached_fragment_draw_program(*next.fragment_draw_inputs, *prepared, source_device, 50);
    ASSERT_TRUE(next_plan && next_plan->rejection_reason().empty());
    retain_plan(*next_plan);
    EXPECT_GT(g::fragment_draw_cache_stats().program_retired, before);
    EXPECT_FALSE(completion_plan->replay_words().empty())
        << "retiring residence cannot revoke a retained completion payload";
}
TEST(FragmentDrawResidency, AliasedOwnersAndIndeterminateLifetimeNeverLaunderEviction) {
    auto owner = std::make_shared<const std::array<uint32_t, 2>>();
    g::FragmentDrawSourceGenerations exact;
    for (uint32_t index = 0; index < 100; ++index)
        exact.remember(std::shared_ptr<const void>(owner, &(*owner)[index % 2]));
    EXPECT_TRUE(exact.live());
    EXPECT_FALSE(exact.indeterminate()) << "aliases share a generation, not separate slots";
    owner.reset();
    EXPECT_FALSE(exact.live());
    g::FragmentDrawSourceGenerations absent;
    absent.remember({});
    EXPECT_TRUE(absent.live());
    EXPECT_TRUE(absent.indeterminate());
    g::FragmentDrawSourceGenerations overflow;
    std::vector<std::shared_ptr<const uint32_t>> generations;
    for (uint32_t index = 0; index < 65; ++index) {
        generations.push_back(std::make_shared<const uint32_t>(index));
        overflow.remember(generations.back());
    }
    generations.clear();
    EXPECT_TRUE(overflow.indeterminate());
    EXPECT_TRUE(overflow.live()) << "unknown ownership is not proof a generation is dead";
}
}   // namespace
