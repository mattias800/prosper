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
    EXPECT_EQ(code_a->collector_shape().fields.size(), 0u);
    EXPECT_FALSE(code_a->replay_words().empty());
    EXPECT_EQ(ta.producing_inputs(), a.fragment_draw_inputs);
    EXPECT_EQ(tb.producing_inputs(), b.fragment_draw_inputs);
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
} // namespace
