// The order in which a colour slot above 1 finds its starting colour (color_slot_clear.h): the
// retained uniform colour the caller passed for the slot, then the pass's own programmed clear,
// then opaque black. No device: the backend half is exercised in test_shadow_compare_render.cpp.
#include "fixtures/color_slot_clear.h"

#include <array>
#include <gtest/gtest.h>
#include <vector>

namespace {
struct Target {
    bool has_clear = false;
    float clear[4] = {0.0f, 0.0f, 0.0f, 1.0f};
};
struct State {
    std::array<Target, 8> color_targets{};
};
struct Draw {
    const State* ps = nullptr;
};
using Colour = std::array<float, 4>;
}   // namespace

TEST(ColorSlotClear, WithNothingSaidTheSlotStartsOpaqueBlack) {
    const State plain;
    const std::vector<Draw> draws{{&plain}, {nullptr}};
    EXPECT_EQ(prosper::test::color_slot_clear_value(nullptr, draws, 2), (Colour{0, 0, 0, 1}));
    EXPECT_EQ(prosper::test::color_slot_clear_value(nullptr, std::vector<Draw>{}, 7),
              (Colour{0, 0, 0, 1}));
}

TEST(ColorSlotClear, ThePassesOwnClearIsTheFirstOneADrawProgramsForThatSlot) {
    State none, red, blue;
    red.color_targets[3].has_clear = true;
    red.color_targets[3].clear[0] = 1.0f;
    blue.color_targets[3].has_clear = true;
    blue.color_targets[3].clear[2] = 1.0f;
    const std::vector<Draw> draws{{nullptr}, {&none}, {&red}, {&blue}};
    EXPECT_EQ(prosper::test::color_slot_clear_value(nullptr, draws, 3), (Colour{1, 0, 0, 1}));
    // Another slot of the same draws is not touched by slot 3's clear.
    EXPECT_EQ(prosper::test::color_slot_clear_value(nullptr, draws, 4), (Colour{0, 0, 0, 1}));
}

TEST(ColorSlotClear, ARetainedUniformColourComesBeforeThePassesOwnClear) {
    State red;
    red.color_targets[2].has_clear = true;
    red.color_targets[2].clear[0] = 1.0f;
    const std::vector<Draw> draws{{&red}};
    const float uniform[4] = {1.0f, 1.0f, 1.0f, 0.0f};
    EXPECT_EQ(prosper::test::color_slot_clear_value(uniform, draws, 2), (Colour{1, 1, 1, 0}));
    EXPECT_EQ(prosper::test::color_slot_clear_value(nullptr, draws, 2), (Colour{1, 0, 0, 1}));
}
