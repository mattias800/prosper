// The order in which a colour slot above 1 finds its starting colour (color_slot_clear.h): the
// retained uniform colour the caller passed for the slot, then the pass's own programmed clear,
// then opaque black. No device: the backend half is exercised in test_shadow_compare_render.cpp.
#include "fixtures/color_slot_clear.h"
#include "shared/rtt/retained_color_clear.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <gtest/gtest.h>
#include <unordered_map>
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

namespace {
constexpr std::size_t kSeedBytes = std::size_t{64} * 32 * 4;
struct RetainedSurface {
    uint32_t w = 64, h = 32, format = 7, volume_depth = 0;
    bool has_uniform_color = true;
    Colour uniform_color{1, 0.5f, 0.25f, 0};
    std::shared_ptr<const std::vector<uint8_t>> rgba;
};
const float* select_clear(const RetainedSurface* surface, Colour& storage,
                          bool load_existing = true, uint32_t volume_depth = 0) {
    return prosper::frontend::retained_color_slot_clear(surface, load_existing, 64, 32, 7u,
                                                        volume_depth, storage);
}
}   // namespace

TEST(ColorSlotClear, RetainedClearOwnsItsValuesAcrossCacheMutationAndErase) {
    std::unordered_map<uint64_t, RetainedSurface> cache{{1, RetainedSurface{}}};
    const Colour expected = cache.at(1).uniform_color;
    Colour storage{};
    const float* clear = select_clear(&cache.at(1), storage);
    ASSERT_NE(clear, nullptr);
    cache.at(1).uniform_color = {0, 1, 0, 1};
    // Mutation distinguishes a borrowed pointer without requiring a use-after-free to crash.
    EXPECT_EQ(prosper::test::color_slot_clear_value(clear, std::vector<Draw>{}, 2), expected);
    ASSERT_EQ(clear, storage.data());
    cache.clear();
    EXPECT_EQ(prosper::test::color_slot_clear_value(clear, std::vector<Draw>{}, 2), expected);
}

TEST(ColorSlotClear, RetainedClearRequiresMatchingEligibleTwoDimensionalSurface) {
    RetainedSurface surface;
    Colour storage{};
    ASSERT_NE(select_clear(&surface, storage), nullptr);
    EXPECT_EQ(select_clear(nullptr, storage), nullptr);
    EXPECT_EQ(select_clear(&surface, storage, false), nullptr);
    EXPECT_EQ(select_clear(&surface, storage, true, 2), nullptr);
    surface.volume_depth = 2;
    EXPECT_EQ(select_clear(&surface, storage), nullptr);
    surface.volume_depth = 0;
    surface.has_uniform_color = false;
    EXPECT_EQ(select_clear(&surface, storage), nullptr);
    surface.has_uniform_color = true;
    ++surface.w;
    EXPECT_EQ(select_clear(&surface, storage), nullptr);
    --surface.w;
    ++surface.h;
    EXPECT_EQ(select_clear(&surface, storage), nullptr);
    --surface.h;
    ++surface.format;
    EXPECT_EQ(select_clear(&surface, storage), nullptr);
    --surface.format;
    EXPECT_NE(select_clear(&surface, storage), nullptr);
}

TEST(ColorSlotClear, EverySlotsSelectedClearHasIndependentStorage) {
    std::array<Colour, 8> storage{}, expected{};
    std::array<const float*, 8> selected{};
    RetainedSurface surface;
    for (uint32_t slot = 0; slot < selected.size(); ++slot) {
        surface.uniform_color = {static_cast<float>(slot) / 8, 0.25f, 0.5f, 1};
        expected[slot] = surface.uniform_color;
        selected[slot] = select_clear(&surface, storage[slot]);
        ASSERT_NE(selected[slot], nullptr);
    }
    surface.uniform_color = {0, 0, 0, 0};
    for (uint32_t slot = 0; slot < selected.size(); ++slot)
        EXPECT_EQ(prosper::test::color_slot_clear_value(selected[slot], std::vector<Draw>{}, slot),
                  expected[slot]);
}

TEST(ColorSlotClear, CpuSeedRetainsTheImmutablePixelOwnerAfterCacheErasure) {
    std::unordered_map<uint64_t, RetainedSurface> cache{{1, RetainedSurface{}}};
    cache.at(1).rgba = std::make_shared<const std::vector<uint8_t>>(kSeedBytes, 0x5a);
    const std::weak_ptr<const std::vector<uint8_t>> observed = cache.at(1).rgba;
    auto seed =
        prosper::frontend::retained_color_slot_seed(&cache.at(1), true, 64, 32, 7u, 0, kSeedBytes);
    ASSERT_NE(seed, nullptr);
    cache.clear(); // mutation: borrowing the cache's pointer loses this last real owner
    ASSERT_FALSE(observed.expired());
    EXPECT_EQ(*seed, std::vector<uint8_t>(kSeedBytes, 0x5a));
    seed.reset();
    EXPECT_TRUE(observed.expired());
}

TEST(ColorSlotClear, CpuSeedRequiresTheExactCurrentTwoDimensionalSurface) {
    RetainedSurface surface;
    surface.rgba = std::make_shared<const std::vector<uint8_t>>(kSeedBytes, 0x5a);
    const auto select = [&](bool load, uint32_t volume, std::size_t bytes) {
        return prosper::frontend::retained_color_slot_seed(&surface, load, 64, 32, 7u, volume,
                                                           bytes);
    };
    ASSERT_NE(select(true, 0, kSeedBytes), nullptr);
    EXPECT_EQ(select(false, 0, kSeedBytes), nullptr);
    EXPECT_EQ(select(true, 2, kSeedBytes), nullptr);
    EXPECT_EQ(select(true, 0, kSeedBytes - 1), nullptr);
    surface.volume_depth = 2;
    EXPECT_EQ(select(true, 0, kSeedBytes), nullptr);
    surface.volume_depth = 0;
    ++surface.w;
    EXPECT_EQ(select(true, 0, kSeedBytes), nullptr);
    --surface.w;
    ++surface.h;
    EXPECT_EQ(select(true, 0, kSeedBytes), nullptr);
    --surface.h;
    ++surface.format;
    EXPECT_EQ(select(true, 0, kSeedBytes), nullptr);
    --surface.format;
    surface.rgba.reset();
    EXPECT_EQ(select(true, 0, kSeedBytes), nullptr);
}
