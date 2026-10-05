// test_avplayer_misc -- sceAvPlayerSetTrickSpeed and sceAvPlayerSetAvailableBandwidth.
//
// Both were unregistered, so the dispatcher answered 0 for every input. The contracts pinned here come
// from the shipped libSceAvPlayer sprx: trick speed refuses 0 and the slow band -399..399 except 100;
// bandwidth is accepted only before a source is added. Each arm names the mutation it catches.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using namespace prosper;

static uint64_t addr(const void* p) { return (uint64_t)(uintptr_t)p; }
// Zero-extended, matching what the handlers in avplayer.cpp return.
static constexpr uint64_t kInvalidParams = 0x806a0001ull;
static constexpr uint64_t kOperationFailed = 0x806a0002ull;
static constexpr uint64_t kInvalidSpeed = 0x806a0004ull;

TEST(AvPlayerMisc, NidsResolveToFirmwareValues) {
    register_builtin_hle();
    EXPECT_EQ(nid_hash("sceAvPlayerSetTrickSpeed"), "av8Z++94rs0");
    EXPECT_EQ(nid_hash("sceAvPlayerSetAvailableBandwidth"), "N6Oy-EjduiY");
    EXPECT_NE(Hle::lookup("av8Z++94rs0"), nullptr);
    EXPECT_NE(Hle::lookup("N6Oy-EjduiY"), nullptr);
}

TEST(AvPlayerMisc, TrickSpeedRefusesTheSlowBand) {
    register_builtin_hle();
    HleFn init = Hle::lookup(nid_hash("sceAvPlayerInit"));
    HleFn trick = Hle::lookup(nid_hash("sceAvPlayerSetTrickSpeed"));
    HleFn add = Hle::lookup(nid_hash("sceAvPlayerAddSource"));
    ASSERT_NE(init, nullptr);
    ASSERT_NE(trick, nullptr);
    ASSERT_NE(add, nullptr);
    const uint64_t player = init(0, 0, 0, 0, 0, 0);
    ASSERT_NE(player, 0u);

    EXPECT_EQ(trick(0, 100, 0, 0, 0, 0), kInvalidParams) << "NULL handle";
    EXPECT_EQ(trick(0xDEADu, 100, 0, 0, 0, 0), kInvalidParams) << "foreign handle";
    // Without a source the player's own setter fails, after the speed checks.
    EXPECT_EQ(trick(player, 100, 0, 0, 0, 0), kOperationFailed) << "no source yet";
    EXPECT_EQ(trick(player, 50, 0, 0, 0, 0), kInvalidSpeed) << "the speed is checked first";
    static const char source[] = "/app0/does-not-exist-avplayer-trick.mp4";
    ASSERT_EQ(add(player, addr(source), 0, 0, 0, 0), 0u);
    EXPECT_EQ(trick(player, 100, 0, 0, 0, 0), 0u) << "normal speed";
    EXPECT_EQ(trick(player, 0, 0, 0, 0, 0), kInvalidSpeed) << "speed 0";
    // Unity passes playbackSpeed * 100, so 0.5x and 2x arrive as 50 and 200: both refused.
    for (int32_t speed : {1, 50, 99, 101, 200, 399, -1, -100, -399})
        EXPECT_EQ(trick(player, (uint64_t)(uint32_t)speed, 0, 0, 0, 0), kInvalidSpeed) << speed;
    for (int32_t speed : {400, 800, 1600, -400, -800})
        EXPECT_EQ(trick(player, (uint64_t)(uint32_t)speed, 0, 0, 0, 0), 0u) << speed;
}

TEST(AvPlayerMisc, BandwidthOnlyBeforeASource) {
    register_builtin_hle();
    HleFn init = Hle::lookup(nid_hash("sceAvPlayerInit"));
    HleFn add = Hle::lookup(nid_hash("sceAvPlayerAddSource"));
    HleFn bw = Hle::lookup(nid_hash("sceAvPlayerSetAvailableBandwidth"));
    ASSERT_NE(init, nullptr);
    ASSERT_NE(add, nullptr);
    ASSERT_NE(bw, nullptr);
    const uint64_t player = init(0, 0, 0, 0, 0, 0);
    ASSERT_NE(player, 0u);

    EXPECT_EQ(bw(0, 1000, 500, 2000, 0, 0), kInvalidParams) << "NULL handle";
    EXPECT_EQ(bw(0xDEADu, 1000, 500, 2000, 0, 0), kInvalidParams) << "foreign handle";
    EXPECT_EQ(bw(player, 1000, 500, 2000, 0, 0), 0u) << "before a source";
    EXPECT_EQ(bw(player, 1000, 2000, 500, 0, 0), 0u) << "the firmware has no min <= max check";

    // An unopenable source is skipped gracefully but still counts as added (#1105).
    static const char source[] = "/app0/does-not-exist-avplayer-misc.mp4";
    ASSERT_EQ(add(player, addr(source), 0, 0, 0, 0), 0u);
    EXPECT_EQ(bw(player, 1000, 500, 2000, 0, 0), kOperationFailed) << "after a source is added";
}
