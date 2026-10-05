// test_avplayer_misc -- sceAvPlayerSetTrickSpeed and sceAvPlayerSetAvailableBandwidth.
//
// Both were unregistered, so the dispatcher answered 0 for every input. The contracts pinned here come
// from libSceAvPlayer.native.sprx, the module PS5 titles link: trick speed needs a source first, then
// refuses 0 and the slow band -399..399 except 100, then refuses a source that is not MP4-family or
// WebM; bandwidth is accepted only before a source is added. Each arm names the mutation it catches.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstddef>
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
    // Without a source the native setter fails BEFORE it looks at the speed (0x9ac0 -> 0x9ade).
    EXPECT_EQ(trick(player, 100, 0, 0, 0, 0), kOperationFailed) << "no source yet";
    EXPECT_EQ(trick(player, 50, 0, 0, 0, 0), kOperationFailed) << "the source is checked first";
    EXPECT_EQ(trick(player, 0, 0, 0, 0, 0), kOperationFailed) << "even for speed 0";
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

// The type gate (native 0x9b22): trick play is accepted only for a source the module classifies as
// MP4-family (1) or WebM (2) -- by the AddSourceEx sourceType when given, else the URI extension.
TEST(AvPlayerMisc, TrickSpeedNeedsAnMp4OrWebmSource) {
    register_builtin_hle();
    HleFn init = Hle::lookup(nid_hash("sceAvPlayerInit"));
    HleFn trick = Hle::lookup(nid_hash("sceAvPlayerSetTrickSpeed"));
    HleFn add = Hle::lookup(nid_hash("sceAvPlayerAddSource"));
    HleFn add_ex = Hle::lookup(nid_hash("sceAvPlayerAddSourceEx"));
    ASSERT_NE(init, nullptr);
    ASSERT_NE(trick, nullptr);
    ASSERT_NE(add, nullptr);
    ASSERT_NE(add_ex, nullptr);
    struct Case { const char* uri; uint64_t want; };
    static const Case cases[] = {
        {"/app0/does-not-exist-trick-a.MP4", 0u},          // case-insensitive
        {"/app0/does-not-exist-trick-b.m4v", 0u},
        {"/app0/does-not-exist-trick-c.mov", 0u},
        {"/app0/does-not-exist-trick-d.webm", 0u},
        {"/app0/does-not-exist-trick-e.m3u8", kInvalidSpeed},  // HLS: type 8
        {"/app0/does-not-exist-trick-f.bik", kInvalidSpeed},   // extension under four chars
        {"/app0/does-not-exist-trick-g", kInvalidSpeed},       // no extension
        {"http://host/clip.mp4?x=1.m3u8", 0u},                 // query cut off a scheme URI
    };
    for (const Case& c : cases) {
        const uint64_t player = init(0, 0, 0, 0, 0, 0);
        ASSERT_NE(player, 0u);
        ASSERT_EQ(add(player, addr(c.uri), 0, 0, 0, 0), 0u) << c.uri;
        EXPECT_EQ(trick(player, 800, 0, 0, 0, 0), c.want) << c.uri;
        EXPECT_EQ(trick(player, 50, 0, 0, 0, 0), kInvalidSpeed) << c.uri << ": the band is checked first";
    }
    // AddSourceEx's explicit sourceType wins over the extension.
    struct Uri { const char* name; uint32_t length; };
    struct Details { Uri uri; uint8_t rsv1[64]; uint32_t source_type; uint8_t rsv2[44]; };
    static_assert(offsetof(Details, source_type) == 0x50, "SceAvPlayerSourceDetails.sourceType");
    static const char hls_named[] = "/app0/does-not-exist-trick-h.m3u8";
    Details details{};
    details.uri = {hls_named, (uint32_t)sizeof(hls_named) - 1};
    details.source_type = 1;
    const uint64_t player = init(0, 0, 0, 0, 0, 0);
    ASSERT_NE(player, 0u);
    ASSERT_EQ(add_ex(player, 0, addr(&details), 0, 0, 0), 0u);
    EXPECT_EQ(trick(player, 800, 0, 0, 0, 0), 0u) << "explicit MP4 type over a .m3u8 name";
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
