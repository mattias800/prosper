// test_avplayer_misc — the AvPlayer remainder: trick speed, bandwidth window, printf.
//
// These four exports were unregistered, so the dispatcher answered `0`: a trick-speed request
// nobody honoured, a bandwidth window nobody checked, log lines never emitted. Every TEST
// drives the real NIDs: handles validate, incoherent bandwidth windows fail, and the printf
// surface refuses a null format instead of faulting on it.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using namespace prosper;

static uint64_t addr(const void* p) { return (uint64_t)(uintptr_t)p; }
// Zero-extended, matching what the handlers in avplayer.cpp actually return.
static constexpr uint64_t kInvalidParams = 0x806a0001ull;

TEST(AvPlayerMisc, AllNidsBound) {
    register_builtin_hle();
    static const char* table[] = {
        "sceAvPlayerSetTrickSpeed", "sceAvPlayerSetAvailableBandwidth",
        "sceAvPlayerPrintf",        "sceAvPlayerVprintf",
    };
    static_assert(sizeof(table) / sizeof(table[0]) == 4, "the 4 remainder exports");
    for (const char* name : table) {
        EXPECT_NE(Hle::lookup(nid_hash(name)), nullptr) << name << " is not registered";
    }
}

TEST(AvPlayerMisc, NidsResolveToStubValues) {
    // All four NIDs reproduce the stub table verbatim.
    EXPECT_EQ(nid_hash("sceAvPlayerSetTrickSpeed"), "av8Z++94rs0");
    EXPECT_EQ(nid_hash("sceAvPlayerSetAvailableBandwidth"), "N6Oy-EjduiY");
    EXPECT_EQ(nid_hash("sceAvPlayerPrintf"), "agig-iDRrTE");
    EXPECT_EQ(nid_hash("sceAvPlayerVprintf"), "yN7Jhuv8g24");
    EXPECT_NE(nid_hash("sceAvPlayerSetTrickSpeed"), "AAAAAAAAAAA")
        << "positive control: the discriminator rejects a wrong NID";
}

TEST(AvPlayerMisc, TrickSpeedAndBandwidthValidateHandles) {
    register_builtin_hle();
    HleFn init = Hle::lookup(nid_hash("sceAvPlayerInit"));
    HleFn trick = Hle::lookup(nid_hash("sceAvPlayerSetTrickSpeed"));
    HleFn bw = Hle::lookup(nid_hash("sceAvPlayerSetAvailableBandwidth"));
    ASSERT_NE(init, nullptr);
    ASSERT_NE(trick, nullptr);
    ASSERT_NE(bw, nullptr);

    const uint64_t player = init(0, 0, 0, 0, 0, 0);
    EXPECT_NE(player, 0u) << "Init with no init data still mints a handle";
    EXPECT_EQ(trick(player, 1, 0, 0, 0, 0), 0u);
    EXPECT_EQ(trick(0xDEADu, 1, 0, 0, 0, 0), kInvalidParams) << "foreign handle refused";
    EXPECT_EQ(bw(player, 1000, 500, 2000, 0, 0), 0u) << "coherent window accepted";
    EXPECT_EQ(bw(player, 1000, 2000, 500, 0, 0), kInvalidParams)
        << "min > max with both nonzero is refused";
    EXPECT_EQ(bw(0xDEADu, 1000, 500, 2000, 0, 0), kInvalidParams) << "foreign handle refused";
}

TEST(AvPlayerMisc, PrintfForwardsFormatAndRefusesNull) {
    register_builtin_hle();
    HleFn printf_fn = Hle::lookup(nid_hash("sceAvPlayerPrintf"));
    HleFn vprintf_fn = Hle::lookup(nid_hash("sceAvPlayerVprintf"));
    ASSERT_NE(printf_fn, nullptr);
    ASSERT_NE(vprintf_fn, nullptr);

    static const char msg[] = "avplayer misc probe";
    EXPECT_EQ(printf_fn(addr(msg), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(vprintf_fn(addr(msg), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(printf_fn(0, 0, 0, 0, 0, 0), kInvalidParams) << "null format refused";
    EXPECT_EQ(vprintf_fn(0, 0, 0, 0, 0, 0), kInvalidParams) << "null format refused";
}
