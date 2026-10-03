// test_win_data_watch -- the Windows data write-watchpoint (PROSPER_HWWATCH_ABS).
//
// The instrument is only worth trusting if its zero means something, so each property is pinned
// from both sides: an armed thread's write is caught (positive control), an UNARMED thread's write
// to the same word is not (debug registers are per-thread, which is why the guest-thread-entry hook
// exists), a read and a write to a neighbouring word are not, and a malformed spec arms nothing.
#include "host/image/win_data_watch.hpp"
#include <gtest/gtest.h>
#include <windows.h>
#include <atomic>
#include <cstdint>
#include <thread>

using namespace prosper;

namespace {

alignas(64) volatile uint64_t g_watched[8];   // [0] is watched; the rest are neighbours

struct WatchFixture : ::testing::Test {
    void SetUp() override {
        win_data_watch_reset_for_test();
        g_watched[0] = 0;
        g_watched[1] = 0;
    }
    void TearDown() override { win_data_watch_reset_for_test(); }
};

uint64_t watched_addr() {
    return reinterpret_cast<uint64_t>(&g_watched[0]);
}

}   // namespace

TEST(WinDataWatchSpec, Dr7EncodesWriteWatchPerLength) {
    // L0 | RW0=01 (write) | LEN0 in bits 18-19: 1->0, 2->1, 4->3, 8->2.
    EXPECT_EQ(win_data_watch_dr7(1), 1ull | (1ull << 16) | (0ull << 18));
    EXPECT_EQ(win_data_watch_dr7(2), 1ull | (1ull << 16) | (1ull << 18));
    EXPECT_EQ(win_data_watch_dr7(4), 1ull | (1ull << 16) | (3ull << 18));
    EXPECT_EQ(win_data_watch_dr7(8), 1ull | (1ull << 16) | (2ull << 18));
    EXPECT_EQ(win_data_watch_dr7(3), 0u);
    EXPECT_EQ(win_data_watch_dr7(16), 0u);
}

TEST(WinDataWatchSpec, ParseAcceptsOnlyWatchableSpecs) {
    uint64_t a = 0;
    unsigned l = 0;
    EXPECT_TRUE(win_data_watch_parse("0x4010428098", &a, &l));
    EXPECT_EQ(a, 0x4010428098ull);
    EXPECT_EQ(l, 8u);
    EXPECT_TRUE(win_data_watch_parse("0x1000a0:4", &a, &l));
    EXPECT_EQ(l, 4u);
    // A typo must disable the watch, never arm a different one.
    EXPECT_FALSE(win_data_watch_parse("", &a, &l));
    EXPECT_FALSE(win_data_watch_parse(nullptr, &a, &l));
    EXPECT_FALSE(win_data_watch_parse("zzz", &a, &l));
    EXPECT_FALSE(win_data_watch_parse("0x4010428098:", &a, &l));
    EXPECT_FALSE(win_data_watch_parse("0x4010428098:3", &a, &l));   // unsupported length
    EXPECT_FALSE(win_data_watch_parse("0x4010428099", &a, &l));   // misaligned for 8
    EXPECT_FALSE(win_data_watch_parse("0x4010428098:8x", &a, &l));   // trailing junk
    EXPECT_FALSE(win_data_watch_parse("0x10", &a, &l));   // not a guest address
}

TEST_F(WatchFixture, ArmedThreadWriteIsCaughtWithValueAndThread) {
    ASSERT_TRUE(win_data_watch_configure(watched_addr(), 8));
    win_data_watch_arm_current_thread();
    g_watched[0] = 0x1122334455667788ull;
    ASSERT_EQ(win_data_watch_hit_count(), 1u) << "an armed thread's write must be caught";
    WinDataWatchHit hit{};
    ASSERT_TRUE(win_data_watch_last_hit(&hit));
    EXPECT_EQ(hit.addr, watched_addr());
    EXPECT_EQ(hit.value, 0x1122334455667788ull);
    EXPECT_EQ(hit.tid, GetCurrentThreadId());
    EXPECT_NE(hit.rip, 0u);
}

TEST_F(WatchFixture, ReadsAndNeighbouringWritesAreNotCaught) {
    ASSERT_TRUE(win_data_watch_configure(watched_addr(), 8));
    win_data_watch_arm_current_thread();
    g_watched[1] = 0xdead;   // adjacent word
    const uint64_t r = g_watched[0];   // a read of the watched word
    (void)r;
    EXPECT_EQ(win_data_watch_hit_count(), 0u);
    g_watched[0] = 1;   // control: the write that must fire
    EXPECT_EQ(win_data_watch_hit_count(), 1u);
}

TEST_F(WatchFixture, UnarmedThreadIsBlindAndArmedWorkerIsSeen) {
    ASSERT_TRUE(win_data_watch_configure(watched_addr(), 8));
    // Debug registers are per thread: a thread that never passed the arming boundary writes
    // invisibly. This is the zero that is NOT evidence, pinned so nobody reads one as such.
    std::thread blind([] { g_watched[0] = 0xb11d; });
    blind.join();
    EXPECT_EQ(win_data_watch_hit_count(), 0u);

    unsigned long armed_tid = 0;
    std::thread armed([&] {
        win_data_watch_arm_current_thread();
        armed_tid = GetCurrentThreadId();
        g_watched[0] = 0xa4ed;
    });
    armed.join();
    ASSERT_EQ(win_data_watch_hit_count(), 1u);
    WinDataWatchHit hit{};
    ASSERT_TRUE(win_data_watch_last_hit(&hit));
    EXPECT_EQ(hit.tid, armed_tid);
    EXPECT_EQ(hit.value, 0xa4edull);
}

TEST_F(WatchFixture, UnconfiguredArmIsANoOp) {
    win_data_watch_arm_current_thread();   // nothing configured: must neither crash nor arm
    g_watched[0] = 5;
    EXPECT_EQ(win_data_watch_hit_count(), 0u);
}

TEST(WinDataWatchSpec, Dr7EncodesEachSlot) {
    // Slot n: Ln = bit 2n, RWn = 01 at bit 16+4n, LENn at bit 18+4n.
    EXPECT_EQ(win_data_watch_dr7_slot(1, 4), (1ull << 2) | (1ull << 20) | (3ull << 22));
    EXPECT_EQ(win_data_watch_dr7_slot(2, 8), (1ull << 4) | (1ull << 24) | (2ull << 26));
    EXPECT_EQ(win_data_watch_dr7_slot(3, 1), (1ull << 6) | (1ull << 28) | (0ull << 30));
    EXPECT_EQ(win_data_watch_dr7_slot(4, 8), 0u);   // only DR0-DR3 exist
    EXPECT_EQ(win_data_watch_dr7_slot(0, 3), 0u);
}

TEST(WinDataWatchSpec, ParseListAcceptsUpToFourAndRefusesAnyBadEntry) {
    WinDataWatchSpec s[kWinDataWatchSlots];
    ASSERT_EQ(win_data_watch_parse_list("0x1000a0,0x1000b0:4", s), 2u);
    EXPECT_EQ(s[0].addr, 0x1000a0ull);
    EXPECT_EQ(s[0].len, 8u);
    EXPECT_EQ(s[1].addr, 0x1000b0ull);
    EXPECT_EQ(s[1].len, 4u);
    EXPECT_EQ(win_data_watch_parse_list("0x10000,0x10008,0x10010,0x10018", s), 4u);
    // Five entries, one bad entry, an empty entry and a trailing comma all disable the whole list:
    // dropping or reinterpreting some of them would watch something the caller did not ask for.
    EXPECT_EQ(win_data_watch_parse_list("0x10000,0x10008,0x10010,0x10018,0x10020", s), 0u);
    EXPECT_EQ(win_data_watch_parse_list("0x10000,0x10009", s), 0u);
    EXPECT_EQ(win_data_watch_parse_list("0x10000,,0x10008", s), 0u);
    EXPECT_EQ(win_data_watch_parse_list("0x10000,", s), 0u);
    EXPECT_EQ(win_data_watch_parse_list("", s), 0u);
    EXPECT_EQ(win_data_watch_parse_list(nullptr, s), 0u);
}

TEST_F(WatchFixture, EverySlotCatchesItsOwnWordAndReportsItsIndex) {
    const WinDataWatchSpec specs[3] = {{reinterpret_cast<uint64_t>(&g_watched[0]), 8},
                                       {reinterpret_cast<uint64_t>(&g_watched[2]), 8},
                                       {reinterpret_cast<uint64_t>(&g_watched[4]), 8}};
    ASSERT_TRUE(win_data_watch_configure_list(specs, 3));
    win_data_watch_arm_current_thread();
    g_watched[2] = 0x22;
    WinDataWatchHit hit{};
    ASSERT_TRUE(win_data_watch_last_hit(&hit));
    EXPECT_EQ(hit.slot, 1u);
    EXPECT_EQ(hit.value, 0x22ull);
    g_watched[4] = 0x44;
    ASSERT_TRUE(win_data_watch_last_hit(&hit));
    EXPECT_EQ(hit.slot, 2u);
    g_watched[0] = 0x11;
    ASSERT_TRUE(win_data_watch_last_hit(&hit));
    EXPECT_EQ(hit.slot, 0u);
    EXPECT_EQ(win_data_watch_hit_count(), 3u);
    g_watched[1] = 1;   // between watched words: must not fire
    g_watched[3] = 1;
    EXPECT_EQ(win_data_watch_hit_count(), 3u);
}
