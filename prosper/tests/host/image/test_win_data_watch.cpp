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
