// test_log_ring — the Game Log panel's bounded line buffer. No SDL, no ImGui, no window:
// the capacity and truncation rules are what can silently go wrong (an unbounded deque behind a
// per-frame snapshot), so they are pinned here rather than eyeballed in the app.
#include "log_ring.hpp"

#include <gtest/gtest.h>

using prosper::frontend::LogRing;

TEST(LogRing, StartsEmpty) {
    LogRing ring;
    EXPECT_TRUE(ring.empty());
    EXPECT_EQ(ring.size(), 0u);
    EXPECT_TRUE(ring.snapshot().empty());
}

TEST(LogRing, KeepsInsertionOrder) {
    LogRing ring;
    ring.push("first");
    ring.push("second");
    const auto lines = ring.snapshot();
    ASSERT_EQ(lines.size(), 2u);
    EXPECT_EQ(lines[0], "first");
    EXPECT_EQ(lines[1], "second");
}

TEST(LogRing, EvictsOldestPastCapacity) {
    LogRing ring;
    for (size_t i = 0; i < LogRing::kMaxLines + 10; i++) ring.push("line" + std::to_string(i));
    EXPECT_EQ(ring.size(), LogRing::kMaxLines);
    const auto lines = ring.snapshot();
    EXPECT_EQ(lines.front(), "line10");
    EXPECT_EQ(lines.back(), "line" + std::to_string(LogRing::kMaxLines + 9));
}

TEST(LogRing, TruncatesOnePathologicalLine) {
    LogRing ring;
    ring.push(std::string(LogRing::kMaxLine + 100, 'x'));
    const auto lines = ring.snapshot();
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_EQ(lines[0].size(), LogRing::kMaxLine);
}

TEST(LogRing, ClearEmpties) {
    LogRing ring;
    ring.push("a");
    ring.clear();
    EXPECT_TRUE(ring.empty());
}
