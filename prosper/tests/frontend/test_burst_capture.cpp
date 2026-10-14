// Consecutive-frame capture (#5152): the PROSPER_SNAP_AT_FLIPS range syntax and the Shift+F9 /
// flicker-alarm burst, both pure halves of prosper-app (no SDL, Vulkan or filesystem).
#include "burst_capture.hpp"
#include "snap_author.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using prosper::frontend::BurstCapture;
using prosper::frontend::burst_continuity;
using prosper::frontend::burst_file_name;
using prosper::frontend::kBurstFramesDefault;
using prosper::frontend::kBurstFramesMax;
using prosper::frontend::kSnapFlipListMaxEntries;
using prosper::frontend::parse_burst_frames;
using prosper::frontend::parse_snap_flip_list;

namespace {
std::vector<int64_t> seq(int64_t first, int64_t last) {
    std::vector<int64_t> v;
    for (int64_t f = first; f <= last; ++f) v.push_back(f);
    return v;
}
}  // namespace

// ---- PROSPER_SNAP_AT_FLIPS ranges -------------------------------------------------------------

TEST(SnapFlipRanges, DashRangeIsInclusive) {
    EXPECT_EQ(parse_snap_flip_list("1200-1207"), seq(1200, 1207));
    EXPECT_EQ(parse_snap_flip_list("5-5"), seq(5, 5));
}

TEST(SnapFlipRanges, PlusFormCountsFrames) {
    EXPECT_EQ(parse_snap_flip_list("1200+8"), seq(1200, 1207));
    EXPECT_EQ(parse_snap_flip_list("0+1"), seq(0, 0));
}

TEST(SnapFlipRanges, RangesMixWithSinglesSortedAndDeduplicated) {
    std::vector<int64_t> want = seq(400, 407);
    want.push_back(1000);
    for (int64_t f = 2000; f < 2004; ++f) want.push_back(f);
    EXPECT_EQ(parse_snap_flip_list("2000+4,400-407,1000"), want);
    // Overlap collapses; a single inside a range adds nothing.
    EXPECT_EQ(parse_snap_flip_list("10-14,12-16,13"), seq(10, 16));
    // Single entries behave exactly as before.
    EXPECT_EQ(parse_snap_flip_list("1200,300,4000"), (std::vector<int64_t>{300, 1200, 4000}));
}

TEST(SnapFlipRanges, MalformedRangeVoidsTheWholeList) {
    // Reversed, empty sides, zero count, both operators, spaces inside a range, a sign.
    for (const char* bad : {"1207-1200", "1200-", "-1200", "1200+", "+8", "1200+0", "1200-1203+2",
                            "1200 - 1207", "1200--1207", "12a0-1207", "1200-12x7", "1,2,3-1"}) {
        EXPECT_TRUE(parse_snap_flip_list(bad).empty()) << bad;
    }
    // One bad range voids the good entries beside it, as one bad single always did.
    EXPECT_TRUE(parse_snap_flip_list("100,200-210,300-290").empty());
}

TEST(SnapFlipRanges, OverflowVoidsInsteadOfWrapping) {
    EXPECT_TRUE(parse_snap_flip_list("9223372036854775807+2").empty());
    EXPECT_TRUE(parse_snap_flip_list("1-99999999999999999999").empty());
    // The largest representable single-frame range is still legal and does not loop forever.
    EXPECT_EQ(parse_snap_flip_list("9223372036854775807+1").size(), 1u);
    EXPECT_EQ(parse_snap_flip_list("9223372036854775806-9223372036854775807").size(), 2u);
}

TEST(SnapFlipRanges, ExpansionIsCapped) {
    const std::string at_cap = "0-" + std::to_string(kSnapFlipListMaxEntries - 1);
    EXPECT_EQ(parse_snap_flip_list(at_cap.c_str()).size(), kSnapFlipListMaxEntries);
    const std::string over = "0-" + std::to_string(kSnapFlipListMaxEntries);
    EXPECT_TRUE(parse_snap_flip_list(over.c_str()).empty());
    // A huge range is refused before it is expanded (it would otherwise allocate gigabytes).
    EXPECT_TRUE(parse_snap_flip_list("0-4000000000").empty());
    // The cap is on the TOTAL, across entries.
    const std::string total = "0+" + std::to_string(kSnapFlipListMaxEntries) + ",100000";
    EXPECT_TRUE(parse_snap_flip_list(total.c_str()).empty());
}

// ---- burst capture ----------------------------------------------------------------------------

TEST(BurstFrames, DefaultValidAndMalformed) {
    EXPECT_EQ(parse_burst_frames(nullptr), kBurstFramesDefault);
    EXPECT_EQ(parse_burst_frames(""), kBurstFramesDefault);
    EXPECT_EQ(parse_burst_frames("16"), 16u);
    EXPECT_EQ(parse_burst_frames(std::to_string(kBurstFramesMax).c_str()), kBurstFramesMax);
    for (const char* bad : {"0", "1", "-4", "8x", " 8", "99999999999999999999"})
        EXPECT_FALSE(parse_burst_frames(bad).has_value()) << bad;
    EXPECT_FALSE(parse_burst_frames(std::to_string(kBurstFramesMax + 1).c_str()).has_value());
}

TEST(BurstCaptureState, CollectsExactlyTheArmedCountOfNewFrames) {
    BurstCapture b;
    EXPECT_FALSE(b.active());
    ASSERT_TRUE(b.arm("out", "burst_X", 3, "Shift+F9"));
    EXPECT_FALSE(b.arm("out", "burst_Y", 3, "again")) << "one burst at a time";
    const uint8_t px[2 * 2 * 4] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    EXPECT_TRUE(b.add(px, 2, 2, 100));
    EXPECT_FALSE(b.add(px, 2, 2, 100)) << "a re-presented flip is not a new frame";
    EXPECT_FALSE(b.add(px, 2, 2, 99));
    EXPECT_TRUE(b.add(px, 2, 2, 101));
    EXPECT_TRUE(b.active());
    EXPECT_TRUE(b.add(px, 2, 2, 103));
    EXPECT_TRUE(b.complete());
    EXPECT_FALSE(b.active());
    EXPECT_FALSE(b.add(px, 2, 2, 104)) << "a complete burst takes nothing more";
    const auto r = b.take();
    ASSERT_EQ(r.frames.size(), 3u);
    EXPECT_EQ(r.frames[2].source_flip, 103u);
    EXPECT_EQ(r.frames[0].rgba.size(), sizeof px);
    EXPECT_EQ(r.frames[0].rgba[15], 16);
    EXPECT_EQ(r.stem, "burst_X");
    EXPECT_FALSE(b.complete());
    EXPECT_TRUE(b.arm("out", "burst_Z", 2, "next")) << "idle again after take()";
}

TEST(BurstCaptureNames, SortInCaptureOrderAndCarryTheFlip) {
    EXPECT_EQ(burst_file_name("burst_PPSA01802_x", 3, 1203), "burst_PPSA01802_x_03_flip1203.bmp");
    EXPECT_LT(burst_file_name("s", 2, 900), burst_file_name("s", 10, 100));
}

TEST(BurstCaptureNames, ContinuityNamesGapsBecauseTheyHideAlternation) {
    EXPECT_EQ(burst_continuity({1200, 1201, 1202, 1203}), "consecutive guest flips 1200-1203");
    const std::string skipped = burst_continuity({1200, 1202, 1204, 1206});
    EXPECT_EQ(skipped.rfind("NOT consecutive: 3 gap(s) 1200->1202,1202->1204,1204->1206", 0), 0u)
        << skipped;
    EXPECT_NE(skipped.find("period-2"), std::string::npos);
    EXPECT_EQ(burst_continuity({7}), "");
}
