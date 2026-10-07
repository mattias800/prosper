// test_unchanged_publication_census -- the census that says how much of what the compute chain publishes to
// the renderer is byte-identical to the previous publication of the same target.
//
// WHAT EACH TEST KILLS:
//   HashSeesEveryByte                  a hash that ignores a region (head, a lane boundary, the 0..31 byte
//                                      tail, the very last byte), which would report a changed frame as unchanged
//   HashSeparatesLengthsAndPaddings    a zero-padded buffer colliding with its prefix
//   FirstPublicationIsNeverIdentical   counting the first sight of a target as "unchanged"
//   RepeatIsIdenticalAndChangeIsNot    the identical/changed decision itself
//   TargetsAreKeyedByAddressAndShape   one target's history leaking into another's, or a resized target at the
//                                      same address being called unchanged
//   EmptyOrNullPublishesNothing        a null or empty write skewing the counts
//   ReportNamesTheBigTargetsFirst      the report listing small targets first, or printing when nothing was counted
#include "shared/perf/unchanged_publication_census.hpp"

#include <gtest/gtest.h>

#include <vector>

using prosper::perf::UnchangedPublicationCensus;
using prosper::perf::hash_published_bytes;

namespace {
std::vector<uint8_t> pattern(size_t n, uint8_t seed = 1) {
    std::vector<uint8_t> v(n);
    uint32_t x = seed * 2654435761u + 12345u;
    for (size_t i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        v[i] = static_cast<uint8_t>(x >> 24);
    }
    return v;
}
}  // namespace

TEST(UnchangedPublicationCensus, HashSeesEveryByte) {
    // Sizes straddle the 32-byte block: below, exactly one block, and with tails of 1, 17 and 31 bytes.
    for (size_t n : {size_t{1}, size_t{31}, size_t{32}, size_t{33}, size_t{49}, size_t{63}, size_t{64}, size_t{4099}}) {
        const std::vector<uint8_t> base = pattern(n);
        const uint64_t h = hash_published_bytes(base.data(), base.size());
        for (size_t pos : {size_t{0}, n / 2, n - 1}) {
            std::vector<uint8_t> changed = base;
            changed[pos] ^= 0x01;
            EXPECT_NE(hash_published_bytes(changed.data(), changed.size()), h) << "n=" << n << " pos=" << pos;
        }
    }
    // Every byte of a lane-boundary-heavy buffer, one at a time.
    const std::vector<uint8_t> base = pattern(100);
    const uint64_t h = hash_published_bytes(base.data(), base.size());
    for (size_t pos = 0; pos < base.size(); ++pos) {
        std::vector<uint8_t> changed = base;
        changed[pos] ^= 0x80;
        EXPECT_NE(hash_published_bytes(changed.data(), changed.size()), h) << "pos=" << pos;
    }
}

TEST(UnchangedPublicationCensus, HashSeparatesLengthsAndPaddings) {
    const std::vector<uint8_t> a = pattern(40);
    std::vector<uint8_t> padded = a;
    padded.resize(64, 0);
    EXPECT_NE(hash_published_bytes(a.data(), a.size()), hash_published_bytes(padded.data(), padded.size()));
    // And the same bytes hash the same, however the buffer is held.
    const std::vector<uint8_t> copy = a;
    EXPECT_EQ(hash_published_bytes(a.data(), a.size()), hash_published_bytes(copy.data(), copy.size()));
}

TEST(UnchangedPublicationCensus, FirstPublicationIsNeverIdentical) {
    UnchangedPublicationCensus c;
    const std::vector<uint8_t> frame = pattern(4096);
    EXPECT_FALSE(c.note(0x1000, 32, 32, 0, frame.data(), frame.size()));
    EXPECT_EQ(c.totals().publications, 1u);
    EXPECT_EQ(c.totals().identical, 0u);
}

TEST(UnchangedPublicationCensus, RepeatIsIdenticalAndChangeIsNot) {
    UnchangedPublicationCensus c;
    std::vector<uint8_t> frame = pattern(4096);
    c.note(0x1000, 32, 32, 0, frame.data(), frame.size());
    EXPECT_TRUE(c.note(0x1000, 32, 32, 0, frame.data(), frame.size())) << "the same bytes again";
    EXPECT_TRUE(c.note(0x1000, 32, 32, 0, frame.data(), frame.size()));
    frame[4095] ^= 0xff;   // the very last byte
    EXPECT_FALSE(c.note(0x1000, 32, 32, 0, frame.data(), frame.size())) << "one changed byte at the end";
    EXPECT_TRUE(c.note(0x1000, 32, 32, 0, frame.data(), frame.size())) << "and the new picture repeats";
    const auto t = c.totals();
    EXPECT_EQ(t.publications, 5u);
    EXPECT_EQ(t.identical, 3u);
    EXPECT_EQ(t.bytes, 5u * 4096u);
    EXPECT_EQ(t.identical_bytes, 3u * 4096u);
}

TEST(UnchangedPublicationCensus, TargetsAreKeyedByAddressAndShape) {
    UnchangedPublicationCensus c;
    const std::vector<uint8_t> frame = pattern(4096);
    c.note(0x1000, 32, 32, 0, frame.data(), frame.size());
    // The same bytes at a DIFFERENT address are a first sight, not a repeat of the other target.
    EXPECT_FALSE(c.note(0x2000, 32, 32, 0, frame.data(), frame.size()));
    // The same address with a different extent or format is a different picture even with equal bytes.
    EXPECT_FALSE(c.note(0x1000, 64, 16, 0, frame.data(), frame.size()));
    EXPECT_FALSE(c.note(0x1000, 64, 16, 7, frame.data(), frame.size()));
    // ...and after the reshape it repeats normally.
    EXPECT_TRUE(c.note(0x1000, 64, 16, 7, frame.data(), frame.size()));
}

TEST(UnchangedPublicationCensus, EmptyOrNullPublishesNothing) {
    UnchangedPublicationCensus c;
    const std::vector<uint8_t> frame = pattern(64);
    EXPECT_FALSE(c.note(1, 1, 1, 0, nullptr, 64));
    EXPECT_FALSE(c.note(1, 1, 1, 0, frame.data(), 0));
    EXPECT_EQ(c.totals().publications, 0u);
    EXPECT_TRUE(c.format().empty()) << "nothing counted prints nothing";
}

TEST(UnchangedPublicationCensus, ReportNamesTheBigTargetsFirst) {
    UnchangedPublicationCensus c;
    const std::vector<uint8_t> small = pattern(1024, 3);
    const std::vector<uint8_t> big = pattern(1 << 20, 4);
    for (int i = 0; i < 3; ++i) c.note(0x10, 16, 16, 0, small.data(), small.size());
    for (int i = 0; i < 2; ++i) c.note(0x20, 512, 512, 1, big.data(), big.size());
    const auto rows = c.rows(8);
    ASSERT_EQ(rows.size(), 2u);
    EXPECT_EQ(rows[0].address, 0x20u) << "the target with the most bytes leads";
    EXPECT_EQ(rows[0].identical, 1u);
    EXPECT_EQ(rows[1].address, 0x10u);
    EXPECT_EQ(rows[1].identical, 2u);
    EXPECT_EQ(c.rows(1).size(), 1u) << "top-N is honoured";
    const std::string text = c.format();
    EXPECT_NE(text.find("[unchanged-census] publications=5 identical=3 (60.0%)"), std::string::npos) << text;
    EXPECT_LT(text.find("addr=0x20"), text.find("addr=0x10")) << text;
}
