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
//   NewWindowForgetsOldHistory         a repeat being claimed across two F8 windows
//   ConcurrentNotesAreCounted          the lock in note() (lost updates under four threads)
//   UnchangedPublicationHook.*         the F8-window gate (hashing outside a window), `inner` skipped or
//                                      run conditionally, and CPU-pixel-less writes being counted
//   ReportNamesTheBigTargetsFirst      the report listing small targets first, or printing when nothing was counted
#include "shared/perf/unchanged_publication_census.hpp"
#include "shared/perf/unchanged_publication_hook.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <thread>
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

TEST(UnchangedPublicationCensus, NewWindowForgetsOldHistory) {
    UnchangedPublicationCensus c;
    const std::vector<uint8_t> frame = pattern(4096);
    c.note(0x1000, 32, 32, 0, frame.data(), frame.size(), 1);
    EXPECT_TRUE(c.note(0x1000, 32, 32, 0, frame.data(), frame.size(), 1));
    EXPECT_FALSE(c.note(0x1000, 32, 32, 0, frame.data(), frame.size(), 2)) << "first sight in a new window";
    EXPECT_TRUE(c.note(0x1000, 32, 32, 0, frame.data(), frame.size(), 2));
}

TEST(UnchangedPublicationCensus, ReportsHashTime) {
    UnchangedPublicationCensus c;
    const std::vector<uint8_t> frame = pattern(1 << 20);
    c.note(0x1, 512, 512, 0, frame.data(), frame.size());
    EXPECT_GT(c.totals().hash_ns, 0u);
    EXPECT_NE(c.format().find("hash_ms="), std::string::npos);
}

TEST(UnchangedPublicationCensus, ConcurrentNotesAreCounted) {
    UnchangedPublicationCensus c;
    const std::vector<uint8_t> frame = pattern(2048);
    constexpr int kThreads = 4, kEach = 500;
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t)
        threads.emplace_back([&, t] {
            for (int i = 0; i < kEach; ++i)
                c.note(0x100 + static_cast<uint64_t>(t % 2), 32, 16, 0, frame.data(), frame.size());
        });
    for (auto& th : threads) th.join();
    const auto totals = c.totals();
    EXPECT_EQ(totals.publications, static_cast<uint64_t>(kThreads * kEach));
    EXPECT_EQ(totals.bytes, static_cast<uint64_t>(kThreads * kEach) * frame.size());
    EXPECT_EQ(totals.identical, totals.publications - 2) << "two distinct targets, one first sight each";
}

namespace {
prosper::gpu::LiveTargetImageWrite make_write(uint64_t addr, const std::vector<uint8_t>& bytes) {
    prosper::gpu::LiveTargetImageWrite w;
    w.gpu_addr = addr;
    w.width = 32;
    w.height = 32;
    w.linear_pixels = std::make_shared<const std::vector<uint8_t>>(bytes);
    return w;
}

struct HookFixture {
    prosper::perf::InteractivePerformanceCapture capture{prosper::perf::CaptureConfig{}};
    UnchangedPublicationCensus census;
    int inner_calls = 0;
    const prosper::gpu::LiveTargetImageWrite* last = nullptr;
    std::filesystem::path dir = std::filesystem::temp_directory_path() / "prosper_unchanged_hook_test";
    HookFixture() { std::filesystem::create_directories(dir); }
    ~HookFixture() {
        capture.cancel();
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
    std::function<void(const prosper::gpu::LiveTargetImageWrite&)> hook() {
        return prosper::perf::with_unchanged_publication_census(
            [this](const prosper::gpu::LiveTargetImageWrite& w) {
                ++inner_calls;
                last = &w;
            },
            capture, census);
    }
    bool arm() {
        return capture
            .arm(dir.string(), "hook-test-id", "hook-test", "test-revision", 0,
                 std::chrono::system_clock::now())
            .ok;
    }
};
}  // namespace

TEST(UnchangedPublicationHook, ClosedWindowCountsNothingAndStillCallsInner) {
    HookFixture f;
    const auto write = make_write(0x40, pattern(4096));
    auto hook = f.hook();
    hook(write);
    hook(write);
    EXPECT_EQ(f.inner_calls, 2) << "the renderer must see every publication";
    EXPECT_EQ(f.last, &write) << "and the very same write";
    EXPECT_EQ(f.census.totals().publications, 0u) << "no hashing outside an F8 window";
}

TEST(UnchangedPublicationHook, OpenWindowCountsAndCancelStops) {
    HookFixture f;
    ASSERT_TRUE(f.arm());
    const auto write = make_write(0x40, pattern(4096));
    auto hook = f.hook();
    hook(write);
    hook(write);
    EXPECT_EQ(f.inner_calls, 2);
    EXPECT_EQ(f.census.totals().publications, 2u);
    EXPECT_EQ(f.census.totals().identical, 1u);
    f.capture.cancel();
    hook(write);
    EXPECT_EQ(f.inner_calls, 3) << "inner is called whatever the window state";
    EXPECT_EQ(f.census.totals().publications, 2u) << "a closed window counts nothing more";
}

TEST(UnchangedPublicationHook, WritesWithoutCpuPixelsAreNotCounted) {
    HookFixture f;
    ASSERT_TRUE(f.arm());
    prosper::gpu::LiveTargetImageWrite mirrored;
    mirrored.gpu_addr = 0x80;
    mirrored.width = mirrored.height = 16;
    auto hook = f.hook();
    hook(mirrored);
    EXPECT_EQ(f.inner_calls, 1);
    EXPECT_EQ(f.census.totals().publications, 0u);
}

TEST(UnchangedPublicationHook, EachWindowStartsFresh) {
    HookFixture f;
    const auto write = make_write(0x40, pattern(4096));
    auto hook = f.hook();
    ASSERT_TRUE(f.arm());
    hook(write);
    f.capture.cancel();
    ASSERT_TRUE(f.arm());
    hook(write);
    EXPECT_EQ(f.census.totals().publications, 2u);
    EXPECT_EQ(f.census.totals().identical, 0u) << "the same bytes across two windows are not a repeat";
}
