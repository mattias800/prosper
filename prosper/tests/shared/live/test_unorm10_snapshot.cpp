// test_unorm10_snapshot -- a compute-written R10G10B10A2 display buffer is narrowed to the RGBA8
// bytes the renderer keeps for that target.
//
// WHAT EACH TEST KILLS:
//   FieldsLandInTheirChannels   R/G/B/A swapped, or read from the wrong bit field
//   ExtremesMapToFullRange      rounding or scale error (1023 -> 255, 3 -> 255, 0 -> 0)
//   WrongSizeIsRefused          a truncated or oversized buffer is published as a picture
//   TablesMatchTheOriginalFormulaForEveryValue   a table entry that differs from the rounding the
//                               per-channel expression always produced (all 1024 + 4 values)
//   FastPathMatchesTheReferenceImageForImage     a layout or lookup slip that only shows on real
//                               pixels, against a literal copy of the previous implementation
//   OutputDoesNotDependOnABufferBeingZeroed      a byte the loop forgot to write (the pooled buffer
//                               is recycled, not zero-filled)
//   MeasuredAgainstThePreviousLoop               prints both timings for a 1920x1080 target
#include "shared/live/unorm10_snapshot.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdio>

using prosper::frontend::unpack_unorm10_to_rgba8;

namespace {
void put(std::vector<uint8_t>& v, uint32_t word) {
    for (int i = 0; i < 4; ++i) v.push_back(static_cast<uint8_t>(word >> (8 * i)));
}
}  // namespace

TEST(Unorm10Snapshot, FieldsLandInTheirChannels) {
    std::vector<uint8_t> in;
    put(in, 1023u);                 // R only
    put(in, 1023u << 10);           // G only
    put(in, 1023u << 20);           // B only
    put(in, 3u << 30);              // A only
    std::vector<uint8_t> out;
    ASSERT_TRUE(unpack_unorm10_to_rgba8(in.data(), in.size(), 4, 1, out));
    const uint8_t expect[16] = {255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255};
    ASSERT_EQ(out.size(), 16u);
    for (int i = 0; i < 16; ++i) EXPECT_EQ(out[i], expect[i]) << "byte " << i;
}

TEST(Unorm10Snapshot, ExtremesMapToFullRange) {
    std::vector<uint8_t> in, out;
    put(in, 0u);
    put(in, 0xffffffffu);
    put(in, (512u) | (1u << 30));   // mid R, alpha 1/3
    ASSERT_TRUE(unpack_unorm10_to_rgba8(in.data(), in.size(), 3, 1, out));
    EXPECT_EQ(out[0], 0);   EXPECT_EQ(out[3], 0);
    EXPECT_EQ(out[4], 255); EXPECT_EQ(out[7], 255);
    EXPECT_EQ(out[8], 128); EXPECT_EQ(out[11], 85);
}

TEST(Unorm10Snapshot, WrongSizeIsRefused) {
    std::vector<uint8_t> in(8, 0), out;
    EXPECT_FALSE(unpack_unorm10_to_rgba8(in.data(), in.size(), 3, 1, out));
    EXPECT_FALSE(unpack_unorm10_to_rgba8(in.data(), in.size(), 0, 1, out));
    EXPECT_FALSE(unpack_unorm10_to_rgba8(nullptr, 8, 2, 1, out));
}

namespace {
// The previous implementation, kept verbatim as the oracle: per-channel division, one texel at a time.
std::vector<uint8_t> reference_unpack(const std::vector<uint8_t>& packed) {
    std::vector<uint8_t> rgba8(packed.size());
    for (size_t t = 0; t < packed.size() / 4; ++t) {
        uint32_t word;
        std::memcpy(&word, packed.data() + t * 4, sizeof(word));
        const uint32_t r = word & 0x3ffu, g = (word >> 10) & 0x3ffu, b = (word >> 20) & 0x3ffu;
        const uint32_t a = word >> 30;
        rgba8[t * 4 + 0] = static_cast<uint8_t>((r * 255u + 511u) / 1023u);
        rgba8[t * 4 + 1] = static_cast<uint8_t>((g * 255u + 511u) / 1023u);
        rgba8[t * 4 + 2] = static_cast<uint8_t>((b * 255u + 511u) / 1023u);
        rgba8[t * 4 + 3] = static_cast<uint8_t>((a * 255u + 1u) / 3u);
    }
    return rgba8;
}

// Deterministic words (splitmix64), so a failure reproduces.
std::vector<uint8_t> random_words(size_t texels, uint64_t seed) {
    std::vector<uint8_t> out(texels * 4);
    for (size_t t = 0; t < texels; ++t) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z ^= z >> 31;
        const uint32_t word = static_cast<uint32_t>(z);
        std::memcpy(out.data() + t * 4, &word, sizeof(word));
    }
    return out;
}
}  // namespace

TEST(Unorm10Snapshot, TablesMatchTheOriginalFormulaForEveryValue) {
    for (uint32_t v = 0; v < 1024; ++v)
        ASSERT_EQ(prosper::frontend::kUnorm10To8[v], static_cast<uint8_t>((v * 255u + 511u) / 1023u)) << v;
    for (uint32_t a = 0; a < 4; ++a)
        ASSERT_EQ(prosper::frontend::kUnorm2To8[a], static_cast<uint8_t>((a * 255u + 1u) / 3u)) << a;
}

TEST(Unorm10Snapshot, FastPathMatchesTheReferenceImageForImage) {
    // Odd sizes on purpose: a loop that assumed a multiple of 4 or 8 texels would slip on the tail.
    for (size_t texels : {size_t{1}, size_t{3}, size_t{7}, size_t{4099}, size_t{100003}}) {
        const std::vector<uint8_t> packed = random_words(texels, 1234 + texels);
        std::vector<uint8_t> fast;
        ASSERT_TRUE(unpack_unorm10_to_rgba8(packed.data(), packed.size(), texels, 1, fast));
        EXPECT_EQ(fast, reference_unpack(packed)) << "texels=" << texels;
    }
    // Every channel value in every channel position, so a field that is never random is still covered.
    std::vector<uint8_t> sweep;
    for (uint32_t v = 0; v < 1024; ++v) {
        put(sweep, v);                       // R
        put(sweep, v << 10);                 // G
        put(sweep, v << 20);                 // B
        put(sweep, static_cast<uint32_t>(v & 3u) << 30);  // A
    }
    std::vector<uint8_t> fast;
    ASSERT_TRUE(unpack_unorm10_to_rgba8(sweep.data(), sweep.size(), sweep.size() / 4, 1, fast));
    EXPECT_EQ(fast, reference_unpack(sweep));
}

TEST(Unorm10Snapshot, OutputDoesNotDependOnABufferBeingZeroed) {
    // The pooled output buffer is recycled and never zero-filled, so the loop must write every byte.
    const std::vector<uint8_t> packed = random_words(5000, 77);
    std::vector<uint8_t> dirty(packed.size(), 0xAB);
    prosper::frontend::unpack_unorm10_words_to_rgba8(packed.data(), 5000, dirty.data());
    EXPECT_EQ(dirty, reference_unpack(packed));
}

TEST(Unorm10Snapshot, MeasuredAgainstThePreviousLoop) {
    // A 1920x1080 target, the size Black Flag publishes. Timings are printed, not asserted: a timing
    // assertion is a flaky test, and the exactness tests above are what protect the output.
    constexpr size_t kTexels = 1920 * 1080;
    const std::vector<uint8_t> packed = random_words(kTexels, 99);
    std::vector<uint8_t> out(kTexels * 4);
    using Clock = std::chrono::steady_clock;
    auto best_ms = [](auto&& fn) {
        double best = 1e30;
        for (int i = 0; i < 5; ++i) {
            const auto t0 = Clock::now();
            fn();
            best = std::min(best, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        }
        return best;
    };
    std::vector<uint8_t> reference;
    const double old_ms = best_ms([&] { reference = reference_unpack(packed); });
    const double new_ms = best_ms([&] {
        prosper::frontend::unpack_unorm10_words_to_rgba8(packed.data(), kTexels, out.data());
    });
    std::printf("[unorm10] 1920x1080: previous loop %.2f ms, table loop %.2f ms (%.2fx)\n", old_ms, new_ms,
                new_ms > 0 ? old_ms / new_ms : 0.0);
    EXPECT_EQ(out, reference);
}
