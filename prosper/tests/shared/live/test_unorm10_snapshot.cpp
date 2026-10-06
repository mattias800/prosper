// test_unorm10_snapshot -- a compute-written R10G10B10A2 display buffer is narrowed to the RGBA8
// bytes the renderer keeps for that target.
//
// WHAT EACH TEST KILLS:
//   FieldsLandInTheirChannels   R/G/B/A swapped, or read from the wrong bit field
//   ExtremesMapToFullRange      rounding or scale error (1023 -> 255, 3 -> 255, 0 -> 0)
//   WrongSizeIsRefused          a truncated or oversized buffer is published as a picture
#include "shared/live/unorm10_snapshot.hpp"

#include <gtest/gtest.h>

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
