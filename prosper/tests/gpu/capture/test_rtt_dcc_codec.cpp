// The DCC association is an append-only v79 value record; old captures carry no association.
#include "gpu/capture/gpu_capture.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {
using namespace prosper::gpu;
constexpr std::size_t kTailBytes = 4 + 32; // count and one fixed-width descriptor record
GpuCaptureFile capture() {
    GpuCaptureFile c;
    c.metadata.width = c.metadata.height = 64;
    GpuCaptureRttSeed seed;
    seed.guest_addr = 0x10000;
    seed.width = seed.height = 64;
    seed.rgba.assign(64 * 64 * 4, 0x81);
    seed.dcc_metadata = {0x30000, 4096, 4, true, 64, 64};
    c.rtt_seeds.push_back(seed);
    return c;
}
std::vector<uint8_t> encode(const GpuCaptureFile& c) {
    std::vector<uint8_t> bytes;
    std::string error;
    EXPECT_TRUE(serialize_gpu_capture(c, bytes, error)) << error;
    return bytes;
}
uint32_t word(const std::vector<uint8_t>& data, std::size_t at) {
    return uint32_t(data[at]) | uint32_t(data[at + 1]) << 8 |
           uint32_t(data[at + 2]) << 16 | uint32_t(data[at + 3]) << 24;
}
} // namespace

TEST(RttDccCodec, NamedAndAbsentAssociationsRoundTripInTheirOwnTail) {
    auto c = capture();
    auto other = c.rtt_seeds.front();
    other.guest_addr = 0x20000;
    other.dcc_metadata = {};
    c.rtt_seeds.push_back(other);
    const auto bytes = encode(c);
    GpuCaptureFile read;
    std::string error;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, read, error)) << error;
    ASSERT_EQ(read.rtt_seeds.size(), 2u);
    EXPECT_EQ(read.format_version, 79u);
    EXPECT_EQ(read.rtt_seeds[0].dcc_metadata, c.rtt_seeds[0].dcc_metadata);
    EXPECT_EQ(read.rtt_seeds[1].dcc_metadata, GpuCaptureRttDccMetadata{});
}
TEST(RttDccCodec, KnownPlaneWithoutDescriptorInterpretationRemainsPartial) {
    auto c = capture();
    c.rtt_seeds[0].dcc_metadata = {0x30000, 4096, 0, false, 0, 0};
    const auto bytes = encode(c);
    GpuCaptureFile read;
    std::string error;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, read, error)) << error;
    ASSERT_EQ(read.rtt_seeds.size(), 1u);
    EXPECT_EQ(read.rtt_seeds[0].dcc_metadata, c.rtt_seeds[0].dcc_metadata);
}
TEST(RttDccCodec, TailUsesTheDocumentedLittleEndianFieldOrder) {
    const auto bytes = encode(capture());
    ASSERT_GT(bytes.size(), kTailBytes);
    const auto start = bytes.size() - kTailBytes;
    EXPECT_EQ(word(bytes, start), 1u);
    EXPECT_EQ(word(bytes, start + 4), 0x30000u);
    EXPECT_EQ(word(bytes, start + 8), 0u);
    EXPECT_EQ(word(bytes, start + 12), 4096u);
    EXPECT_EQ(word(bytes, start + 16), 0u);
    EXPECT_EQ(word(bytes, start + 20), 4u);
    EXPECT_EQ(word(bytes, start + 24), 1u);
    EXPECT_EQ(word(bytes, start + 28), 64u);
    EXPECT_EQ(word(bytes, start + 32), 64u);
}
TEST(RttDccCodec, MissingAssociationDoesNotForceANewerVersion) {
    auto c = capture();
    c.rtt_seeds[0].dcc_metadata = {};
    const auto bytes = encode(c);
    GpuCaptureFile read;
    std::string error;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, read, error)) << error;
    ASSERT_EQ(read.rtt_seeds.size(), 1u);
    EXPECT_LT(read.format_version, 79u);
    EXPECT_EQ(read.rtt_seeds[0].dcc_metadata, GpuCaptureRttDccMetadata{});
}
TEST(RttDccCodec, Version78PrefixClearsAssociationInAReusedOutput) {
    auto bytes = encode(capture());
    ASSERT_GT(bytes.size(), kTailBytes);
    bytes.resize(bytes.size() - kTailBytes); // the complete v78 prefix, with no v79 tail
    bytes[8] = 78;
    bytes[9] = bytes[10] = bytes[11] = 0;
    auto read = capture();
    std::string error;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, read, error)) << error;
    ASSERT_EQ(read.rtt_seeds.size(), 1u);
    EXPECT_EQ(read.format_version, 78u);
    EXPECT_EQ(read.rtt_seeds[0].dcc_metadata, GpuCaptureRttDccMetadata{});
}
TEST(RttDccCodec, TruncatedAndMismatchedCountTailsRefuse) {
    const auto original = encode(capture());
    ASSERT_GT(original.size(), kTailBytes);
    for (bool truncated : {false, true}) {
        auto bytes = original;
        if (truncated) bytes.pop_back();
        else bytes[bytes.size() - kTailBytes] = 2;
        GpuCaptureFile read;
        std::string error;
        EXPECT_FALSE(deserialize_gpu_capture(bytes, read, error));
        EXPECT_FALSE(error.empty());
    }
}
TEST(RttDccCodec, InvalidAlphaEncodingRefuses) {
    auto bytes = encode(capture());
    ASSERT_GT(bytes.size(), kTailBytes);
    bytes[bytes.size() - kTailBytes + 24] = 2;
    GpuCaptureFile read;
    std::string error;
    EXPECT_FALSE(deserialize_gpu_capture(bytes, read, error));
    EXPECT_NE(error.find("alpha convention"), std::string::npos) << error;
}
TEST(RttDccCodec, IncompleteAndWrappingRecordsRefuseAtBothBoundaries) {
    const auto original = capture();
    for (uint32_t bad : {0u, 1u, 2u, 3u, 4u, 5u}) {
        auto c = original;
        auto& m = c.rtt_seeds[0].dcc_metadata;
        switch (bad) {
            case 0: m.metadata_addr = 0; break;
            case 1: m.metadata_bytes = 0; break;
            case 2: m.num_components = 5; break;
            case 3: m.guest_width = 0; break;
            case 4: m.guest_height = 0; break;
            case 5: m.metadata_addr = UINT64_MAX - 3; break;
        }
        std::vector<uint8_t> bytes;
        std::string error;
        EXPECT_FALSE(serialize_gpu_capture(c, bytes, error)) << bad;
    }
    const auto bytes = encode(original);
    ASSERT_GT(bytes.size(), kTailBytes);
    for (std::size_t field : {4u, 12u, 20u, 28u, 32u}) {
        auto changed = bytes;
        const auto at = changed.size() - kTailBytes + field;
        std::fill_n(changed.begin() + at, 4, uint8_t{0});
        GpuCaptureFile read;
        std::string error;
        EXPECT_FALSE(deserialize_gpu_capture(changed, read, error)) << field;
    }
}
