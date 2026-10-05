// #4496: the native BC mip-chain audit compares EVERY level with the 2x2 box of the level above it.
// Solid-colour BC1 blocks make a misplaced or foreign level unmistakable.
#include "shared/live/submit_renderer/native_bc_chain_audit.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

using prosper::frontend::submit_renderer::audit_native_bc_chain;
using prosper::frontend::submit_renderer::NativeBcChainAudit;
using prosper::gpu::DataFormat;

namespace {
constexpr uint32_t kBlockBytes = 8;   // BC1

// One BC1 block whose sixteen texels are all `grey`: equal endpoints, every index zero.
std::array<uint8_t, kBlockBytes> solid_block(uint8_t grey) {
    const auto c565 = static_cast<uint16_t>(((grey >> 3) << 11) | ((grey >> 2) << 5) | (grey >> 3));
    const auto low = static_cast<uint8_t>(c565 & 0xffu);
    const auto high = static_cast<uint8_t>(c565 >> 8);
    return {low, high, low, high, 0, 0, 0, 0};
}

// A 16x16 chain of five levels (16, 8, 4, 2, 1), level 0 first; `level_grey[i]` fills level i.
// Level 0 can instead be split into a black left half and a white right half.
std::vector<uint8_t> chain(const std::array<uint8_t, 5>& level_grey,
                           bool split_wide_levels = false) {
    std::vector<uint8_t> bytes;
    for (uint32_t level = 0; level < 5; ++level) {
        const uint32_t extent = 16u >> level;
        const uint32_t blocks = (extent + 3u) / 4u;
        for (uint32_t y = 0; y < blocks; ++y)
            for (uint32_t x = 0; x < blocks; ++x) {
                const bool split = split_wide_levels && blocks >= 2u;
                const uint8_t grey =
                    split ? static_cast<uint8_t>(x * 2u >= blocks ? 255 : 0) : level_grey[level];
                const auto block = solid_block(grey);
                bytes.insert(bytes.end(), block.begin(), block.end());
            }
    }
    return bytes;
}

NativeBcChainAudit audit(const std::vector<uint8_t>& bytes, uint32_t levels = 5) {
    return audit_native_bc_chain(bytes.data(), bytes.size(), 16, 16, levels, kBlockBytes,
                                 DataFormat::Bc1);
}
}   // namespace

TEST(NativeBcChainAudit, AConsistentChainIsQuietAtEveryLevel) {
    const NativeBcChainAudit result = audit(chain({128, 128, 128, 128, 128}));
    ASSERT_EQ(result.level_mad.size(), 4u);   // levels 1..4
    for (const double mad : result.level_mad) EXPECT_LT(mad, 1.0);
}

TEST(NativeBcChainAudit, AWrongDeepLevelIsNamedAndLevelOneAloneWouldMissIt) {
    // Level 3 holds foreign data. The old audit looked at level 1 only and reported this chain
    // clean; a distant surface samples exactly these small levels.
    const NativeBcChainAudit result = audit(chain({128, 128, 128, 255, 128}));
    ASSERT_EQ(result.level_mad.size(), 4u);
    EXPECT_LT(result.level_mad[0], 1.0);   // level 1 against level 0
    EXPECT_LT(result.level_mad[1], 1.0);   // level 2 against level 1
    EXPECT_GT(result.level_mad[2], 100.0);   // level 3 against level 2
    EXPECT_GT(result.level_mad[3], 100.0);   // level 4 against the foreign level 3
}

TEST(NativeBcChainAudit, TheShiftedControlSeparatesFlatFromPlaced) {
    // A flat texture cannot show placement: the level-1 difference and its shifted control are both
    // zero. A structured one can: level 1 matches its box, and the half-width shift does not.
    const NativeBcChainAudit flat = audit(chain({64, 64, 64, 64, 64}));
    EXPECT_LT(flat.level_mad[0], 1.0);
    EXPECT_LT(flat.shifted_control, 1.0);
    const NativeBcChainAudit structured = audit(chain({0, 0, 0, 0, 0}, true));
    EXPECT_LT(structured.level_mad[0], 1.0);
    EXPECT_GT(structured.shifted_control, 200.0);
}

TEST(NativeBcChainAudit, OnlyTheLevelsTheBytesCoverAreReported) {
    std::vector<uint8_t> bytes = chain({128, 128, 128, 128, 128});
    bytes.resize(size_t{16 + 4 + 1} * kBlockBytes);   // levels 0, 1 and 2 only
    EXPECT_EQ(audit(bytes).level_mad.size(), 2u);
    EXPECT_TRUE(audit(bytes, 1).level_mad.empty());   // a single level has nothing to compare
    EXPECT_TRUE(audit_native_bc_chain(bytes.data(), bytes.size(), 16, 16, 5, 4, DataFormat::Unorm8)
                    .level_mad.empty());   // not a block format
}
