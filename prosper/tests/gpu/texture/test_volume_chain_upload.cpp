// test_volume_chain_upload -- the all-in-tail volume chain uploader's contract.
//
// Levels 1..N-1 of a thick-3D tail chain upload through their own in-block origins
// (upload_volume_tail_chain_levels); level zero travels the single-level path. These arms
// pin, against hand-built allocations rather than any live title: the staging math (exact
// offsets, alignment, totals), native byte-identity per level, the R8 1-to-4 broadcast per
// level, and every fail-visible refusal (unmodelled conversion, overrun, non-volume plan).
// A wrong per-level origin binds silent wrong texels, so each level's bytes are asserted,
// not just the return code.
#include "gpu/resources/volume_chain_upload.hpp"
#include "gpu/texture/tile.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace prosper::gpu;

namespace {

// A 16^3 R8 four-level pyramid (SW_64KB_S), tiled level by level into one 64 KiB block
// through the proven tail origins, so each level's expected linear bytes are known
// independently of the uploader: detile the same block through the same origin.
constexpr uint32_t kW = 16, kH = 16, kD = 16, kBpe = 1, kTile = 9, kMaxMip = 3;

MipChainPlan make_chain() {
    MipChainPlan plan;
    plan.valid = true;
    plan.level_count = kMaxMip + 1;
    plan.allocation_bytes = 65536;
    plan.levels.resize(plan.level_count);
    for (uint32_t level = 0; level < plan.level_count; ++level) {
        const TiledVolumeTailLayout layout =
            tiled_volume_tail_layout(kW, kH, kD, kBpe, kTile, kMaxMip, level);
        MipChainLevel& out = plan.levels[level];
        out.width = std::max(kW >> level, 1u);
        out.height = std::max(kH >> level, 1u);
        out.depth = std::max(kD >> level, 1u);
        out.in_tail = true;
        out.byte_offset = 0;
        out.byte_size = 65536;
        out.tail_x = layout.tail_x;
        out.tail_y = layout.tail_y;
        out.tail_block_bytes = 65536;
    }
    return plan;
}

} // namespace

TEST(VolumeChainUpload, StagingLayoutIsExact) {
    const MipChainPlan plan = make_chain();
    ASSERT_TRUE(mip_chain_plan_is_volume_tail(plan)) << "fixture really is a volume chain";
    // Backend width 1, alignment 4: L0 4096, L1 512 @4096, L2 64 @4608, L3 8 @4672.
    const VolumeChainStaging staging = volume_chain_staging_layout(plan, 1, 4);
    ASSERT_EQ(staging.offsets.size(), 4u) << "one offset per level";
    EXPECT_EQ(staging.offsets[0], 0u) << "level zero opens the upload";
    EXPECT_EQ(staging.offsets[1], 4096u) << "L1 follows the 4096-byte L0";
    EXPECT_EQ(staging.offsets[2], 4608u) << "L2 follows the 512-byte L1";
    EXPECT_EQ(staging.offsets[3], 4672u) << "L3 follows the 64-byte L2";
    EXPECT_EQ(staging.total_bytes, 4680u) << "total is the exact packed span";
    // Backend width 4 (R8 broadcast): L0 16384, L1 2048 @16384, L2 256 @18432,
    // L3 32 @18688, total 18720.
    const VolumeChainStaging wide = volume_chain_staging_layout(plan, 4, 4);
    EXPECT_EQ(wide.offsets[1], 16384u) << "wide L1 follows the 16384-byte L0";
    EXPECT_EQ(wide.offsets[2], 18432u) << "wide L2 follows the 2048-byte L1";
    EXPECT_EQ(wide.offsets[3], 18688u) << "wide L3 follows the 256-byte L2";
    EXPECT_EQ(wide.total_bytes, 18720u) << "wide total is exact";
    // Refusals, not guesses.
    EXPECT_TRUE(volume_chain_staging_layout(plan, 1, 0).offsets.empty())
        << "zero alignment refuses";
    EXPECT_TRUE(volume_chain_staging_layout(plan, 0, 4).offsets.empty())
        << "zero texel width refuses";
    MipChainPlan invalid;
    EXPECT_TRUE(volume_chain_staging_layout(invalid, 1, 4).offsets.empty())
        << "invalid plan refuses";
    EXPECT_FALSE(mip_chain_plan_is_volume_tail(invalid)) << "invalid plan is not volume";
    MipChainPlan flat = plan;
    flat.levels[2].depth = 0;
    EXPECT_FALSE(mip_chain_plan_is_volume_tail(flat)) << "a depthless level is not a volume chain";
}

TEST(VolumeChainUpload, NativeLevelsAreByteIdentical) {
    const MipChainPlan plan = make_chain();
    // Fill the block with a function of the TILED byte index, then require the
    // uploader's per-level output to equal a direct detile of the same block through the
    // same origin. The direct detile is the independent path (mechanism identity proven
    // in VolumeTailDetileMechanism*); this arm proves the uploader routes each level to
    // its own origin and staging offset.
    std::vector<uint8_t> block(65536);
    for (size_t i = 0; i < block.size(); ++i) block[i] = static_cast<uint8_t>(i * 13u + 5u);
    const VolumeChainStaging staging = volume_chain_staging_layout(plan, 1, 4);
    ASSERT_EQ(staging.total_bytes, 4680u) << "staging math agreed above";
    std::vector<uint8_t> upload(staging.total_bytes, 0xA5);
    ASSERT_TRUE(upload_volume_tail_chain_levels(plan, block.data(), block.size(), kTile, kBpe, 1,
                                                false, upload.data(), staging.offsets.data(),
                                                staging.offsets.size()))
        << "native upload of levels 1..3 succeeds";
    for (uint32_t level = 1; level < plan.level_count; ++level) {
        const MipChainLevel& source = plan.levels[level];
        std::vector<uint8_t> expect(size_t(source.width) * source.height * source.depth);
        ASSERT_TRUE(detile_volume_tail_level(expect.data(), block.data(), block.size(),
                                             source.width, source.height, source.depth, kTile, kBpe,
                                             source.tail_x, source.tail_y))
            << "direct detile admits level " << level;
        EXPECT_EQ(std::memcmp(upload.data() + staging.offsets[level], expect.data(), expect.size()),
                  0)
            << "level " << level << " bytes equal the direct detile";
    }
}

TEST(VolumeChainUpload, R8BroadcastRepeatsPerLevel) {
    const MipChainPlan plan = make_chain();
    std::vector<uint8_t> block(65536);
    for (size_t i = 0; i < block.size(); ++i) block[i] = static_cast<uint8_t>(i * 13u + 5u);
    const VolumeChainStaging staging = volume_chain_staging_layout(plan, 4, 4);
    ASSERT_EQ(staging.total_bytes, 18720u) << "wide staging math agreed above";
    std::vector<uint8_t> upload(staging.total_bytes, 0xA5);
    ASSERT_TRUE(upload_volume_tail_chain_levels(plan, block.data(), block.size(), kTile, 1, 4, true,
                                                upload.data(), staging.offsets.data(),
                                                staging.offsets.size()))
        << "R8 broadcast upload succeeds";
    for (uint32_t level = 1; level < plan.level_count; ++level) {
        const MipChainLevel& source = plan.levels[level];
        std::vector<uint8_t> linear(size_t(source.width) * source.height * source.depth);
        ASSERT_TRUE(detile_volume_tail_level(linear.data(), block.data(), block.size(),
                                             source.width, source.height, source.depth, kTile, 1,
                                             source.tail_x, source.tail_y))
            << "direct detile admits level " << level;
        const uint8_t* out = upload.data() + staging.offsets[level];
        bool replicated = true;
        for (size_t t = 0; t < linear.size(); ++t) {
            if (!(out[t * 4u + 0] == linear[t] && out[t * 4u + 1] == linear[t] &&
                  out[t * 4u + 2] == linear[t] && out[t * 4u + 3] == linear[t])) {
                replicated = false;
                break;
            }
        }
        EXPECT_TRUE(replicated) << "level " << level << " broadcasts every texel 1-to-4";
    }
}

TEST(VolumeChainUpload, RefusalsAreFailVisible) {
    const MipChainPlan plan = make_chain();
    const VolumeChainStaging staging = volume_chain_staging_layout(plan, 1, 4);
    std::vector<uint8_t> block(65536, 0x11);
    std::vector<uint8_t> upload(staging.total_bytes, 0xA5);
    // An unmodelled conversion (2-byte guest to 4-byte backend without broadcast proof).
    EXPECT_FALSE(upload_volume_tail_chain_levels(plan, block.data(), block.size(), kTile, 2, 4,
                                                 false, upload.data(), staging.offsets.data(),
                                                 staging.offsets.size()))
        << "unmodelled width conversion refuses";
    // A short offset vector cannot name every level.
    const std::vector<uint64_t> short_offsets{0, 4096};
    EXPECT_FALSE(upload_volume_tail_chain_levels(plan, block.data(), block.size(), kTile, 1, 1,
                                                 false, upload.data(), short_offsets.data(),
                                                 short_offsets.size()))
        << "truncated offsets refuse";
    // A non-volume plan takes the 2D uploader, never this one.
    MipChainPlan flat = plan;
    for (auto& level : flat.levels) level.depth = 0;
    EXPECT_FALSE(upload_volume_tail_chain_levels(flat, block.data(), block.size(), kTile, 1, 1,
                                                 false, upload.data(), staging.offsets.data(),
                                                 staging.offsets.size()))
        << "depthless plan refuses";
    // The wrong tile mode detiles nothing.
    EXPECT_FALSE(upload_volume_tail_chain_levels(plan, block.data(), block.size(), 5, 1, 1, false,
                                                 upload.data(), staging.offsets.data(),
                                                 staging.offsets.size()))
        << "non-S3 tile mode refuses";
    EXPECT_EQ(upload[staging.offsets[1]], 0xA5)
        << "no refusal arm leaves partial level bytes behind";
}
