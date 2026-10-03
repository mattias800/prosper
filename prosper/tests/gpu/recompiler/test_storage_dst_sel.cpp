// test_storage_dst_sel -- the storage-image DST_SEL routing rules, without a Vulkan device.
// test_storage_image_copy executes the same rules end to end; this pins them where they are decided.
#include "gpu/recompiler/storage_dst_sel.hpp"

#include <gtest/gtest.h>

using namespace prosper::gpu;

TEST(StorageDstSel, AlphaOnlyStoreTakesTheWComponent) {
    const uint32_t alpha8[4] = {0, 0, 0, 4};
    int source[4];
    ASSERT_TRUE(storage_store_sources(alpha8, 1, source));
    EXPECT_EQ(source[0], 3) << "stored X <- VDATA.w (Silksong's glyph upload)";
    EXPECT_EQ(source[1], kDstSelNoSource);
}

TEST(StorageDstSel, StoreIsTheInverseOfAPermutation) {
    const uint32_t bgra[4] = {6, 5, 4, 7};
    int source[4];
    ASSERT_TRUE(storage_store_sources(bgra, 4, source));
    EXPECT_EQ(source[0], 2);
    EXPECT_EQ(source[1], 1);
    EXPECT_EQ(source[2], 0);
    EXPECT_EQ(source[3], 3);
    const uint32_t identity[4] = {4, 5, 6, 7};
    ASSERT_TRUE(storage_store_sources(identity, 4, source));
    for (int c = 0; c < 4; ++c) EXPECT_EQ(source[c], c) << "identity reduces to the old store";
}

TEST(StorageDstSel, StoreRefusesWhatItCannotDecode) {
    int source[4];
    const uint32_t constant_w[4] = {4, 5, 6, 1};
    EXPECT_FALSE(storage_store_sources(constant_w, 4, source)) << "stored W named by no selector";
    EXPECT_TRUE(storage_store_sources(constant_w, 3, source)) << "a 3-channel format stores no W";
    const uint32_t duplicate[4] = {4, 4, 6, 7};
    EXPECT_FALSE(storage_store_sources(duplicate, 4, source));
    const uint32_t reserved[4] = {4, 5, 3, 7};
    EXPECT_FALSE(storage_store_sources(reserved, 4, source));
}

TEST(StorageDstSel, LoadRoutesForwardWithTheFormatsOne) {
    const uint32_t sel[4] = {1, 0, 5, 4};
    int channel[4];
    uint32_t constant[4];
    ASSERT_TRUE(storage_load_selects(sel, DataFormat::Unorm8, channel, constant));
    EXPECT_EQ(channel[0], kDstSelNoSource);
    EXPECT_EQ(constant[0], 0x3f800000u) << "SQ_SEL_1 on a float/UNORM format is 1.0f";
    EXPECT_EQ(channel[1], kDstSelNoSource);
    EXPECT_EQ(constant[1], 0u);
    EXPECT_EQ(channel[2], 1);
    EXPECT_EQ(channel[3], 0);
    ASSERT_TRUE(storage_load_selects(sel, DataFormat::Uint32, channel, constant));
    EXPECT_EQ(constant[0], 1u) << "SQ_SEL_1 on an integer format is 1";
    const uint32_t reserved[4] = {2, 5, 6, 7};
    EXPECT_FALSE(storage_load_selects(reserved, DataFormat::Unorm8, channel, constant));
}
