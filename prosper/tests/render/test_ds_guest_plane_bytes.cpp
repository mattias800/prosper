// A retained depth image's guest footprint follows the GUEST depth format (#4556).
//
// The host image is D32 whatever the guest programmed, and the invalidator sized every depth plane
// at four bytes a texel. For a Z_16 plane that is twice its size, and the second half is never
// that surface. Where it reaches the guest's next allocation, that neighbour's writes discard the
// depth. Where it runs off the end of the mapping, the mapping table cannot place the range at
// all, and the invalidator -- which preserves an aspect only on a proven-disjoint answer -- then
// discards the depth on EVERY guest write, wherever it lands.
//
// MOUSE: P.I. For Hire renders its shadow atlases as Z_16 and samples them in the lighting pass,
// and each was invalid by then. What had invalidated them was writes to other memory altogether,
// the commonest a 4-byte label.
//
// The entries are built through persistent_ds_entry_for, the function the backend attaches with,
// so removing the recording there fails these arms. What they cannot see is the backend ceasing
// to call it: that one line in render_draw_pass_rgba is covered by the title, not by this file.
#include "fixtures/render_runner.h"
#include "hle/dispatch/dispatch.hpp"

#include <cstdint>
#include <gtest/gtest.h>

namespace {
using prosper::test::invalidate_persistent_ds_guest_write;
using prosper::test::PersistentDsImage;

// DB_Z_INFO.FORMAT.
constexpr uint32_t kUndescribed = 0, kZ16 = 1, kZ32Float = 3;
constexpr uint32_t kWidth = 256, kHeight = 256;
constexpr uint64_t kPixels = uint64_t{kWidth} * kHeight;

// The decode itself, including the value MOUSE's shadow passes program: only the format field
// counts, whatever else the register carries.
static_assert(prosper::gpu::guest_depth_texel_bytes(kUndescribed) == 0);
static_assert(prosper::gpu::guest_depth_texel_bytes(kZ16) == 2);
static_assert(prosper::gpu::guest_depth_texel_bytes(2) == 4);
static_assert(prosper::gpu::guest_depth_texel_bytes(kZ32Float) == 4);
static_assert(prosper::gpu::guest_depth_texel_bytes(0xa2900801u) == 2);
static_assert(prosper::gpu::guest_depth_texel_bytes(0xa2900800u) == 0);
// NUM_SAMPLES is bits 3:2. A multisampled Z_16 plane is not narrowed.
static_assert(prosper::gpu::guest_depth_texel_bytes(kZ16 | (1u << 2)) == 4);
static_assert(prosper::gpu::guest_depth_texel_bytes(kZ16 | (2u << 2)) == 4);
static_assert(prosper::gpu::guest_depth_texel_bytes(kZ32Float | (2u << 2)) == 4);

// One tracked guest allocation. The invalidator asks the mapping table whether a write can alias
// a plane, and a range the table does not cover in full is answered "it may".
class DsGuestPlaneBytes : public ::testing::Test {
protected:
    static constexpr uint64_t kMappedBytes = 0x400000;

    void SetUp() override {
        prosper::register_builtin_hle();
        const auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
        unmap_ = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
        ASSERT_TRUE(map && unmap_);
        ASSERT_EQ(map(reinterpret_cast<uint64_t>(&base_), kMappedBytes, 2, 0,
                      reinterpret_cast<uint64_t>("ds-guest-plane-bytes"), 0),
                  0u);
        ASSERT_NE(base_, 0u);
        prosper::test::persistent_ds_cache().clear();
    }
    void TearDown() override {
        prosper::test::persistent_ds_cache().clear();
        if (base_) unmap_(base_, kMappedBytes, 0, 0, 0, 0);
    }

    // A depth-only surface at `depth` holding valid depth, attached by a pass that programmed
    // `z_format`.
    static PersistentDsImage& attach(uint64_t depth, uint32_t z_format) {
        prosper::gpu::ResolvedPipelineState ps;
        ps.depth_read_base = ps.depth_write_base = depth;
        ps.db_z_info = z_format;
        PersistentDsImage& image = prosper::test::persistent_ds_entry_for(
            ps, prosper::test::persistent_ds_key_for(ps, /*htile=*/0, kWidth, kHeight,
                                                     VK_FORMAT_D32_SFLOAT));
        image.depth_valid = true;
        return image;
    }

    uint64_t base_ = 0;
    prosper::HleFn unmap_ = nullptr;
};

TEST_F(DsGuestPlaneBytes, AZ16PlaneEndsAtTwoBytesATexel) {
    PersistentDsImage& image = attach(base_, kZ16);
    // The first byte past the plane, which a four-byte texel would still claim.
    EXPECT_EQ(invalidate_persistent_ds_guest_write(base_ + kPixels * 2, 4), 0u);
    EXPECT_TRUE(image.depth_valid);
    // Its last byte.
    EXPECT_EQ(invalidate_persistent_ds_guest_write(base_ + kPixels * 2 - 1, 1), 1u);
    EXPECT_FALSE(image.depth_valid);
}

TEST_F(DsGuestPlaneBytes, AZ32PlaneStillEndsAtFour) {
    PersistentDsImage& image = attach(base_, kZ32Float);
    EXPECT_EQ(invalidate_persistent_ds_guest_write(base_ + kPixels * 4, 4), 0u);
    EXPECT_TRUE(image.depth_valid);
    EXPECT_EQ(invalidate_persistent_ds_guest_write(base_ + kPixels * 4 - 1, 1), 1u);
    EXPECT_FALSE(image.depth_valid);
}

// A two-sample Z_16 plane occupies four bytes a pixel, so its range must not be halved.
TEST_F(DsGuestPlaneBytes, AMultisampledZ16PlaneKeepsTheFourByteRange) {
    PersistentDsImage& image = attach(base_, kZ16 | (1u << 2));
    EXPECT_EQ(image.guest_depth_texel_bytes, 4u);
    EXPECT_EQ(invalidate_persistent_ds_guest_write(base_ + kPixels * 4 - 1, 1), 1u);
    EXPECT_FALSE(image.depth_valid);
}

// Nothing has said what the guest plane is, so nothing licenses a shorter range.
TEST_F(DsGuestPlaneBytes, APlaneNoPassDescribedKeepsTheFourByteRange) {
    PersistentDsImage& image = attach(base_, kUndescribed);
    EXPECT_EQ(image.guest_depth_texel_bytes, 0u);
    EXPECT_EQ(invalidate_persistent_ds_guest_write(base_ + kPixels * 4 - 1, 1), 1u);
    EXPECT_FALSE(image.depth_valid);
}

TEST_F(DsGuestPlaneBytes, TheLastPassToDescribeThePlaneDecides) {
    PersistentDsImage& image = attach(base_, kZ32Float);
    EXPECT_EQ(image.guest_depth_texel_bytes, 4u);
    EXPECT_EQ(&attach(base_, kZ16), &image);
    EXPECT_EQ(image.guest_depth_texel_bytes, 2u);
    attach(base_, kUndescribed);
    EXPECT_EQ(image.guest_depth_texel_bytes, 2u);
}

// The MOUSE shape: the plane is the last thing in its mapping, and the write is nowhere near it.
TEST_F(DsGuestPlaneBytes, AZ16PlaneAtTheEndOfItsMappingSurvivesAWriteElsewhere) {
    const uint64_t plane = base_ + kMappedBytes - kPixels * 2;
    // The premise, on these same addresses: a four-byte plane here runs off the mapping, the
    // table cannot place it, and a write at the far end of the allocation discards it. Were that
    // to stop holding, the arm below would pass whatever the plane's size.
    {
        PersistentDsImage& wide = attach(plane, kZ32Float);
        EXPECT_EQ(invalidate_persistent_ds_guest_write(base_, 4), 1u);
        EXPECT_FALSE(wide.depth_valid);
    }
    PersistentDsImage& image = attach(plane, kZ16);
    EXPECT_EQ(invalidate_persistent_ds_guest_write(base_, 4), 0u);
    EXPECT_TRUE(image.depth_valid);
}
}   // namespace
