// A replay process answers "can this write land in that retained plane?" by address (#4619).
//
// The live rule asks the guest mapping table, because distinct guest addresses can name the same
// physical bytes, and it keeps a retained depth image only on a proven-disjoint answer. A process
// that replays a capture has no mapping table at all. There every answer was Unknown, so every
// guest write discarded every retained depth image: a 20-byte write, six of six entries, on each
// of the 70 writes of one replayed MOUSE: P.I. For Hire frame. A pass that sampled depth after
// any compute writeback then read it as missing, and the replayed frame came out nearly white
// where the live one was nearly black.
//
// Nothing here maps guest memory, on purpose: that is the replay process's situation, and the
// first case states what it means for the live rule before the others rely on it.
#include "fixtures/render_runner.h"

#include <cstdint>
#include <gtest/gtest.h>

namespace {
using prosper::test::invalidate_persistent_ds_guest_write;
using prosper::test::PersistentDsImage;

constexpr uint32_t kWidth = 256, kHeight = 256;
constexpr uint64_t kPixels = uint64_t{kWidth} * kHeight;
// Guest addresses as a capture carries them. Nothing is mapped at any of them.
constexpr uint64_t kDepth = 0x4043160000ull, kStencil = 0x403f460000ull, kHtile = 0x4011508000ull;
constexpr uint64_t kElsewhere = 0x4011739040ull;

class DsReplayAlias : public ::testing::Test {
protected:
    void SetUp() override {
        prosper::test::persistent_ds_cache().clear();
        prosper::gpu::guest_plane_alias_by_address() = false;
    }
    void TearDown() override {
        prosper::gpu::guest_plane_alias_by_address() = false;
        prosper::test::persistent_ds_cache().clear();
    }
    // A Z_32 depth and stencil surface with HTILE, both aspects valid.
    static PersistentDsImage& retained() {
        prosper::gpu::ResolvedPipelineState ps;
        ps.depth_read_base = ps.depth_write_base = kDepth;
        ps.stencil_read_base = ps.stencil_write_base = kStencil;
        ps.db_z_info = 3;
        PersistentDsImage& image = prosper::test::persistent_ds_entry_for(
            ps, prosper::test::persistent_ds_key_for(ps, kHtile, kWidth, kHeight,
                                                     VK_FORMAT_D32_SFLOAT_S8_UINT));
        image.depth_valid = image.stencil_valid = true;
        return image;
    }
};

// The premise. With no mapping table the live rule cannot place either range, and says so the
// only safe way: the image is discarded by a write that is nowhere near it.
TEST_F(DsReplayAlias, WithNoMappingTableTheLiveRuleDiscardsOnAnyWrite) {
    PersistentDsImage& image = retained();
    EXPECT_EQ(invalidate_persistent_ds_guest_write(kElsewhere, 20), 1u);
    EXPECT_FALSE(image.depth_valid);
    EXPECT_FALSE(image.stencil_valid);
}

TEST_F(DsReplayAlias, AReplayProcessKeepsAnImageAWriteDoesNotTouch) {
    prosper::gpu::guest_plane_alias_by_address() = true;
    PersistentDsImage& image = retained();
    EXPECT_EQ(invalidate_persistent_ds_guest_write(kElsewhere, 20), 0u);
    // The bytes just before and just after the depth plane.
    EXPECT_EQ(invalidate_persistent_ds_guest_write(kDepth - 16, 16), 0u);
    EXPECT_EQ(invalidate_persistent_ds_guest_write(kDepth + kPixels * 4, 16), 0u);
    EXPECT_TRUE(image.depth_valid);
    EXPECT_TRUE(image.stencil_valid);
}

TEST_F(DsReplayAlias, AReplayProcessStillDiscardsWhatAWriteCovers) {
    prosper::gpu::guest_plane_alias_by_address() = true;
    // The last byte of the depth plane: depth goes, stencil stays.
    PersistentDsImage& depth = retained();
    EXPECT_EQ(invalidate_persistent_ds_guest_write(kDepth + kPixels * 4 - 1, 1), 1u);
    EXPECT_FALSE(depth.depth_valid);
    EXPECT_TRUE(depth.stencil_valid);
    // A write that straddles the start of the stencil plane: stencil goes, depth stays.
    PersistentDsImage& stencil = retained();
    EXPECT_EQ(invalidate_persistent_ds_guest_write(kStencil - 8, 16), 1u);
    EXPECT_TRUE(stencil.depth_valid);
    EXPECT_FALSE(stencil.stencil_valid);
    // A rewrite of the HTILE plane describes both aspects.
    PersistentDsImage& htile = retained();
    EXPECT_EQ(invalidate_persistent_ds_guest_write(kHtile, 4096), 1u);
    EXPECT_FALSE(htile.depth_valid);
    EXPECT_FALSE(htile.stencil_valid);
}

// The predicate itself at its edges, including the two the mapping table also answers "it may".
TEST_F(DsReplayAlias, TheAddressAnswerAtItsEdges) {
    prosper::gpu::guest_plane_alias_by_address() = true;
    using prosper::gpu::guest_write_may_alias_plane;
    EXPECT_TRUE(guest_write_may_alias_plane(0x1000, 0x100, 0x10ff, 1));
    EXPECT_FALSE(guest_write_may_alias_plane(0x1000, 0x100, 0x1100, 1));
    EXPECT_FALSE(guest_write_may_alias_plane(0x1100, 1, 0x1000, 0x100));
    EXPECT_TRUE(guest_write_may_alias_plane(0x1000, 0, 0x9000, 0x100));
    EXPECT_TRUE(guest_write_may_alias_plane(0x1000, 0x100, 0x9000, 0));
    // A range that runs off the top of the address space ends there; it does not wrap to zero.
    EXPECT_TRUE(guest_write_may_alias_plane(UINT64_MAX - 8, 64, UINT64_MAX - 4, 1));
    EXPECT_FALSE(guest_write_may_alias_plane(UINT64_MAX - 8, 64, 0x1000, 0x100));
}
}  // namespace
