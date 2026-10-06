// test_sampled_dcc_fast_clear -- a DCC-compressed sampled T# whose control plane holds one uniform
// fast-clear code is the clear colour, whatever its base bytes hold.
//
// THE DEFECT. Assassin's Creed Black Flag Resynced composites its epilepsy-warning screen with a
// compute pass that samples a fast-cleared RGBA8 surface through a compressed T#. The fast-clear
// decode only admitted RGBA16F, so the pass read the surface's stale base bytes instead of the
// cleared colour and the warning screen's background came out red.
//
// WHAT EACH TEST KILLS:
//   Unorm8x4ClearIsMaterialized     the fast-clear decode refuses RGBA8 again
//   ClearCodesMapToTheirColours     a code decodes to the wrong colour or alpha
//   AlphaFollowsTheDescriptorsComponentOrder  the clear alpha lands in the wrong raw component
//   NonUniformPlaneIsRefused        a plane with any compressed block is read as a clear
//   OtherFormatsStayRefused         the widening admits a format it has no evidence for
#include "shared/compute/sampled_dcc_fast_clear.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace prosper::gpu;
using prosper::frontend::compute_sampled_dcc_fast_clear_rgba8;

namespace {

ShaderResource compressed_rgba8() {
    ShaderResource r;
    r.cls = ResourceClass::Texture;
    r.format = DataFormat::Unorm8;
    r.num_components = 4;
    r.width = r.height = 64;
    r.depth = 1;
    r.img_dim = 1;
    r.declared_mip_levels = 1;
    r.tile_mode = 27;
    r.compression_enabled = true;
    r.metadata_addr = 0x10000;
    r.alpha_is_on_msb = true;   // RGBA raw component order: alpha is the fourth component
    return r;
}

std::vector<uint8_t> plane(const ShaderResource& r, uint8_t code) {
    return std::vector<uint8_t>(static_cast<size_t>(gpu_capture_dcc_metadata_footprint(r)), code);
}

bool decode(const ShaderResource& r, const std::vector<uint8_t>& meta, uint8_t pixel[4],
            uint8_t* code = nullptr) {
    return compute_sampled_dcc_fast_clear_rgba8(r, true, false, false, pixel, 1, meta.data(),
                                                meta.size(), code);
}

}  // namespace

TEST(SampledDccFastClear, Unorm8x4ClearIsMaterialized) {
    const ShaderResource r = compressed_rgba8();
    const auto meta = plane(r, 0x00);
    ASSERT_FALSE(meta.empty()) << "the fixture shape must have a DCC footprint";
    uint8_t pixel[4] = {9, 9, 9, 9};
    ASSERT_TRUE(decode(r, meta, pixel));
    EXPECT_EQ(pixel[0], 0);
    EXPECT_EQ(pixel[1], 0);
    EXPECT_EQ(pixel[2], 0);
    EXPECT_EQ(pixel[3], 0);
}

TEST(SampledDccFastClear, ClearCodesMapToTheirColours) {
    const ShaderResource r = compressed_rgba8();
    struct Case { uint8_t code, rgb, alpha; } cases[] = {
        {0x00, 0, 0}, {0x40, 0, 255}, {0x80, 255, 0}, {0xc0, 255, 255}};
    for (const Case& c : cases) {
        uint8_t pixel[4] = {};
        uint8_t code = 0;
        ASSERT_TRUE(decode(r, plane(r, c.code), pixel, &code)) << "code " << int(c.code);
        EXPECT_EQ(code, c.code);
        EXPECT_EQ(pixel[0], c.rgb);
        EXPECT_EQ(pixel[1], c.rgb);
        EXPECT_EQ(pixel[2], c.rgb);
        EXPECT_EQ(pixel[3], c.alpha);
    }
}

TEST(SampledDccFastClear, AlphaFollowsTheDescriptorsComponentOrder) {
    ShaderResource r = compressed_rgba8();
    r.alpha_is_on_msb = false;   // the clear alpha lands in raw component 0 instead
    uint8_t pixel[4] = {};
    ASSERT_TRUE(decode(r, plane(r, 0x40), pixel));
    EXPECT_EQ(pixel[0], 255);
    EXPECT_EQ(pixel[3], 0);
}

TEST(SampledDccFastClear, NonUniformPlaneIsRefused) {
    const ShaderResource r = compressed_rgba8();
    auto meta = plane(r, 0x00);
    meta.back() = 0xff;   // one block holds real (uncompressed) texels
    uint8_t pixel[4] = {};
    EXPECT_FALSE(decode(r, meta, pixel));
}

TEST(SampledDccFastClear, OtherFormatsStayRefused) {
    uint8_t pixel[4] = {};
    ShaderResource three = compressed_rgba8();
    three.num_components = 3;
    EXPECT_FALSE(decode(three, plane(three, 0x00), pixel));
    ShaderResource wide = compressed_rgba8();
    wide.format = DataFormat::Unorm16;
    EXPECT_FALSE(decode(wide, plane(wide, 0x00), pixel));
}
