// test_mimg_nsa_forms_render -- three fragment MIMG forms that were refused in #4808's census, each
// in the NSA (non-sequential address) encoding the titles issue, executed through the recompiled
// fragment stage on the real backend:
//
//   image_sample_c  [dref, u, v]    Yakuza Kiwami      (implicit-LOD depth compare)
//   image_sample_o  [offset, u, v]  Bendy and the Ink Machine (implicit-LOD packed texel offset)
//   image_get_lod   [u, v]          House of the Dead 2 (NSA form of an already-lowered op)
//
// _c and _o are admitted only on a ONE-level texture, where the implicit LOD cannot change the
// answer; the controls pin that a mipmapped texture (and, for _c, unequal min/mag filters or
// anisotropy) stays refused. Encodings are llvm-mc gfx1030 round trips.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "fixtures/render_runner.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t W = 128, H = 128;

// Fullscreen triangle + PARAM0 = (u, v, 0, 1) -- test_shadow_compare_render's VS.
const uint32_t kVs[] = {
    0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x100602f6u, 0x100804f6u,
    0x060606f3u, 0x060808f3u, 0x100a02f4u, 0x100c04f4u, 0x7e0e0280u, 0x7e1002f2u,
    0xf80008cfu, 0x08070403u, 0xf800020fu, 0x08070605u, 0xbf810000u,
};

// v2 = interpolated u, v3 = interpolated v, then `v1 = <operand>` and one NSA MIMG instruction
// writing v4, exported as MRT0 v4..v7.
std::vector<uint32_t> pixel_shader(uint32_t operand_mov, uint32_t word0, uint32_t word1,
                                   uint32_t word2) {
    return {0xc8080000u, 0xc8090001u, 0xc80c0100u, 0xc80d0101u,   // v2 = u, v3 = v
            operand_mov, word0,       word1,       word2,
            0xf800080fu, 0x07060504u,   // exp mrt0 v4..v7
            0xbf810000u};
}

constexpr uint32_t kMovV1Half = 0x7e0202f0u;   // v_mov_b32 v1, 0.5
constexpr uint32_t kMovV1Two = 0x7e020282u;   // v_mov_b32 v1, 2
constexpr uint32_t kMovV1Zero = 0x7e020280u;   // v_mov_b32 v1, 0
// image_sample_c v4, [v1, v2, v3], s[8:15], s[16:19] dmask:0x1 dim:2D
constexpr uint32_t kSampleC[3] = {0xf0a0010au, 0x00820401u, 0x00000302u};
// image_sample_o v4, [v1, v2, v3], s[8:15], s[16:19] dmask:0x1 dim:2D
constexpr uint32_t kSampleO[3] = {0xf0c0010au, 0x00820401u, 0x00000302u};

ShaderResourceTable texture_table(bool depth_compare, uint32_t levels) {
    ShaderResourceTable rt;
    ShaderResource t{};
    t.cls = ResourceClass::Texture;
    t.binding = 4;
    t.img_dim = 1;
    t.width = 8;
    t.height = 8;
    t.sgpr_base = 8;
    t.declared_mip_levels = levels;
    t.depth_compare = depth_compare;
    t.depth_compare_func = 4;   // GREATER: pass = dref > stored
    t.mag_filter = t.min_filter = 0;
    rt.resources.push_back(t);
    return rt;
}

// 8x8: columns 0..3 hold R=0, columns 4..7 R=230 (as a depth, about 0.9).
std::vector<uint8_t> split_texture() {
    std::vector<uint8_t> tex(8 * 8 * 4);
    for (uint32_t y = 0; y < 8; y++)
        for (uint32_t x = 0; x < 8; x++) {
            uint8_t* t = &tex[(y * 8 + x) * 4];
            t[0] = x < 4 ? 0 : 230;
            t[1] = 0;
            t[2] = 0;
            t[3] = 255;
        }
    return tex;
}

// Renders `ps` and returns MRT0's red channel along the middle row, or an empty vector.
std::vector<uint8_t> render_row(const std::vector<uint32_t>& ps, const ShaderResourceTable& rt) {
    const std::vector<uint32_t> vert = recompile_vertex(kVs, std::size(kVs));
    const std::vector<uint32_t> frag = recompile_fragment(ps.data(), ps.size(), &rt);
    if (vert.empty() || frag.empty()) return {};
    static const std::vector<uint8_t> tex = split_texture();
    prosper::test::FrameResource resource;
    resource.binding = 4;
    resource.set = 1;
    resource.tex_rgba = tex.data();
    resource.tw = 8;
    resource.th = 8;
    resource.mag_filter = resource.min_filter = 0;
    std::vector<prosper::test::FrameResource> resources{resource};
    const std::vector<uint8_t> px = prosper::test::render_triangle_rgba(
        vert, frag, W, H, nullptr, nullptr, nullptr, nullptr, &resources);
    if (px.size() != size_t{W} * H * 4) return {};
    std::vector<uint8_t> row(W);
    for (uint32_t x = 0; x < W; ++x) row[x] = px[((size_t{H} / 2) * W + x) * 4];
    return row;
}

bool span_is(const std::vector<uint8_t>& row, uint32_t from, uint32_t to, bool high) {
    for (uint32_t x = from; x < to; ++x)
        if (high ? row[x] < 200 : row[x] > 50) return false;
    return true;
}

}   // namespace

// GREATER with dref 0.5: the left half (stored 0.0) passes to 1.0, the right half (about 0.9) fails.
TEST(MimgNsaFormsRender, SampleCComparesEachSideOfTheSplit) {
    const auto ps = pixel_shader(kMovV1Half, kSampleC[0], kSampleC[1], kSampleC[2]);
    const auto row = render_row(ps, texture_table(true, 1));
    ASSERT_EQ(row.size(), W) << "image_sample_c NSA recompiles and renders (refused before #4808)";
    EXPECT_TRUE(span_is(row, 8, W / 2 - 8, true)) << "dref 0.5 > stored 0.0 passes";
    EXPECT_TRUE(span_is(row, W / 2 + 8, W - 8, false)) << "dref 0.5 > stored 0.9 fails";
}

TEST(MimgNsaFormsRender, SampleCIsRefusedWhereTheImplicitLodCouldMatter) {
    const auto ps = pixel_shader(kMovV1Half, kSampleC[0], kSampleC[1], kSampleC[2]);
    const ShaderResourceTable one = texture_table(true, 1), chain = texture_table(true, 2),
                              colour = texture_table(false, 1);
    EXPECT_FALSE(recompile_fragment(ps.data(), ps.size(), &one).empty());
    EXPECT_TRUE(recompile_fragment(ps.data(), ps.size(), &chain).empty())
        << "a mip chain: the implicit LOD selects a level";
    ShaderResourceTable filters = texture_table(true, 1);
    filters.resources[0].min_filter = 1;
    EXPECT_TRUE(recompile_fragment(ps.data(), ps.size(), &filters).empty())
        << "min != mag: the LOD sign picks the filter";
    ShaderResourceTable aniso = texture_table(true, 1);
    aniso.resources[0].max_aniso_ratio = 2;
    EXPECT_TRUE(recompile_fragment(ps.data(), ps.size(), &aniso).empty())
        << "anisotropy takes derivatives the manual compare does not";
    EXPECT_TRUE(recompile_fragment(ps.data(), ps.size(), &colour).empty())
        << "a non-compare S# is not silently read as colour";
}

// A +2 texel offset in x moves the split two texels (a quarter of the target) left: the band
// u in [0.25, 0.5) reads columns 4..5 and turns high. Offset 0 is the control.
TEST(MimgNsaFormsRender, SampleOShiftsTheSplitByItsTexelOffset) {
    const auto shifted = render_row(pixel_shader(kMovV1Two, kSampleO[0], kSampleO[1], kSampleO[2]),
                                    texture_table(false, 1));
    ASSERT_EQ(shifted.size(), W)
        << "image_sample_o NSA recompiles and renders (refused before #4808)";
    EXPECT_TRUE(span_is(shifted, 4, W / 4 - 4, false))
        << "u < 0.25 + 2 texels still reads columns 2-3";
    EXPECT_TRUE(span_is(shifted, W / 4 + 4, W - 4, true)) << "u >= 0.25 now reads columns 4+";
    const auto plain = render_row(pixel_shader(kMovV1Zero, kSampleO[0], kSampleO[1], kSampleO[2]),
                                  texture_table(false, 1));
    ASSERT_EQ(plain.size(), W);
    EXPECT_TRUE(span_is(plain, W / 4 + 4, W / 2 - 4, false)) << "control: offset 0 keeps the split";
    EXPECT_TRUE(span_is(plain, W / 2 + 4, W - 4, true));
    const auto ps = pixel_shader(kMovV1Two, kSampleO[0], kSampleO[1], kSampleO[2]);
    const ShaderResourceTable chain = texture_table(false, 2);
    EXPECT_TRUE(recompile_fragment(ps.data(), ps.size(), &chain).empty())
        << "a mip chain: an offset folded at level 0's size is wrong at level 1";
}

// The NSA form [v2, v3] reads exactly what the contiguous v[2:3] form reads, so the two modules are
// identical; [v2, v5] reads another register and is not; a set byte past the second address is an
// operand this form does not have and is refused.
TEST(MimgNsaFormsRender, GetLodNsaReadsItsSecondAddressByte) {
    const ShaderResourceTable rt = texture_table(false, 1);
    const auto compile = [&](std::vector<uint32_t> mimg) {
        std::vector<uint32_t> ps = {0xc8080000u, 0xc8090001u, 0xc80c0100u, 0xc80d0101u,
                                    0x7e0a02f0u};   // v_mov_b32 v5, 0.5
        ps.insert(ps.end(), mimg.begin(), mimg.end());
        ps.insert(ps.end(), {0xf800080fu, 0x07060504u, 0xbf810000u});
        return recompile_fragment(ps.data(), ps.size(), &rt);
    };
    const auto contiguous = compile({0xf1800108u, 0x00820402u});
    const auto nsa = compile({0xf180010au, 0x00820402u, 0x00000003u});
    const auto other = compile({0xf180010au, 0x00820402u, 0x00000005u});
    ASSERT_FALSE(contiguous.empty());
    ASSERT_FALSE(nsa.empty()) << "image_get_lod NSA recompiles (refused before #4808)";
    EXPECT_EQ(nsa, contiguous);
    ASSERT_FALSE(other.empty());
    EXPECT_NE(other, contiguous) << "[v2, v5] must read v5";
    EXPECT_TRUE(compile({0xf180010au, 0x00820402u, 0x00000503u}).empty())
        << "a third address byte is refused";
}
