// test_fragment_scalar_pair_mask — a Wave64 pixel shader's B64 mask logical with an ordinary scalar
// DATA pair as one source projects that pair onto this pixel's own lane bit (#4706).
//
// Kena's post-New-Game pixel program 0x5008cc0000 does `s_and_b64 s[30:31], s[30:31], s[36:37]` at
// pc308, where s[30:31] came from `s_buffer_load_dwordx2` (data) and s[36:37] from a VOPC (a mask).
// Compute already projected a data pair through the lane id; the fragment stage refused, and every
// draw of the program was lost.
//
// WHAT EACH ARM KILLS:
//   DataPairCompiles        the refusal itself
//   EachHalfSelectsItsLanes a projection that ignores the lane (low dword's bit 0 for every pixel:
//                           all-or-nothing), or reads the wrong half (the two programs swap)
//
// The render arm needs a host that can REQUIRE a 64-lane fragment subgroup, and it counts pixels
// rather than placing them, so it holds for any assignment of pixels to lanes: a 64x64 fullscreen
// draw is 64 full waves, so exactly half of the pixels are lanes 0..31.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/render_runner.h"
#include <gtest/gtest.h>

#include <cstdint>
#include <iterator>
#include <vector>

using namespace prosper::gpu;

namespace {

// s_mov_b32 s0, LO | s_mov_b32 s1, HI | v_cmp_eq_u32_e64 s[2:3], 0, 0 (every lane) |
// s_and_b64 s[4:5], s[0:1], s[2:3] | v_cndmask_b32_e64 v1, 0, 1.0, s[4:5] |
// v0 = 0, v2 = 0, v3 = 1.0 | exp mrt0 v0..v3 | s_endpgm
#define PAIR_PROGRAM(lo, hi)                                                                     \
    {lo, hi, 0xd4c20002u, 0x00010080u, 0x87840200u, 0xd5010001u, 0x0011e480u, 0x7e000280u,      \
     0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u}
constexpr uint32_t kLowLanes[] = PAIR_PROGRAM(0xbe8003c1u /* s0 = -1 */, 0xbe810380u /* s1 = 0 */);
constexpr uint32_t kHighLanes[] = PAIR_PROGRAM(0xbe800380u /* s0 = 0 */, 0xbe8103c1u /* s1 = -1 */);
#undef PAIR_PROGRAM

// Fullscreen triangle from gl_VertexIndex (test_fragment_partial_wave_exec's vertex program).
constexpr uint32_t kFullscreenVs[] = {
    0x7e140d00u, 0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x100202f6u, 0x100404f6u,
    0x060202f3u, 0x060404f3u, 0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xf800020fu,
    0x0403030au, 0xbf810000u,
};

std::vector<uint8_t> render(const std::vector<uint32_t>& fs, uint32_t w, uint32_t h) {
    prosper::test::BackendDraw d;
    d.vs = recompile_vertex(kFullscreenVs, std::size(kFullscreenVs));
    d.fs = fs;
    d.vcount = 3;
    for (uint32_t set = 0; set < 2; ++set) {
        prosper::test::FrameResource cb; cb.binding = 2; cb.set = set; d.R.push_back(cb);
        prosper::test::FrameResource vb; vb.binding = 3; vb.set = set; d.R.push_back(vb);
    }
    return prosper::test::render_draws_rgba({d}, w, h);
}

struct Counts { uint32_t green = 0, black = 0, other = 0; };

Counts count(const std::vector<uint8_t>& px) {
    Counts c;
    for (size_t i = 0; i + 3 < px.size(); i += 4) {
        const uint8_t* p = &px[i];
        if (p[0] < 0x10 && p[1] > 0xf0 && p[2] < 0x10) ++c.green;
        else if (p[0] < 0x10 && p[1] < 0x10 && p[2] < 0x10) ++c.black;
        else ++c.other;
    }
    return c;
}

}  // namespace

TEST(FragmentScalarPairMask, DataPairCompiles) {
    const std::vector<uint32_t> fs = recompile_fragment(kLowLanes, std::size(kLowLanes));
    ASSERT_FALSE(fs.empty()) << "a data pair ANDed with a VOPC mask is a per-lane bit, not a refusal";
    EXPECT_EQ(fragment_spirv_required_subgroup_size(fs), 64u)
        << "the projection reads the lane id, so the module carries the Wave64 contract";
}

TEST(FragmentScalarPairMask, EachHalfSelectsItsLanes) {
    const std::vector<uint32_t> low = recompile_fragment(kLowLanes, std::size(kLowLanes));
    const std::vector<uint32_t> high = recompile_fragment(kHighLanes, std::size(kHighLanes));
    ASSERT_FALSE(low.empty());
    ASSERT_FALSE(high.empty());
    const auto& ctx = prosper::test::render_vk_ctx();
    const bool native64 = ctx.subgroup_size_control &&
        (ctx.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
        (ctx.subgroup_operations & VK_SUBGROUP_FEATURE_BALLOT_BIT) &&
        ctx.min_subgroup_size <= 64 && ctx.max_subgroup_size >= 64 &&
        (ctx.required_subgroup_size_stages & VK_SHADER_STAGE_FRAGMENT_BIT);
    if (!native64) GTEST_SKIP() << "the device cannot require a 64-lane fragment subgroup";
    constexpr uint32_t W = 64, H = 64;
    const Counts lo = count(render(low, W, H));
    const Counts hi = count(render(high, W, H));
    EXPECT_EQ(lo.other, 0u) << "every pixel is written green or black";
    EXPECT_EQ(hi.other, 0u) << "every pixel is written green or black";
    EXPECT_EQ(lo.green, W * H / 2) << "low dword set: lanes 0..31 of each full wave are green";
    EXPECT_EQ(lo.black, W * H / 2);
    EXPECT_EQ(hi.green, W * H / 2) << "high dword set: lanes 32..63 of each full wave are green";
    EXPECT_EQ(hi.black, W * H / 2);
}
