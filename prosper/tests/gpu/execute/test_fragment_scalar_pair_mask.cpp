// test_fragment_scalar_pair_mask — a Wave64 pixel shader's B64 mask logical with an ordinary scalar
// DATA pair as one source projects that pair onto this pixel's own lane bit, but only when the pair
// is definitely assigned and is not a memory pattern (#4706, RECOMPILER_REMAINING.md's #2790 row).
//
// Kena's post-New-Game pixel program 0x5008cc0000 spills a compare mask with v_writelane at
// pc252/262 (the writelane stores the exact ballot words), reloads it with v_readlane at pc295/297
// and does `s_and_b64 s[30:31], s[30:31], s[36:37]` at pc308 with a fresh compare mask. Compute
// already projected such a pair through the lane id; the fragment stage refused it, and every draw
// of the program was lost.
//
// WHAT EACH ARM KILLS:
//   DataPairCompiles                  the refusal itself
//   EachHalfSelectsItsLanes           a projection that ignores the lane (low dword's bit 0 for
//                                     every pixel: all-or-nothing), or reads the wrong half
//   SpilledMaskReloadSelectsItsLanes  Kena's route: a lane-dependent ballot spilled and reloaded
//                                     must come back on the lanes that set it
//   PairAssignedOnOnePathRefuses      projecting the structured merge's fabricated zero (the #2790
//                                     row's obstacle; the review of #4711 built this arm), directly,
//                                     through a copy, and as Sonic Frontiers' #2790 loop accumulator
//   MemoryPatternRefuses              projecting a pattern loaded from memory onto host lanes,
//                                     directly and through a spill slot; its control overwrites the
//                                     loaded words and must compile
//
// The render arms need a host that can REQUIRE a 64-lane fragment subgroup, and they count pixels
// rather than placing them, so they hold for any assignment of pixels to lanes: a 64x64 fullscreen
// draw is 64 full waves, so exactly half of the pixels are lanes 0..31.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "fixtures/render_runner.h"
#include <gtest/gtest.h>

#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

using namespace prosper::gpu;

namespace {

// s_mov_b32 s0, LO | s_mov_b32 s1, HI | v_cmp_eq_u32_e64 s[2:3], 0, 0 (every lane) |
// s_and_b64 s[4:5], s[0:1], s[2:3] | v_cndmask_b32_e64 v1, 0, 1.0, s[4:5] |
// v0 = 0, v2 = 0, v3 = 1.0 | exp mrt0 v0..v3 | s_endpgm
#define PAIR_PROGRAM(lo, hi)                                                                       \
    {lo,          hi,          0xd4c20002u, 0x00010080u, 0x87840200u, 0xd5010001u, 0x0011e480u,    \
     0x7e000280u, 0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u}
constexpr uint32_t kLowLanes[] = PAIR_PROGRAM(0xbe8003c1u /* s0 = -1 */, 0xbe810380u /* s1 = 0 */);
constexpr uint32_t kHighLanes[] = PAIR_PROGRAM(0xbe800380u /* s0 = 0 */, 0xbe8103c1u /* s1 = -1 */);
#undef PAIR_PROGRAM

// v5 = lane id (mbcnt lo/hi) | s[2:3] = (v5 < 32) | v12[1] = s2, v12[2] = s3 (spill) |
// s4 = v12[1], s5 = v12[2] (reload) | s[6:7] = every lane | s[8:9] = s[4:5] & s[6:7] |
// v1 = s[8:9] ? 1.0 : 0 | export (0, v1, 0, 1). Assembled with llvm-mc -mcpu=gfx1030
// -mattr=+wavefrontsize64.
constexpr uint32_t kSpilledMaskReload[] = {
    0xd7650005u, 0x000100c1u, 0xd7660005u, 0x00020ac1u, 0xd4c40002u, 0x00020aa0u, 0xd761000cu,
    0x00010202u, 0xd761000cu, 0x00010403u, 0xd7600004u, 0x0001030cu, 0xd7600005u, 0x0001050cu,
    0xd4c20006u, 0x00010080u, 0x87880604u, 0xd5010001u, 0x0021e480u, 0x7e000280u, 0x7e040280u,
    0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
};

// The review's arm: s[4:5] is written on one path of a scalar if whose bound the fold cannot know
// (v_readfirstlane), then ANDed with a compare mask after the merge, where the skipped edge's value
// is the structured emitter's fabricated zero.
constexpr uint32_t kPairAssignedOnOnePath[] = {
    0x7e0a0280u, 0x7e000505u, 0xd4c20002u, 0x00010080u, 0xbf068000u, 0xbf850002u,
    0xbe8403c1u, 0xbe8503c1u, 0x87860204u, 0xd5010001u, 0x0019e480u, 0x7e000280u,
    0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
};

// The same, but the merged pair is first copied (`s_mov_b64 s[10:11], s[4:5]`): the mark must follow
// the copy, or one move launders the fabricated zero.
constexpr uint32_t kPairAssignedOnOnePathCopied[] = {
    0x7e0a0280u, 0x7e000505u, 0xd4c20002u, 0x00010080u, 0xbf068000u, 0xbf850002u,
    0xbe8403c1u, 0xbe8503c1u, 0xbe8a0404u, 0x8786020au, 0xd5010001u, 0x0019e480u,
    0x7e000280u, 0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
};

// Sonic Frontiers' #2790 HDR-producer shape: the divergent-boolean accumulator
// `s_andn2_b64 s[68:69], s[84:85], EXEC` (pc20 here, pc328 there) inside nested EXEC loops, with
// s84 defined only through the back edge. This is test_fragment_loop_mask_exec's fixture with its
// pre-loop `s_mov_b32 s84, 0` replaced by s_nop, which is what #2790 traced in the real program:
// no definition of s84 on the loop-entry path. `s_mov_b32 s85, 0` is kept so both words are present
// at the read, as #2790's live probe showed (`sreg=1/1`); the only thing left to separate this from
// the defined fixture is that the low word's first-iteration value is the loop phi's fabricated
// zero, so the projection must refuse.
constexpr uint32_t kAccumulatorWithoutEntryDefinition[] = {
    0xbf800000u, 0xbed50380u, 0xbed8047eu, 0xd7650007u, 0x000100c1u, 0xd7660007u, 0x00020ec1u,
    0x36120e83u, 0x4a121281u, 0x36160e81u, 0x7e140280u, 0xbf880019u, 0x7e100280u, 0xd4c2005cu,
    0x0002150bu, 0xbefe045cu, 0xbed6047eu, 0xbf88000du, 0xd4c6005cu, 0x00010109u, 0x8ac47e54u,
    0xbeea045cu, 0x88d4445cu, 0xbefe045cu, 0xbf860006u, 0x4a101081u, 0x7d8c1308u, 0x8afe6a5cu,
    0x8aea5c54u, 0x88d46a5cu, 0xbf89fff2u, 0xbefe0458u, 0x4a141481u, 0xd4c1006au, 0x0001050au,
    0xbefe046au, 0xbf89ffe6u, 0xbefe0458u, 0x87ea5854u, 0xd5010000u, 0x01a9e480u, 0x7e0202f2u,
    0x7e040280u, 0x7e0602f2u, 0xf800080fu, 0x03020100u, 0xbf810000u,
};

// s_buffer_load_dwordx2 s[0:1], s[4:7], 0 | s_waitcnt | <two words> | s[2:3] = every lane |
// s[8:9] = s[0:1] & s[2:3] | v1 = s[8:9] ? 1.0 : 0 | export
#define MEMORY_PROGRAM(w0, w1)                                                                     \
    {0xf4240002u, 0xfa000000u, 0xbf8cc07fu, w0,          w1,          0xd4c20002u,                 \
     0x00010080u, 0x87880200u, 0xd5010001u, 0x0021e480u, 0x7e000280u, 0x7e040280u,                 \
     0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u}
constexpr uint32_t kMemoryPattern[] = MEMORY_PROGRAM(0xbf800000u, 0xbf800000u);   // s_nop, s_nop
constexpr uint32_t kMemoryOverwritten[] =
    MEMORY_PROGRAM(0xbe8003c1u /* s0 = -1 */, 0xbe810380u /* s1 = 0 */);
// The loaded pair spilled with v_writelane and reloaded with v_readlane before the AND: the memory
// mark must survive the spill slot.
constexpr uint32_t kMemoryPatternSpilled[] = {
    0xf4240002u, 0xfa000000u, 0xbf8cc07fu, 0xd761000cu, 0x00010200u, 0xd761000cu,
    0x00010401u, 0xd760000au, 0x0001030cu, 0xd760000bu, 0x0001050cu, 0xd4c20002u,
    0x00010080u, 0x8788020au, 0xd5010001u, 0x0021e480u, 0x7e000280u, 0x7e040280u,
    0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
};
#undef MEMORY_PROGRAM

ShaderResourceTable memory_table() {   // the V# at s[4:7]
    ShaderResourceTable table;
    ShaderResource cbuf{};
    cbuf.cls = ResourceClass::ConstantBuffer;
    cbuf.format = DataFormat::Uint32;
    cbuf.num_components = 1;
    cbuf.binding = 2;
    cbuf.sgpr_base = 4;
    cbuf.size = 64;
    table.resources.push_back(cbuf);
    return table;
}

// Fullscreen triangle from gl_VertexIndex (test_fragment_partial_wave_exec's vertex program).
constexpr uint32_t kFullscreenVs[] = {
    0x7e140d00u, 0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x100202f6u,
    0x100404f6u, 0x060202f3u, 0x060404f3u, 0x7e060280u, 0x7e0802f2u, 0xf80008cfu,
    0x04030201u, 0xf800020fu, 0x0403030au, 0xbf810000u,
};

std::vector<uint8_t> render(const std::vector<uint32_t>& fs, uint32_t w, uint32_t h) {
    prosper::test::BackendDraw d;
    d.vs = recompile_vertex(kFullscreenVs, std::size(kFullscreenVs));
    d.fs = fs;
    d.vcount = 3;
    for (uint32_t set = 0; set < 2; ++set) {
        prosper::test::FrameResource cb;
        cb.binding = 2;
        cb.set = set;
        d.R.push_back(cb);
        prosper::test::FrameResource vb;
        vb.binding = 3;
        vb.set = set;
        d.R.push_back(vb);
    }
    return prosper::test::render_draws_rgba({d}, w, h);
}

struct Counts {
    uint32_t green = 0, black = 0, other = 0;
};

Counts count(const std::vector<uint8_t>& px) {
    Counts c;
    for (size_t i = 0; i + 3 < px.size(); i += 4) {
        const uint8_t* p = &px[i];
        if (p[0] < 0x10 && p[1] > 0xf0 && p[2] < 0x10)
            ++c.green;
        else if (p[0] < 0x10 && p[1] < 0x10 && p[2] < 0x10)
            ++c.black;
        else
            ++c.other;
    }
    return c;
}

bool native_fragment_wave64() {
    const auto& ctx = prosper::test::render_vk_ctx();
    return ctx.subgroup_size_control && (ctx.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
           (ctx.subgroup_operations & VK_SUBGROUP_FEATURE_BALLOT_BIT) &&
           ctx.min_subgroup_size <= 64 && ctx.max_subgroup_size >= 64 &&
           (ctx.required_subgroup_size_stages & VK_SHADER_STAGE_FRAGMENT_BIT);
}

}   // namespace

TEST(FragmentScalarPairMask, DataPairCompiles) {
    const std::vector<uint32_t> fs = recompile_fragment(kLowLanes, std::size(kLowLanes));
    ASSERT_FALSE(fs.empty())
        << "a definite constant pair ANDed with a VOPC mask is a per-lane bit, not a refusal";
    EXPECT_EQ(fragment_spirv_required_subgroup_size(fs), 64u)
        << "the projection reads the lane id, so the module carries the Wave64 contract";
}

TEST(FragmentScalarPairMask, EachHalfSelectsItsLanes) {
    const std::vector<uint32_t> low = recompile_fragment(kLowLanes, std::size(kLowLanes));
    const std::vector<uint32_t> high = recompile_fragment(kHighLanes, std::size(kHighLanes));
    ASSERT_FALSE(low.empty());
    ASSERT_FALSE(high.empty());
    if (!native_fragment_wave64()) GTEST_SKIP() << "the device cannot require 64 fragment lanes";
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

TEST(FragmentScalarPairMask, SpilledMaskReloadSelectsItsLanes) {
    const std::vector<uint32_t> fs =
        recompile_fragment(kSpilledMaskReload, std::size(kSpilledMaskReload));
    ASSERT_FALSE(fs.empty()) << "a reloaded ballot ANDed with a compare mask is a per-lane bit";
    if (!native_fragment_wave64()) GTEST_SKIP() << "the device cannot require 64 fragment lanes";
    constexpr uint32_t W = 64, H = 64;
    const Counts c = count(render(fs, W, H));
    EXPECT_EQ(c.other, 0u) << "every pixel is written green or black";
    EXPECT_EQ(c.green, W * H / 2)
        << "the lanes below 32 that set the ballot are the lanes selected";
    EXPECT_EQ(c.black, W * H / 2);
}

TEST(FragmentScalarPairMask, PairAssignedOnOnePathRefuses) {
    EXPECT_TRUE(
        recompile_fragment(kPairAssignedOnOnePath, std::size(kPairAssignedOnOnePath)).empty())
        << "the skipped edge's fabricated zero must refuse, not project";
    EXPECT_TRUE(
        recompile_fragment(kPairAssignedOnOnePathCopied, std::size(kPairAssignedOnOnePathCopied))
            .empty())
        << "a copy of the merged pair carries its mark";
    // The program is refused at a later pc too, so name WHERE: the accumulator's own read.
    constexpr uint64_t kAddress = 0xa4706002ull;
    EXPECT_TRUE(recompile_fragment(kAccumulatorWithoutEntryDefinition,
                                   std::size(kAccumulatorWithoutEntryDefinition), nullptr, nullptr,
                                   UINT32_MAX, nullptr, false,
                                   {RecompileDiagnosticStage::Fragment, kAddress})
                    .empty())
        << "#2790's accumulator with no loop-entry definition: the phi's zero must not project";
    const std::string reason = last_terminal_reject_reason(kAddress);
    EXPECT_NE(reason.find("pc=20 words=8ac47e54"), std::string::npos)
        << "refused at the s_andn2_b64 accumulator read itself: " << reason;
}

TEST(FragmentScalarPairMask, MemoryPatternRefuses) {
    const ShaderResourceTable table = memory_table();
    EXPECT_FALSE(
        recompile_fragment(kMemoryOverwritten, std::size(kMemoryOverwritten), &table).empty())
        << "control: the same program with the loaded words overwritten compiles";
    EXPECT_TRUE(recompile_fragment(kMemoryPattern, std::size(kMemoryPattern), &table).empty())
        << "a memory-loaded lane pattern is not a ballot of this wave; host lanes are not PS5's";
    EXPECT_TRUE(
        recompile_fragment(kMemoryPatternSpilled, std::size(kMemoryPatternSpilled), &table).empty())
        << "spilling the loaded words through v_writelane/v_readlane keeps the memory mark";
}
