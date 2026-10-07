// test_fragment_loop_vcc_scratch — a fragment loop whose body recycles VCC as scalar scratch (#4508).
//
// Space Adventure Cobra's per-pixel light loop, reduced. Three of its fragment programs carry it
// at pc 102..173 and all three were rejected, which dropped most of the in-game 3D:
//
//     102  v_cmp_lt_i32 vcc, s30, v4        header: recompute the predicate
//     103  s_cbranch_vccz 174
//     104  s_lshl_b32 vcc_lo, s30, 2        body: VCC is the nearest free scratch pair
//     105  s_and_b32  vcc_hi, s30, 3
//     ...
//     173  s_branch 102
//
// The loop needs TWO things to reach the defect, and the older VCCZ-loop fixtures in
// test_recompiled_fragment.cpp have neither: a VCC mask live BEFORE the loop (otherwise no header
// phi is created for it) and a body that leaves VCC without a mask.
//
// Encodings are assembled by hand from forms already used by the fixtures in this directory and
// by the live shader; DecodesAsDescribed checks the three that matter through the project's own
// decoder, so a mis-assembled word fails there instead of quietly changing what the arms test.
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include "fixtures/render_runner.h"
#include "fixtures/spirv_triangle.h"
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

using namespace prosper::gpu;

namespace {

//  for (s0 = 0; s0 < 4; ++s0) v0 += 0.125, with the increment routed THROUGH vcc_lo. A loop that
//  mishandles the scratch therefore runs the wrong number of iterations, or never ends. The step
//  is small enough that five trips do not saturate the target: the count is checked both ways.
constexpr uint32_t kScratchLoop[] = {
    0xBE800380u,   //  0  s_mov_b32 s0, 0
    0x7E000280u,   //  1  v_mov_b32 v0, 0
    0x7E020284u,   //  2  v_mov_b32 v1, 4
    0x7E0602F2u,   //  3  v_mov_b32 v3, 1.0
    0x7D020200u,   //  4  v_cmp_lt_i32 vcc, s0, v1    VCC is a live mask before the loop
    0x7D020200u,   //  5  HEADER: v_cmp_lt_i32 vcc, s0, v1
    0xBF860006u,   //  6  s_cbranch_vccz 13
    0x816A8100u,   //  7  s_add_i32 vcc_lo, s0, 1     scratch: the next counter value
    0x876B8300u,   //  8  s_and_b32 vcc_hi, s0, 3     scratch
    0x060000FFu, 0x3E000000u,   //  9  v_add_f32 v0, 0.125, v0
    0xBE80036Au,   // 11  s_mov_b32 s0, vcc_lo        the counter comes back out of VCC
    0xBF82FFF8u,   // 12  s_branch 5
    0x7E020300u,   // 13  v_mov_b32 v1, v0
    0x7E040300u,   // 14  v_mov_b32 v2, v0
    0xF800080Fu, 0x03020100u,   // 15  exp mrt0 v0, v1, v2, v3
    0xBF810000u,   // 17  s_endpgm
};

//  The same loop, but the header reads the CARRIED pair as scalar data before redefining it. On
//  the back-edge that is the body's scratch, which is representable; on the ENTRY path it is the
//  compare's mask, which a fragment invocation cannot hand out as a dword. Admitting this loop would
//  feed the read a zero placeholder on the first iteration.
constexpr uint32_t kHeaderReadsCarriedVcc[] = {
    0xBE800380u, 0x7E000280u, 0x7E020284u, 0x7E0602F2u,
    0x7D020200u,   //  4  v_cmp_lt_i32 vcc, s0, v1
    0xBE82036Au,   //  5  HEADER: s_mov_b32 s2, vcc_lo        <<< reads the carried pair
    0x7D020200u,   //  6  v_cmp_lt_i32 vcc, s0, v1
    0xBF860006u,   //  7  s_cbranch_vccz 14
    0x816A8100u,   //  8  s_add_i32 vcc_lo, s0, 1
    0x876B8300u,   //  9  s_and_b32 vcc_hi, s0, 3
    0x060000FFu, 0x3E800000u,   // 10  v_add_f32 v0, 0.25, v0
    0xBE80036Au,   // 12  s_mov_b32 s0, vcc_lo
    0xBF82FFF7u,   // 13  s_branch 5
    0x7E020300u, 0x7E040300u, 0xF800080Fu, 0x03020100u, 0xBF810000u,
};

//  kScratchLoop, then a DATA read of vcc_lo after the loop. On the exit path vcc_lo holds the
//  header compare's mask, not the scratch the previous iteration left there.
constexpr uint32_t kDataReadAfterLoop[] = {
    0xBE800380u, 0x7E000280u, 0x7E020284u, 0x7E0602F2u, 0x7D020200u,
    0x7D020200u,   //  5  HEADER
    0xBF860006u,   //  6  s_cbranch_vccz 13
    0x816A8100u, 0x876B8300u, 0x060000FFu, 0x3E800000u, 0xBE80036Au,
    0xBF82FFF8u,   // 12  s_branch 5
    0xBE81036Au,   // 13  s_mov_b32 s1, vcc_lo        <<< the exit mask, as data
    0x7E020201u,   // 14  v_mov_b32 v1, s1
    0x7E040300u, 0xF800080Fu, 0x03020100u, 0xBF810000u,
};

//  #4526: the same post-loop data read after two loops that compiled before #4508's fix, so they
//  never took its placeholder path. Their exit state has the same stale scratch in vcc_lo.
//  (a) No VCC mask is live before the loop, so the loop has no VCC header phi at all.
constexpr uint32_t kDataReadNoMaskBefore[] = {
    0xBE800380u, 0x7E000280u, 0x7E020284u, 0x7E0602F2u,
    0x7D020200u,   //  4  HEADER: v_cmp_lt_i32 vcc, s0, v1
    0xBF860006u,   //  5  s_cbranch_vccz 12
    0x816A8100u, 0x876B8300u, 0x060000FFu, 0x3E000000u, 0xBE80036Au,
    0xBF82FFF8u,   // 11  s_branch 4
    0xBE81036Au,   // 12  s_mov_b32 s1, vcc_lo        <<< the exit mask, as data
    0x7E020201u,   // 13  v_mov_b32 v1, s1
    0x7E040300u, 0xF800080Fu, 0x03020100u, 0xBF810000u,
};
//  (b) The body ends with a compare, so the back-edge carries a real mask and needs no placeholder.
constexpr uint32_t kDataReadAfterMaskBackEdge[] = {
    0xBE800380u, 0x7E000280u, 0x7E020284u, 0x7E0602F2u,
    0x7D020200u,   //  4  v_cmp_lt_i32 vcc, s0, v1
    0x7D020200u,   //  5  HEADER: v_cmp_lt_i32 vcc, s0, v1
    0xBF860007u,   //  6  s_cbranch_vccz 14
    0x816A8100u,   //  7  s_add_i32 vcc_lo, s0, 1
    0x876B8300u,   //  8  s_and_b32 vcc_hi, s0, 3
    0x060000FFu, 0x3E000000u,   //  9  v_add_f32 v0, 0.125, v0
    0xBE80036Au,   // 11  s_mov_b32 s0, vcc_lo
    0x7D020200u,   // 12  v_cmp_lt_i32 vcc, s0, v1    VCC is a mask again before the back-edge
    0xBF82FFF7u,   // 13  s_branch 5
    0xBE81036Au,   // 14  s_mov_b32 s1, vcc_lo        <<< the exit mask, as data
    0x7E020201u, 0x7E040300u, 0xF800080Fu, 0x03020100u, 0xBF810000u,
};

//  A zero-trip twin for the exit state. The mask before the loop is SET (0 < 4) and the header's
//  is CLEAR (0 < 0), so the loop leaves at once and green is selected by whichever mask the merge
//  hands on: the header's (green) or one that leaked from the entry edge of the phi (black). Both
//  compares are the opcode whose meaning ScratchLoopRunsExactlyFourIterations already pins.
constexpr uint32_t kMaskReadAfterLoop[] = {
    0xBE800380u,   //  0  s_mov_b32 s0, 0
    0x7E000280u,   //  1  v_mov_b32 v0, 0
    0x7E020280u,   //  2  v_mov_b32 v1, 0             bound 0: zero trips
    0x7E0602F2u,   //  3  v_mov_b32 v3, 1.0
    0x7E0A02F2u,   //  4  v_mov_b32 v5, 1.0
    0x7E0C0284u,   //  5  v_mov_b32 v6, 4
    0x7D020C00u,   //  6  v_cmp_lt_i32 vcc, s0, v6    live mask before the loop: SET
    0x7D020200u,   //  7  HEADER: v_cmp_lt_i32 vcc, s0, v1   CLEAR
    0xBF860006u,   //  8  s_cbranch_vccz 15
    0x816A8100u,   //  9  s_add_i32 vcc_lo, s0, 1
    0x876B8300u,   // 10  s_and_b32 vcc_hi, s0, 3
    0x060000FFu, 0x3E800000u,   // 11  v_add_f32 v0, 0.25, v0
    0xBE80036Au,   // 13  s_mov_b32 s0, vcc_lo
    0xBF82FFF8u,   // 14  s_branch 7
    0x02020105u,   // 15  v_cndmask_b32 v1, v5, v0    <<< v1 = exit mask ? v0 (0) : v5 (1.0)
    0x7E040300u,   // 16  v_mov_b32 v2, v0
    0xF800080Fu, 0x03020100u,   // 17  exp mrt0 v0, v1, v2, v3
    0xBF810000u,   // 19  s_endpgm
};

//  kScratchLoop with an interior EXECZ that leaves the loop from INSIDE the body, after the scratch
//  writes. That path reaches the merge with VCC holding scalar data, while the check path reaches
//  it with a mask: the join has no Bool.
constexpr uint32_t kBreakFromScratchBody[] = {
    0xBE800380u, 0x7E000280u, 0x7E020284u, 0x7E0602F2u,
    0x7D020200u,   //  4  v_cmp_lt_i32 vcc, s0, v1
    0x7D020200u,   //  5  HEADER: v_cmp_lt_i32 vcc, s0, v1
    0xBF860007u,   //  6  s_cbranch_vccz 14
    0x816A8100u,   //  7  s_add_i32 vcc_lo, s0, 1
    0x876B8300u,   //  8  s_and_b32 vcc_hi, s0, 3
    0xBF880004u,   //  9  s_cbranch_execz 14        <<< direct break, VCC is scratch here
    0x060000FFu, 0x3E000000u,   // 10  v_add_f32 v0, 0.125, v0
    0xBE80036Au,   // 12  s_mov_b32 s0, vcc_lo
    0xBF82FFF7u,   // 13  s_branch 5
    0x7E020300u, 0x7E040300u, 0xF800080Fu, 0x03020100u, 0xBF810000u,
};

//  #4680: a COUNTED loop (SCC exit, unconditional back-edge) whose induction variable lives in
//  vcc_hi. Kena: Bridge of Spirits' wave64 light-list pixel shader, reduced to its scalar skeleton
//  with the live body's exact scalar words (pc 60..64 there): the header is
//  `s_cmp_lt_i32 vcc_hi, <bound>`, and the body turns the counter into a constant-buffer offset in
//  vcc_lo and steps vcc_hi by two. VCC is a mask before the loop (the emitter then gives VCC a
//  lane-mask header phi), and the body leaves the pair as scalar data, so that phi had no back-edge
//  value and the shader was refused. The bound comes from a VGPR so the scalar fold cannot know it.
constexpr uint32_t kCountedVccHiLoop[] = {
    0xBE800380u,   //  0  s_mov_b32 s0, 0
    0x7E000280u,   //  1  v_mov_b32 v0, 0
    0x7E020288u,   //  2  v_mov_b32 v1, 8
    0x7E0602F2u,   //  3  v_mov_b32 v3, 1.0
    0x7D020200u,   //  4  v_cmp_lt_i32 vcc, s0, v1    VCC is a live mask before the loop
    0xBEEB0380u,   //  5  s_mov_b32 vcc_hi, 0         the induction variable: scalar data
    0x7E040501u,   //  6  v_readfirstlane_b32 s2, v1  the bound (8), unknown to the scalar fold
    0xBF04026Bu,   //  7  HEADER: s_cmp_lt_i32 vcc_hi, s2   a scalar read of the carried half
    0xBF840009u,   //  8  s_cbranch_scc0 18
    0x916A9F6Bu,   //  9  s_ashr_i32 vcc_lo, vcc_hi, 31     Kena pc 60..64, word for word
    0x876A816Au,   // 10  s_and_b32 vcc_lo, vcc_lo, 1
    0x816A6A6Bu,   // 11  s_add_i32 vcc_lo, vcc_hi, vcc_lo
    0x916A816Au,   // 12  s_ashr_i32 vcc_lo, vcc_lo, 1
    0x816B826Bu,   // 13  s_add_i32 vcc_hi, vcc_hi, 2
    0x060000FFu, 0x3E000000u,   // 14  v_add_f32 v0, 0.125, v0
    0xBE83036Au,   // 16  s_mov_b32 s3, vcc_lo        the offset is consumed in the body
    0xBF82FFF5u,   // 17  s_branch 7
    0x7E020300u,   // 18  v_mov_b32 v1, v0
    0x7E040300u,   // 19  v_mov_b32 v2, v0
    0xF800080Fu, 0x03020100u,   // 20  exp mrt0 v0, v1, v2, v3
    0xBF810000u,   // 22  s_endpgm
};

struct Compiled {
    std::vector<uint32_t> spirv;
    std::string reason;
};

template <size_t N>
Compiled compile(const uint32_t (&words)[N], uint64_t address) {
    Compiled out;
    out.spirv = recompile_fragment(words, N, nullptr, nullptr, UINT32_MAX, nullptr,
                                   /*wave32=*/false, {RecompileDiagnosticStage::Fragment, address});
    out.reason = last_terminal_reject_reason(address);
    return out;
}

// Whether this device can run `fragment` as compiled. A module that votes over the guest wave
// declares an exact 64-lane subgroup, which lavapipe (fixed at 8) cannot supply; the backend then
// leaves the target untouched, so a pixel assertion would be testing the clear colour.
bool device_can_execute(const std::vector<uint32_t>& fragment) {
    if (fragment_spirv_required_subgroup_size(fragment) != 64) return true;
    const auto& ctx = prosper::test::render_vk_ctx();
    return ctx.subgroup_size_control && ctx.min_subgroup_size <= 64 &&
           ctx.max_subgroup_size >= 64 &&
           (ctx.required_subgroup_size_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
           (ctx.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
           (ctx.subgroup_operations & VK_SUBGROUP_FEATURE_VOTE_BIT) &&
           (ctx.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);
}

// The centre pixel of a 64x64 triangle drawn with `fragment`, or an empty vector if it did not render.
std::vector<uint8_t> centre_pixel(const std::vector<uint32_t>& fragment) {
    constexpr uint32_t kSide = 64;
    const std::vector<uint32_t> vertex(std::begin(kTriVertSpv), std::end(kTriVertSpv));
    const std::vector<uint8_t> pixels =
        prosper::test::render_triangle_rgba(vertex, fragment, kSide, kSide);
    if (pixels.size() != size_t{kSide} * kSide * 4) return {};
    const size_t centre = (size_t{kSide} / 2 * kSide + kSide / 2) * 4;
    return {pixels.begin() + centre, pixels.begin() + centre + 4};
}

}   // namespace

TEST(FragmentLoopVccScratch, DecodesAsDescribed) {
    std::vector<Rdna2Inst> ins;
    ASSERT_EQ(rdna2_walk(kScratchLoop, std::size(kScratchLoop), ins), std::size(kScratchLoop));
    const auto at = [&](uint32_t pc) -> const Rdna2Inst& {
        for (const Rdna2Inst& in : ins)
            if (in.pc == pc) return in;
        ADD_FAILURE() << "no instruction at pc " << pc;
        return ins.front();
    };
    // The predicate before the loop and at the header: a compare into the VCC pair.
    for (uint32_t pc : {4u, 5u}) {
        EXPECT_EQ(at(pc).fmt, Rdna2Format::VOPC) << "pc " << pc;
        EXPECT_EQ(at(pc).dst.value, 106) << "pc " << pc;
    }
    // The body writes BOTH halves with scalar ops...
    EXPECT_EQ(at(7).fmt, Rdna2Format::SOP2);
    EXPECT_EQ(at(7).dst.kind, OperandKind::SGPR);
    EXPECT_EQ(at(7).dst.value, 106);
    EXPECT_EQ(at(8).fmt, Rdna2Format::SOP2);
    EXPECT_EQ(at(8).dst.value, 107);
    // ...and reads vcc_lo back as data, so the scratch is live inside the body.
    EXPECT_EQ(at(11).fmt, Rdna2Format::SOP1);
    EXPECT_EQ(at(11).dst.value, 0);
    EXPECT_EQ(at(11).src[0].value, 106);
    // Back-edge to the header, exit just past it.
    EXPECT_EQ(static_cast<int64_t>(at(12).pc) + at(12).len_dwords + at(12).simm16, 5);
    EXPECT_EQ(static_cast<int64_t>(at(6).pc) + at(6).len_dwords + at(6).simm16, 13);
}

TEST(FragmentLoopVccScratch, BodyScratchDoesNotRejectTheLoop) {
    const Compiled loop = compile(kScratchLoop, 0x4508A001ull);
    ASSERT_FALSE(loop.spirv.empty())
        << "a loop that redefines VCC at its header must survive a body that uses VCC as "
           "scratch; reason: "
        << loop.reason;
    EXPECT_EQ(loop.spirv[0], 0x07230203u);
}

TEST(FragmentLoopVccScratch, ScratchLoopRunsExactlyFourIterations) {
    const Compiled loop = compile(kScratchLoop, 0x4508A002ull);
    ASSERT_FALSE(loop.spirv.empty()) << loop.reason;
    if (!device_can_execute(loop.spirv))
        GTEST_SKIP() << "device cannot execute the fragment wave64 contract this module declares";
    const std::vector<uint8_t> pixel = centre_pixel(loop.spirv);
    ASSERT_EQ(pixel.size(), 4u) << "the triangle did not render";
    // Four trips of +0.125 are 0.5, i.e. 127 or 128 of 255. One trip more or less moves the
    // channel by 32, so this band holds exactly one trip count -- a counter that came back out of
    // vcc_lo stale, early or late, lands outside it.
    for (int channel = 0; channel < 3; ++channel) {
        EXPECT_GT(pixel[channel], 0x70) << "channel " << channel;
        EXPECT_LT(pixel[channel], 0x90) << "channel " << channel;
    }

    // Control: the same loop bounded at five must leave the band, or the band proves nothing.
    uint32_t five_trips[std::size(kScratchLoop)];
    std::copy(std::begin(kScratchLoop), std::end(kScratchLoop), five_trips);
    five_trips[2] = 0x7E020285u;   // v_mov_b32 v1, 5
    const Compiled longer = compile(five_trips, 0x4508A007ull);
    ASSERT_FALSE(longer.spirv.empty()) << longer.reason;
    const std::vector<uint8_t> longer_pixel = centre_pixel(longer.spirv);
    ASSERT_EQ(longer_pixel.size(), 4u);
    EXPECT_GT(longer_pixel[0], 0x98) << "five trips of +0.125 are 0.625";
    EXPECT_LT(longer_pixel[0], 0xA8);
}

TEST(FragmentLoopVccScratch, WaveVotedBoundCompilesToo) {
    // The live loop's bound is a VGPR the proof cannot call wave-uniform, so its exit is a vote
    // over the guest wave. Same loop, bound taken from an unresolved lane-varying register.
    uint32_t varying[std::size(kScratchLoop)];
    std::copy(std::begin(kScratchLoop), std::end(kScratchLoop), varying);
    varying[2] = 0x7E020302u;   // v_mov_b32 v1, v2
    const Compiled loop = compile(varying, 0x4508A006ull);
    ASSERT_FALSE(loop.spirv.empty()) << loop.reason;
    EXPECT_EQ(fragment_spirv_required_subgroup_size(loop.spirv), 64u)
        << "a lane-varying VCC exit is decided by the complete 64-lane wave";
}

TEST(FragmentLoopVccScratch, HeaderReadOfTheCarriedPairStillRejects) {
    const Compiled loop = compile(kHeaderReadsCarriedVcc, 0x4508A003ull);
    EXPECT_TRUE(loop.spirv.empty())
        << "the header reads the pair carried round the loop; on entry that is a mask";
    // WHERE and WHY: an empty result alone is satisfied by any malformed fixture.
    EXPECT_NE(loop.reason.find("loop-carried VCC is scalar data at the back-edge"),
              std::string::npos)
        << loop.reason;
    EXPECT_NE(loop.reason.find("s106 is read from header pc=5"), std::string::npos) << loop.reason;
    EXPECT_NE(loop.reason.find("blocker pc=5 kind=source-dword"), std::string::npos) << loop.reason;
}

TEST(FragmentLoopVccScratch, BreakOutOfAScratchBodyStillRejects) {
    const Compiled loop = compile(kBreakFromScratchBody, 0x4508A008ull);
    EXPECT_TRUE(loop.spirv.empty()) << "the merge would join a mask with scalar data";
    EXPECT_NE(loop.reason.find("direct break reaches the merge (header pc=5)"), std::string::npos)
        << loop.reason;
}

TEST(FragmentLoopVccScratch, DataReadAfterTheLoopIsNotServedFromScratch) {
    const Compiled loop = compile(kDataReadAfterLoop, 0x4508A004ull);
    // The exit path's vcc_lo is the header compare's mask. A fragment invocation has no exact
    // dword for it, so the read rejects -- the same answer as after a compare in straight-line
    // code. What must NOT happen is a module that compiles by reading the loop's scratch.
    EXPECT_TRUE(loop.spirv.empty())
        << "compiled a data read of the exit mask; it can only have come from loop scratch";
    EXPECT_NE(loop.reason.find("pc=13"), std::string::npos) << loop.reason;
}

TEST(FragmentLoopVccScratch, DataReadAfterLoopsThatNeededNoPlaceholderIsNotServedFromScratch) {
    // Both loops compiled before the placeholder existed, and both then compiled this read from
    // the scratch the loop left in vcc_lo. Each must name the read as the reject.
    const Compiled no_mask = compile(kDataReadNoMaskBefore, 0x4526A001ull);
    EXPECT_TRUE(no_mask.spirv.empty()) << "no VCC mask before the loop";
    EXPECT_NE(no_mask.reason.find("pc=12"), std::string::npos) << no_mask.reason;

    const Compiled mask_edge = compile(kDataReadAfterMaskBackEdge, 0x4526A002ull);
    EXPECT_TRUE(mask_edge.spirv.empty()) << "a real mask on the back-edge";
    EXPECT_NE(mask_edge.reason.find("pc=14"), std::string::npos) << mask_edge.reason;

    // Control: without the read, both loops still compile -- the cleanup removed a stale value,
    // not the loop.
    uint32_t no_read[std::size(kDataReadNoMaskBefore)];
    std::copy(std::begin(kDataReadNoMaskBefore), std::end(kDataReadNoMaskBefore), no_read);
    no_read[12] = 0xBE810380u;   // s_mov_b32 s1, 0
    EXPECT_FALSE(compile(no_read, 0x4526A003ull).spirv.empty());
    uint32_t no_read_edge[std::size(kDataReadAfterMaskBackEdge)];
    std::copy(std::begin(kDataReadAfterMaskBackEdge), std::end(kDataReadAfterMaskBackEdge),
              no_read_edge);
    no_read_edge[14] = 0xBE810380u;   // s_mov_b32 s1, 0
    EXPECT_FALSE(compile(no_read_edge, 0x4526A004ull).spirv.empty());
}

TEST(FragmentLoopVccScratch, MaskReadAfterTheLoopSeesTheExitMask) {
    const Compiled loop = compile(kMaskReadAfterLoop, 0x4508A005ull);
    ASSERT_FALSE(loop.spirv.empty()) << loop.reason;
    if (!device_can_execute(loop.spirv))
        GTEST_SKIP() << "device cannot execute the fragment wave64 contract this module declares";
    const std::vector<uint8_t> pixel = centre_pixel(loop.spirv);
    ASSERT_EQ(pixel.size(), 4u) << "the triangle did not render";
    EXPECT_LT(pixel[0], 0x10) << "red is the accumulator: zero trips";
    EXPECT_GT(pixel[1], 0xF0) << "green is selected by the mask the loop exited with";
    EXPECT_LT(pixel[2], 0x10);
}

TEST(FragmentLoopVccScratch, CountedLoopDecodesAsDescribed) {
    std::vector<Rdna2Inst> ins;
    ASSERT_EQ(rdna2_walk(kCountedVccHiLoop, std::size(kCountedVccHiLoop), ins),
              std::size(kCountedVccHiLoop));
    const auto at = [&](uint32_t pc) -> const Rdna2Inst& {
        for (const Rdna2Inst& in : ins)
            if (in.pc == pc) return in;
        ADD_FAILURE() << "no instruction at pc " << pc;
        return ins.front();
    };
    EXPECT_EQ(at(5).fmt, Rdna2Format::SOP1);
    EXPECT_EQ(at(5).dst.value, 107);
    EXPECT_EQ(at(7).fmt, Rdna2Format::SOPC);
    EXPECT_EQ(at(7).src[0].value, 107);
    EXPECT_EQ(at(7).src[1].value, 2);
    for (uint32_t pc : {9u, 10u, 11u, 12u}) {
        EXPECT_EQ(at(pc).fmt, Rdna2Format::SOP2) << "pc " << pc;
        EXPECT_EQ(at(pc).dst.value, 106) << "pc " << pc;
    }
    EXPECT_EQ(at(9).src[0].value, 107);
    EXPECT_EQ(at(13).dst.value, 107);
    EXPECT_EQ(at(13).src[0].value, 107);
    EXPECT_EQ(at(16).src[0].value, 106);
    EXPECT_EQ(static_cast<int64_t>(at(8).pc) + at(8).len_dwords + at(8).simm16, 18);
    EXPECT_EQ(static_cast<int64_t>(at(17).pc) + at(17).len_dwords + at(17).simm16, 7);
    const CountedLoop loop = detect_counted_loop(ins);
    ASSERT_TRUE(loop.found) << "the fixture must take the counted-loop route";
    EXPECT_EQ(loop.header_pc, 7u);
    EXPECT_EQ(loop.backedge_pc, 17u);
}

TEST(FragmentLoopVccScratch, CountedLoopWithAVccHiCounterCompiles) {
    const Compiled loop = compile(kCountedVccHiLoop, 0x4686A001ull);
    ASSERT_FALSE(loop.spirv.empty())
        << "VCC's mask is dead from the header and its scalar halves are carried by their own "
           "phis; reason: "
        << loop.reason;
    EXPECT_EQ(loop.spirv[0], 0x07230203u);
}

TEST(FragmentLoopVccScratch, CountedLoopWithAVccHiCounterRunsExactlyFourIterations) {
    const Compiled loop = compile(kCountedVccHiLoop, 0x4686A002ull);
    ASSERT_FALSE(loop.spirv.empty()) << loop.reason;
    if (!device_can_execute(loop.spirv))
        GTEST_SKIP() << "device cannot execute the fragment wave64 contract this module declares";
    const std::vector<uint8_t> pixel = centre_pixel(loop.spirv);
    ASSERT_EQ(pixel.size(), 4u) << "the triangle did not render";
    // vcc_hi = 0, 2, 4, 6: four trips of +0.125 are 0.5. A vcc_hi seeded with anything but its
    // tracked 0 (the mask's dword, or an invented value) moves the trip count and leaves this band.
    for (int channel = 0; channel < 3; ++channel) {
        EXPECT_GT(pixel[channel], 0x70) << "channel " << channel;
        EXPECT_LT(pixel[channel], 0x90) << "channel " << channel;
    }
    // Control: bound five must leave the band, or the band proves nothing.
    uint32_t five_trips[std::size(kCountedVccHiLoop)];
    std::copy(std::begin(kCountedVccHiLoop), std::end(kCountedVccHiLoop), five_trips);
    five_trips[2] = 0x7E02028Au;   // v_mov_b32 v1, 10
    const Compiled longer = compile(five_trips, 0x4686A003ull);
    ASSERT_FALSE(longer.spirv.empty()) << longer.reason;
    const std::vector<uint8_t> longer_pixel = centre_pixel(longer.spirv);
    ASSERT_EQ(longer_pixel.size(), 4u);
    EXPECT_GT(longer_pixel[0], 0x98) << "five trips of +0.125 are 0.625";
    EXPECT_LT(longer_pixel[0], 0xA8);
}

TEST(FragmentLoopVccScratch, CountedLoopReadingAMaskHalfAsDataStillRejects) {
    // vcc_hi is NOT scalar data before the loop: it is the high dword of the compare's mask. The
    // header reads it as data, and its phi would be seeded with an invented zero.
    uint32_t untracked[std::size(kCountedVccHiLoop)];
    std::copy(std::begin(kCountedVccHiLoop), std::end(kCountedVccHiLoop), untracked);
    untracked[5] = 0xBE850380u;   // s_mov_b32 s5, 0 -- vcc_hi keeps the mask's dword
    const Compiled loop = compile(untracked, 0x4686A004ull);
    EXPECT_TRUE(loop.spirv.empty()) << "the entry value of vcc_hi is a mask dword, not data";
    EXPECT_NE(loop.reason.find("s107 is read from header pc=7"), std::string::npos)
        << loop.reason;
    EXPECT_NE(loop.reason.find("tracked-at-entry=0"), std::string::npos) << loop.reason;
}

TEST(FragmentLoopVccScratch, CountedLoopWithAMaskReadAfterTheExitStillRejects) {
    // On the exit path the pair was never redefined, so a mask read there would observe the
    // placeholder the back-edge carries round the loop.
    uint32_t mask_read[std::size(kCountedVccHiLoop)];
    std::copy(std::begin(kCountedVccHiLoop), std::end(kCountedVccHiLoop), mask_read);
    mask_read[18] = 0x02020105u;   // v_cndmask_b32 v1, v5, v0, vcc
    const Compiled loop = compile(mask_read, 0x4686A005ull);
    EXPECT_TRUE(loop.spirv.empty()) << "a mask read reachable from the header";
    EXPECT_NE(loop.reason.find("from header pc=7"), std::string::npos) << loop.reason;
    EXPECT_NE(loop.reason.find("blocker pc=18 kind=vop2-implicit-vcc"), std::string::npos)
        << loop.reason;
}
