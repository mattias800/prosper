// test_scc_do_while_loops — several bottom-tested SCC loops in one program (#4518).
//
// A do-while compiles to a body that ends in `s_cmp ...; s_cbranch_scc1 HEADER`. CountedLoop
// lowers that shape only when it is the program's single back-edge, and the multi-loop detector
// accepted only s_branch and s_cbranch_execnz back-edges. A fragment program with two do-while
// loops in sequence was therefore refused whole. Kena's blur programs are four such programs: two
// five-tap loops counted in VCC_LO, with the compare twenty dwords above the back-edge.
//
// The kernels are hand-written and assembled with llvm-mc -mcpu=gfx1030 -mattr=+wavefrontsize64
// (encodings and displacements are the assembler's). Loop 1 counts in VCC_LO and places vector
// ALU and a waitcnt between its compare and its back-edge, as Kena does. Loop 2 counts in s2.
// Each negative changes one thing about the positive (kCompareBeforeHeader needs two edits to do
// it: moving the compare and replacing the increment, which would otherwise write SCC itself), so
// the detector's refusal can come only from that property:
//   * a second exit (a break) inside the do-while;
//   * a scalar add, which may write SCC, between the compare and the back-edge;
//   * no compare inside the body, so SCC at the back-edge is a stale value from before the loop;
//   * a taken branch landing on the back-edge, so SCC there has a second producer.
// The two negatives that add a forward branch have three branches, which routes the whole program
// to the CFG dispatcher (`complex_graphics_cfg`, more than two branches); those are executed and
// must compute the guest's answer, and the route is asserted from the module's structure (the
// dispatcher is an OpSwitch state machine; the structured route emits one OpLoopMerge per loop).
// The other two negatives stay refused.
// The positives are executed: the program exports sum(0..4)=10 and sum(0..3)=6 as unorm bytes.
// A third positive puts a forward if inside an SCC do-while; it also has three branches, so it
// went to the dispatcher before this change and takes the structured route now.
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/render_runner.h"
#include "fixtures/spirv_triangle.h"
#include <gtest/gtest.h>

#include <cstdint>
#include <iterator>
#include <vector>

using namespace prosper::gpu;

namespace {

// vcc_lo=0; v1=0; do { v1+=vcc_lo; vcc_lo++; s_cmp_lt vcc_lo,5; v5=v1; waitcnt; v6=v5+1 } while(SCC)
// s2=0; v2=0; do { v2+=s2; s2++; s_cmp_lt s2,4 } while(SCC)
// export (v1/255, v2/255, 0, 1)
const std::vector<uint32_t> kTwoLoops = {
    0xBEEA0380u, 0x7E020280u, 0x4A02026Au, 0x806A816Au, 0xBF0A856Au, 0x7E0A0301u, 0xBF8C0000u,
    0x4A0C0A81u, 0xBF85FFF9u, 0xBE820380u, 0x7E040280u, 0x4A040402u, 0x81028102u, 0xBF0A8402u,
    0xBF85FFFCu, 0x7E020D01u, 0x7E040D02u, 0x100002FFu, 0x3B808081u, 0x100204FFu, 0x3B808081u,
    0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u,
};
// As kTwoLoops, with loop 2 continuing while SCC is CLEAR: s_cmp_ge_u32 s2,4; s_cbranch_scc0.
const std::vector<uint32_t> kOppositePolarity = {
    0xBEEA0380u, 0x7E020280u, 0x4A02026Au, 0x806A816Au, 0xBF0A856Au, 0x7E0A0301u, 0xBF8C0000u,
    0x4A0C0A81u, 0xBF85FFF9u, 0xBE820380u, 0x7E040280u, 0x4A040402u, 0x81028102u, 0xBF098402u,
    0xBF84FFFCu, 0x7E020D01u, 0x7E040D02u, 0x100002FFu, 0x3B808081u, 0x100204FFu, 0x3B808081u,
    0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u,
};
// N_exit. Loop 2 gains `s_cmp_eq_u32 s2,3; s_cbranch_scc1 <loop exit>` after its increment.
const std::vector<uint32_t> kInteriorExit = {
    0xBEEA0380u, 0x7E020280u, 0x4A02026Au, 0x806A816Au, 0xBF0A856Au, 0x7E0A0301u, 0xBF8C0000u,
    0x4A0C0A81u, 0xBF85FFF9u, 0xBE820380u, 0x7E040280u, 0x4A040402u, 0x81028102u, 0xBF068302u,
    0xBF850002u, 0xBF0A8402u, 0xBF85FFFAu, 0x7E020D01u, 0x7E040D02u, 0x100002FFu, 0x3B808081u,
    0x100204FFu, 0x3B808081u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u,
};
// N_clobber. Loop 2 gains `s_add_u32 s3,s3,1` between its compare and its back-edge.
const std::vector<uint32_t> kScalarAfterCompare = {
    0xBEEA0380u, 0x7E020280u, 0x4A02026Au, 0x806A816Au, 0xBF0A856Au, 0x7E0A0301u, 0xBF8C0000u,
    0x4A0C0A81u, 0xBF85FFF9u, 0xBE820380u, 0x7E040280u, 0x4A040402u, 0x81028102u, 0xBF0A8402u,
    0x80038103u, 0xBF85FFFBu, 0x7E020D01u, 0x7E040D02u, 0x100002FFu, 0x3B808081u, 0x100204FFu,
    0x3B808081u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u,
};
// N_stale. Loop 2's compare moves above its header and its increment becomes s_mov_b32 s2,1
// (no SCC write), so nothing in the body produces the SCC its back-edge reads.
const std::vector<uint32_t> kCompareBeforeHeader = {
    0xBEEA0380u, 0x7E020280u, 0x4A02026Au, 0x806A816Au, 0xBF0A856Au, 0x7E0A0301u, 0xBF8C0000u,
    0x4A0C0A81u, 0xBF85FFF9u, 0xBE820380u, 0x7E040280u, 0xBF0A8402u, 0x4A040402u, 0xBE820381u,
    0xBF85FFFDu, 0x7E020D01u, 0x7E040D02u, 0x100002FFu, 0x3B808081u, 0x100204FFu, 0x3B808081u,
    0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u,
};
// N_land. Loop 2 gains `s_cmp_lg_u32 s2,2; s_cbranch_scc0 <back-edge>` before its compare. When s2
// reaches 2 the branch is taken and arrives at the back-edge with SCC clear, so the loop exits with
// v2 = 0+1 = 1 instead of continuing.
const std::vector<uint32_t> kBranchOntoBackEdge = {
    0xBEEA0380u, 0x7E020280u, 0x4A02026Au, 0x806A816Au, 0xBF0A856Au, 0x7E0A0301u, 0xBF8C0000u,
    0x4A0C0A81u, 0xBF85FFF9u, 0xBE820380u, 0x7E040280u, 0x4A040402u, 0x81028102u, 0xBF078202u,
    0xBF840001u, 0xBF0A8402u, 0xBF85FFFAu, 0x7E020D01u, 0x7E040D02u, 0x100002FFu, 0x3B808081u,
    0x100204FFu, 0x3B808081u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u,
};
// P_if. Loop 2's body is `if (s2 != 2) v2 += s2` (s_cmp_eq_u32 s2,2; s_cbranch_scc1 over the add),
// taken on one of four iterations: v2 = 0+1+3 = 4.
const std::vector<uint32_t> kForwardIfInBody = {
    0xBEEA0380u, 0x7E020280u, 0x4A02026Au, 0x806A816Au, 0xBF0A856Au, 0x7E0A0301u, 0xBF8C0000u,
    0x4A0C0A81u, 0xBF85FFF9u, 0xBE820380u, 0x7E040280u, 0xBF068202u, 0xBF850001u, 0x4A040402u,
    0x81028102u, 0xBF0A8402u, 0xBF85FFFAu, 0x7E020D01u, 0x7E040D02u, 0x100002FFu, 0x3B808081u,
    0x100204FFu, 0x3B808081u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u,
};

std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& words) {
    std::vector<Rdna2Inst> ins;
    const size_t consumed = rdna2_walk(words.data(), words.size(), ins);
    EXPECT_EQ(consumed, words.size());
    return ins;
}

std::vector<DivLoop> loops_of(const std::vector<uint32_t>& words) {
    return detect_divergent_loops(decode(words), {}, /*exact_fragment_wave_breaks*/ true);
}

std::vector<uint32_t> fragment(const std::vector<uint32_t>& words, uint32_t tag) {
    return recompile_fragment(words.data(), words.size(), nullptr, nullptr, UINT32_MAX, nullptr,
                              false, {RecompileDiagnosticStage::Fragment, tag});
}

// Count SPIR-V instructions with opcode `op` (OpLoopMerge = 246, OpSwitch = 251).
size_t count_op(const std::vector<uint32_t>& spirv, uint32_t op) {
    size_t n = 0;
    for (size_t i = 5; i < spirv.size();) {
        const uint32_t words = spirv[i] >> 16;
        if (!words) break;
        if ((spirv[i] & 0xffffu) == op) ++n;
        i += words;
    }
    return n;
}
constexpr uint32_t kOpLoopMerge = 246, kOpSwitch = 251;

TEST(SccDoWhileLoops, DetectorAcceptsSequentialBottomTestedLoops) {
    const auto loops = loops_of(kTwoLoops);
    ASSERT_EQ(loops.size(), 2u);
    for (const DivLoop& L : loops) {
        EXPECT_TRUE(L.bottom_tested);
        EXPECT_EQ(L.condition, DivLoop::Condition::Scc);
        EXPECT_EQ(L.exit_branch_pc, L.backedge_pc);
        EXPECT_TRUE(L.continue_on_set);   // s_cbranch_scc1: continue while SCC is set
    }
    EXPECT_EQ(loops[0].header_pc, 2u);
    EXPECT_EQ(loops[0].backedge_pc, 8u);
    EXPECT_EQ(loops[1].header_pc, 11u);
    EXPECT_EQ(loops[1].backedge_pc, 14u);

    const auto with_if = loops_of(kForwardIfInBody);
    ASSERT_EQ(with_if.size(), 2u) << "a forward if inside the body stays inside the loop";
    EXPECT_TRUE(with_if[1].bottom_tested);
    EXPECT_EQ(with_if[1].header_pc, 11u);
    EXPECT_EQ(with_if[1].backedge_pc, 16u);

    const auto opposite = loops_of(kOppositePolarity);
    ASSERT_EQ(opposite.size(), 2u);
    EXPECT_TRUE(opposite[0].continue_on_set);
    EXPECT_FALSE(opposite[1].continue_on_set);   // s_cbranch_scc0: continue while SCC is clear
}

TEST(SccDoWhileLoops, DetectorRefusesWhatItCannotProve) {
    EXPECT_TRUE(loops_of(kInteriorExit).empty()) << "a break inside an SCC do-while";
    EXPECT_TRUE(loops_of(kScalarAfterCompare).empty()) << "possible SCC write after the compare";
    EXPECT_TRUE(loops_of(kCompareBeforeHeader).empty()) << "no compare inside the body";
    EXPECT_TRUE(loops_of(kBranchOntoBackEdge).empty()) << "a second SCC producer reaches the test";

    // Two branches: no other route admits these, so the refusal reaches the recompile.
    EXPECT_TRUE(fragment(kScalarAfterCompare, 0x45180004u).empty());
    EXPECT_TRUE(fragment(kCompareBeforeHeader, 0x45180005u).empty());
}

// Render `program` over a triangle and count covered pixels that do not read (red, green, 0, 255).
// `skipped` is set when the device cannot run a required fragment wave64.
struct RenderResult { unsigned covered = 0, wrong = 0; bool skipped = false; };
RenderResult render(const std::vector<uint32_t>& program, uint8_t red, uint8_t green) {
    RenderResult result;
    const auto& ctx = prosper::test::render_vk_ctx();
    const bool wave64 = ctx.subgroup_size_control && ctx.min_subgroup_size <= 64 &&
                        ctx.max_subgroup_size >= 64 &&
                        (ctx.required_subgroup_size_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
                        (ctx.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT);
    if (fragment_spirv_required_subgroup_size(program) == 64 && !wave64) {
        result.skipped = true;
        return result;
    }
    const std::vector<uint32_t> vertex(std::begin(kTriVertSpv), std::end(kTriVertSpv));
    // Coverage reference: an opaque yellow program over the same triangle.
    const uint32_t yellow[] = {0x7e0002f2u, 0x7e0202f2u, 0x7e040280u, 0x7e0602f2u,
                               0xf800180fu, 0x03020100u, 0xbf810000u};
    const auto coverage =
        prosper::test::render_triangle_rgba(vertex, recompile_fragment(yellow, std::size(yellow)),
                                            64, 64);
    const auto pixels = prosper::test::render_triangle_rgba(vertex, program, 64, 64);
    EXPECT_EQ(coverage.size(), 64u * 64u * 4u);
    EXPECT_EQ(pixels.size(), coverage.size());
    if (pixels.size() != coverage.size()) return result;
    for (size_t i = 0; i + 3 < pixels.size(); i += 4) {
        if (coverage[i] <= 200 || coverage[i + 1] <= 200) continue;
        ++result.covered;
        if (pixels[i] != red || pixels[i + 1] != green || pixels[i + 2] != 0 ||
            pixels[i + 3] != 255)
            ++result.wrong;
    }
    return result;
}

TEST(SccDoWhileLoops, FragmentProgramRunsBothLoopsToCompletion) {
    const auto two = fragment(kTwoLoops, 0x45180001u);
    const auto opposite = fragment(kOppositePolarity, 0x45180002u);
    ASSERT_FALSE(two.empty()) << "two sequential SCC do-while loops recompile";
    ASSERT_FALSE(opposite.empty()) << "opposite back-edge polarity recompiles";
    for (const auto* program : {&two, &opposite}) {
        EXPECT_EQ(count_op(*program, kOpLoopMerge), 2u) << "one structured loop per do-while";
        EXPECT_EQ(count_op(*program, kOpSwitch), 0u) << "not the CFG dispatcher";
    }
    for (const auto* program : {&two, &opposite}) {
        const RenderResult r = render(*program, 10, 6);
        if (r.skipped) GTEST_SKIP() << "device cannot execute the required fragment wave64 contract";
        EXPECT_GT(r.covered, 256u);
        EXPECT_EQ(r.wrong, 0u) << "every covered pixel exports sum(0..4)=10 and sum(0..3)=6";
    }
}

TEST(SccDoWhileLoops, ForwardIfInsideTheBodyTakesTheStructuredRoute) {
    // Three branches: before #4518 this program went to the CFG dispatcher.
    const auto with_if = fragment(kForwardIfInBody, 0x45180007u);
    ASSERT_FALSE(with_if.empty());
    EXPECT_EQ(count_op(with_if, kOpLoopMerge), 2u);
    EXPECT_EQ(count_op(with_if, kOpSwitch), 0u);
    const RenderResult r = render(with_if, 10, 4);
    if (r.skipped) GTEST_SKIP() << "device cannot execute the required fragment wave64 contract";
    EXPECT_GT(r.covered, 256u);
    EXPECT_EQ(r.wrong, 0u) << "the if skips s2 == 2: v2 = 0+1+3 = 4";
}

TEST(SccDoWhileLoops, DispatcherRoutedNegativesComputeTheGuestAnswer) {
    // The break leaves loop 2 after s2 reaches 3: v2 = 0+1+2 = 3.
    const auto with_break = fragment(kInteriorExit, 0x45180003u);
    // The landing branch is taken at s2 == 2 with SCC clear, so loop 2 exits with v2 = 1.
    const auto with_landing = fragment(kBranchOntoBackEdge, 0x45180006u);
    ASSERT_FALSE(with_break.empty());
    ASSERT_FALSE(with_landing.empty());
    EXPECT_GE(count_op(with_break, kOpSwitch), 1u) << "the CFG dispatcher, not a structured loop";
    EXPECT_GE(count_op(with_landing, kOpSwitch), 1u) << "the CFG dispatcher, not a structured loop";
    const RenderResult a = render(with_break, 10, 3);
    const RenderResult b = render(with_landing, 10, 1);
    if (a.skipped || b.skipped)
        GTEST_SKIP() << "device cannot execute the required fragment wave64 contract";
    EXPECT_GT(a.covered, 256u);
    EXPECT_EQ(a.wrong, 0u);
    EXPECT_GT(b.covered, 256u);
    EXPECT_EQ(b.wrong, 0u);
}

} // namespace
