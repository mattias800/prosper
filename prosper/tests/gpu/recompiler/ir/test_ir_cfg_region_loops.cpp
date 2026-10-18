// test_ir_cfg_region_loops -- Loop and CountedLoop regions (region_loops.cpp) on synthetic
// streams: each loop shape the detectors admit, ifs and loops nested inside a loop, breaks, the
// RegionPolicy loop rules, and every loop refusal that leaves the program Unstructured.
// Comments give each stream as `pc: instruction`; a branch names its target pc.
#include "gpu/recompiler/ir/cfg/region_tree.hpp"

#include "gpu/recompiler/ir/cfg_streams.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace prosper::gpu::ir::cfg;
using namespace prosper::test::cfg_streams;
using Kind = Region::Kind;
using Flag = BranchCondition::Flag;

namespace {

struct Built {
    std::vector<prosper::gpu::Rdna2Inst> ins;
    GuestCfg cfg;
    RegionTree tree;
};

Built build(const std::vector<uint32_t>& words, RegionPolicy policy = {}) {
    Built b;
    b.ins = walk(words);
    b.cfg = build_guest_cfg(b.ins, SplitPolicy{});
    policy.code = words.data();
    policy.dwords = words.size();
    b.tree = build_region_tree(b.cfg, b.ins, policy);
    return b;
}

std::vector<const Region*> of_kind(const RegionTree& tree, Kind kind) {
    std::vector<const Region*> out;
    for (const Region& r : tree.regions)
        if (r.kind == kind) out.push_back(&r);
    return out;
}

const Region& only(const Built& b, Kind kind) {
    const auto found = of_kind(b.tree, kind);
    EXPECT_EQ(found.size(), 1u) << (b.tree.has_unstructured() ? b.tree.regions[0].reason : "");
    static const Region none;
    return found.empty() ? none : *found[0];
}

void expect_unstructured(const Built& b, const std::string& reason_part) {
    ASSERT_TRUE(b.tree.has_unstructured());
    EXPECT_NE(b.tree.regions[b.tree.root].reason.find(reason_part), std::string::npos)
        << b.tree.regions[b.tree.root].reason;
}

TEST(IrRegionLoops, TopTestedExeczLoopIsALoop) {
    // 0: s_mov; 1: execz -> 4 (header and exit); 2: v_nop; 3: s_branch -> 1; 4: s_endpgm
    const Built b = build({kSMovS0Zero, sopp(kExecz, 2), kVNop, sopp(kBranch, -3), kEndpgm});
    const Region& loop = only(b, Kind::Loop);
    EXPECT_EQ(loop.header, 1u);   // blocks: [0] [1] [2,3] [4]
    EXPECT_EQ(loop.exit_branch, 1u);
    EXPECT_EQ(loop.continue_block, 2u);
    EXPECT_EQ(loop.merge, 3u);
    EXPECT_EQ(loop.first_block, 1u);
    EXPECT_EQ(loop.end_block, 3u);
    EXPECT_EQ(loop.condition.flag, Flag::Exec);
    EXPECT_TRUE(loop.loop.continue_on_set);
    EXPECT_FALSE(loop.loop.bottom_tested);
    ASSERT_EQ(loop.children.size(), 1u);
    const Region& body = b.tree.regions[loop.children[0]];
    EXPECT_EQ(body.kind, Kind::Seq);
    EXPECT_EQ(body.first_block, 1u);
    ASSERT_EQ(body.children.size(), 2u);
    EXPECT_EQ(b.tree.regions[body.children[0]].kind, Kind::Block);   // the header itself
    EXPECT_TRUE(of_kind(b.tree, Kind::IfThen).empty());   // the exit is claimed, not an if
}

TEST(IrRegionLoops, CountedTopTestedLoopIsACountedLoop) {
    // 0: s_mov; 1: scc0 -> 3; 2: s_branch -> 0; 3: s_endpgm
    const Built b = build({kSMovS0Zero, sopp(kScc0, 1), sopp(kBranch, -3), kEndpgm});
    const Region& loop = only(b, Kind::CountedLoop);
    EXPECT_EQ(loop.header, 0u);   // blocks: [0,1] [2] [3]
    EXPECT_EQ(loop.exit_branch, 0u);
    EXPECT_EQ(loop.continue_block, 1u);
    EXPECT_EQ(loop.merge, 2u);
    EXPECT_EQ(loop.condition.flag, Flag::Scc);
    EXPECT_TRUE(loop.loop.continue_on_set);   // exits on SCC==0
    EXPECT_FALSE(loop.loop.bottom_tested);
    EXPECT_TRUE(of_kind(b.tree, Kind::Loop).empty());
}

TEST(IrRegionLoops, CountedDoWhileIsBottomTested) {
    // 0: v_nop; 1: scc0 -> 0 (continues while SCC==0); 2: s_endpgm
    const Built b = build({kVNop, sopp(kScc0, -2), kEndpgm});
    const Region& loop = only(b, Kind::CountedLoop);
    EXPECT_EQ(loop.header, 0u);
    EXPECT_EQ(loop.continue_block, 0u);
    EXPECT_EQ(loop.exit_branch, 0u);
    EXPECT_TRUE(loop.loop.bottom_tested);
    EXPECT_FALSE(loop.loop.continue_on_set);
}

TEST(IrRegionLoops, ExecnzBackEdgeIsABottomTestedExecLoop) {
    // 0: v_nop; 1: s_mov_b64 exec, 0; 2: execnz -> 0; 3: s_endpgm
    const Built b = build({kVNop, kSMovExecZero, sopp(kExecnz, -3), kEndpgm});
    const Region& loop = only(b, Kind::Loop);
    EXPECT_EQ(loop.condition.flag, Flag::Exec);
    EXPECT_TRUE(loop.loop.bottom_tested);
    EXPECT_EQ(loop.exit_branch, loop.continue_block);
}

TEST(IrRegionLoops, AnIfNestsInsideALoop) {
    // 0: s_mov; 1: execz -> 6; 2: scc0 -> 4; 3: v_nop; 4: v_nop; 5: s_branch -> 1; 6: s_endpgm
    const Built b = build(
        {kSMovS0Zero, sopp(kExecz, 4), sopp(kScc0, 1), kVNop, kVNop, sopp(kBranch, -5), kEndpgm});
    const Region& loop = only(b, Kind::Loop);
    const Region& inner = only(b, Kind::IfThen);
    EXPECT_GE(inner.first_block, loop.first_block);
    EXPECT_LE(inner.end_block, loop.end_block);
    EXPECT_EQ(inner.merge, loop.continue_block);   // the if rejoins at the latch block
    EXPECT_FALSE(inner.loop_break);
}

TEST(IrRegionLoops, LoopsNest) {
    // 0: s_mov; 1: execz -> 8; 2: v_nop; 3: execz -> 6; 4: v_nop; 5: s_branch -> 3;
    // 6: v_nop; 7: s_branch -> 1; 8: s_endpgm
    const Built b = build({kSMovS0Zero, sopp(kExecz, 6), kVNop, sopp(kExecz, 2), kVNop,
                           sopp(kBranch, -3), kVNop, sopp(kBranch, -7), kEndpgm});
    const auto loops = of_kind(b.tree, Kind::Loop);
    ASSERT_EQ(loops.size(), 2u) << b.tree.regions[0].reason;
    const Region& inner = *loops[0];   // children are pushed before their parent
    const Region& outer = *loops[1];
    EXPECT_GT(inner.header, outer.header);
    EXPECT_LT(inner.end_block, outer.end_block);
    EXPECT_EQ(inner.merge, inner.continue_block + 1);
}

TEST(IrRegionLoops, BreakIsAnIfOverTheRestOfTheBody) {
    // 0: s_mov; 1: execz -> 6 (header and exit); 2: v_nop; 3: execz -> 6 (break); 4: v_nop;
    // 5: execnz -> 1; 6: s_endpgm
    const Built b = build(
        {kSMovS0Zero, sopp(kExecz, 4), kVNop, sopp(kExecz, 2), kVNop, sopp(kExecnz, -5), kEndpgm});
    const Region& loop = only(b, Kind::Loop);
    ASSERT_EQ(loop.breaks.size(), 1u);
    EXPECT_EQ(loop.breaks[0], 2u);   // blocks: [0] [1] [2,3] [4,5] [6]
    const Region& brk = only(b, Kind::IfThen);
    EXPECT_TRUE(brk.loop_break);
    EXPECT_EQ(brk.header, 2u);
    EXPECT_EQ(brk.taken, loop.continue_block);
    EXPECT_EQ(brk.end_block, loop.end_block);
    EXPECT_EQ(brk.condition.flag, Flag::Exec);
}

TEST(IrRegionLoops, BreakInsideAnInnerIfIsRefused) {
    // 0: s_mov; 1: execz -> 7; 2: scc0 -> 5; 3: execz -> 7 (break inside the if); 4: v_nop;
    // 5: v_nop; 6: execnz -> 1; 7: s_endpgm
    const Built b = build({kSMovS0Zero, sopp(kExecz, 5), sopp(kScc0, 2), sopp(kExecz, 3), kVNop,
                           kVNop, sopp(kExecnz, -6), kEndpgm});
    expect_unstructured(b, "break leaves its region");
}

TEST(IrRegionLoops, VccExitNeedsTheFragmentWaveVoteOrAUniformityProof) {
    // 0: s_mov; 1: vccz -> 4; 2: v_nop; 3: s_branch -> 1; 4: s_endpgm
    const std::vector<uint32_t> words = {kSMovS0Zero, sopp(kVccz, 2), kVNop, sopp(kBranch, -3),
                                         kEndpgm};
    expect_unstructured(build(words), "vcc-exit-not-proven-wave-uniform");
    RegionPolicy fragment;
    fragment.fragment_wave_breaks = true;
    EXPECT_EQ(only(build(words, fragment), Kind::Loop).condition.flag, Flag::Vcc);
}

TEST(IrRegionLoops, IrreducibleLoopIsUnstructured) {
    // 0: scc0 -> 3 (into the loop); 1: v_nop (header); 2: v_nop; 3: v_nop; 4: s_branch -> 1;
    // 5: s_endpgm. The header does not dominate the latch.
    const Built b = build({sopp(kScc0, 2), kVNop, kVNop, kVNop, sopp(kBranch, -4), kEndpgm});
    expect_unstructured(b, "irreducible back edge");
}

TEST(IrRegionLoops, BackwardBranchesThatCloseNoAdmittedLoopAreUnstructured) {
    // A backward vccz is neither detector's back edge.
    expect_unstructured(build({kVNop, sopp(kVccz, -2), kEndpgm}), "not a loop back edge");
    // An s_branch loop with no exit test: not counted (no exit), and the divergent rules need an
    // execnz back edge for a bottom-tested loop.
    expect_unstructured(build({kVNop, sopp(kBranch, -2), kEndpgm}),
                        "bottom-tested-needs-execnz-backedge");
}

TEST(IrRegionLoops, LinearizedBackwardBranchIsNotALoop) {
    const std::vector<uint32_t> words = {kVNop, sopp(kExecz, -2), kEndpgm};
    const std::unordered_set<uint32_t> safe = {1};
    RegionPolicy policy;
    policy.linearized = &safe;
    const Built b = build(words, policy);
    EXPECT_FALSE(b.tree.has_unstructured()) << b.tree.regions[0].reason;
    EXPECT_TRUE(of_kind(b.tree, Kind::Loop).empty());
    EXPECT_TRUE(of_kind(b.tree, Kind::CountedLoop).empty());
}

TEST(IrRegionLoops, ALoopAfterACountedLoopIsRefused) {
    // 0: s_mov; 1: scc0 -> 3; 2: s_branch -> 0 (counted); 3: v_nop; 4: s_mov_b64 exec, 0;
    // 5: execnz -> 3; 6: s_endpgm
    const Built b = build({kSMovS0Zero, sopp(kScc0, 1), sopp(kBranch, -3), kVNop, kSMovExecZero,
                           sopp(kExecnz, -3), kEndpgm});
    expect_unstructured(b, "loop after a counted loop");
}

TEST(IrRegionLoops, ALoopBeforeACountedLoopTakesTheGeneralRoute) {
    // emit_body's counted prelude scan refuses the execnz loop and falls back to the divergent
    // rules, which admit both loops. 0: v_nop; 1: s_mov_b64 exec, 0; 2: execnz -> 0; 3: s_mov;
    // 4: scc0 -> 6; 5: s_branch -> 3; 6: s_endpgm
    const Built b = build({kVNop, kSMovExecZero, sopp(kExecnz, -3), kSMovS0Zero, sopp(kScc0, 1),
                           sopp(kBranch, -3), kEndpgm});
    const auto loops = of_kind(b.tree, Kind::Loop);
    ASSERT_EQ(loops.size(), 2u) << b.tree.regions[0].reason;
    EXPECT_TRUE(of_kind(b.tree, Kind::CountedLoop).empty());
    EXPECT_EQ(loops[1]->condition.flag, Flag::Scc);
}

TEST(IrRegionLoops, CountedRouteExecGuardsAreNotIfs) {
    // Inside the loop. 0: s_mov; 1: scc0 -> 5; 2: execz -> 4; 3: v_nop; 4: s_branch -> 0;
    // 5: s_endpgm
    const Built inside =
        build({kSMovS0Zero, sopp(kScc0, 3), sopp(kExecz, 1), kVNop, sopp(kBranch, -5), kEndpgm});
    EXPECT_EQ(of_kind(inside.tree, Kind::CountedLoop).size(), 1u) << inside.tree.regions[0].reason;
    EXPECT_TRUE(of_kind(inside.tree, Kind::IfThen).empty());
    // Spanning the loop from before it. 0: execz -> 4; 1: s_mov; 2: scc0 -> 4; 3: s_branch -> 1;
    // 4: s_endpgm
    const Built spanning =
        build({sopp(kExecz, 3), kSMovS0Zero, sopp(kScc0, 1), sopp(kBranch, -3), kEndpgm});
    EXPECT_EQ(of_kind(spanning.tree, Kind::CountedLoop).size(), 1u)
        << spanning.tree.regions[0].reason;
    EXPECT_TRUE(of_kind(spanning.tree, Kind::IfThen).empty());
}

TEST(IrRegionLoops, TwoSequentialLoopsAreBothLoops) {
    // Two execz loops in a row. Two s_branch candidates, so not the counted route.
    // 0: s_mov; 1: execz -> 4; 2: v_nop; 3: s_branch -> 1; 4: execz -> 7; 5: v_nop;
    // 6: s_branch -> 4; 7: s_endpgm
    const Built b = build({kSMovS0Zero, sopp(kExecz, 2), kVNop, sopp(kBranch, -3), sopp(kExecz, 2),
                           kVNop, sopp(kBranch, -3), kEndpgm});
    EXPECT_EQ(of_kind(b.tree, Kind::Loop).size(), 2u) << b.tree.regions[0].reason;
}

// detect_divergent_loops keeps "no exit test yet" as exit_branch_pc == 0, so a loop whose exit
// test is the instruction at pc 0 reads as having none. The tree uses the same rule and refuses
// it the same way (the shadow agrees); real programs do not start with a loop exit.
TEST(IrRegionLoops, AnExitTestAtPcZeroIsRefusedAsTheDetectorRefusesIt) {
    // 0: execz -> 3; 1: v_nop; 2: s_branch -> 0; 3: s_endpgm
    expect_unstructured(build({sopp(kExecz, 2), kVNop, sopp(kBranch, -3), kEndpgm}),
                        "bottom-tested-needs-execnz-backedge");
}

}   // namespace
