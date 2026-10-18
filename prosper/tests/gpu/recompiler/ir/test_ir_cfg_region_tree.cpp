// test_ir_cfg_region_tree -- build_region_tree on synthetic streams: the region kinds step 1
// builds, each RegionPolicy rule, and every refusal that leaves the program Unstructured.
#include "gpu/recompiler/ir/cfg/region_tree.hpp"

#include "gpu/recompiler/ir/cfg_streams.hpp"

#include <gtest/gtest.h>

#include <string>
#include <unordered_set>
#include <vector>

using namespace prosper::gpu::ir::cfg;
using namespace prosper::test::cfg_streams;
using Kind = Region::Kind;

namespace {

struct Built {
    std::vector<prosper::gpu::Rdna2Inst> ins;
    GuestCfg cfg;
    RegionTree tree;
};

RegionPolicy graphics() {
    return RegionPolicy{};
}
RegionPolicy compute() {
    RegionPolicy p;
    p.allow_vcc = false;
    p.compute_wave_branches = true;
    return p;
}

Built build(const std::vector<uint32_t>& words, RegionPolicy policy) {
    Built b;
    b.ins = walk(words);
    b.cfg = build_guest_cfg(b.ins, SplitPolicy{});
    policy.code = words.data();
    policy.dwords = words.size();
    b.tree = build_region_tree(b.cfg, b.ins, policy);
    return b;
}

std::vector<const Region*> ifs(const RegionTree& tree) {
    std::vector<const Region*> out;
    for (const Region& r : tree.regions)
        if (r.kind == Kind::IfThen || r.kind == Kind::IfElse) out.push_back(&r);
    return out;
}

void expect_unstructured(const Built& b, const std::string& reason_part) {
    ASSERT_TRUE(b.tree.has_unstructured());
    ASSERT_EQ(b.tree.regions.size(), 1u);
    EXPECT_EQ(b.tree.regions[b.tree.root].kind, Kind::Unstructured);
    EXPECT_NE(b.tree.regions[b.tree.root].reason.find(reason_part), std::string::npos)
        << b.tree.regions[b.tree.root].reason;
}

TEST(IrRegionTree, StraightLineIsASeqOfOneBlock) {
    const Built b = build({kSMovS0Zero, kEndpgm}, graphics());
    ASSERT_FALSE(b.tree.has_unstructured());
    const Region& root = b.tree.regions[b.tree.root];
    EXPECT_EQ(root.kind, Kind::Seq);
    ASSERT_EQ(root.children.size(), 1u);
    EXPECT_EQ(b.tree.regions[root.children[0]].kind, Kind::Block);
}

TEST(IrRegionTree, ForwardScalarIfIsIfThen) {
    // 0: scc0 -> 2; 1: s_mov; 2: v_nop; 3: s_endpgm (a target AT s_endpgm is an early-out)
    const Built b = build({sopp(kScc0, 1), kSMovS0Zero, kVNop, kEndpgm}, graphics());
    const auto found = ifs(b.tree);
    ASSERT_EQ(found.size(), 1u);
    const Region& r = *found[0];
    EXPECT_EQ(r.kind, Kind::IfThen);
    EXPECT_EQ(r.header, 0u);
    EXPECT_EQ(r.merge, 2u);
    EXPECT_EQ(r.taken, 2u);
    EXPECT_EQ(r.condition.flag, BranchCondition::Flag::Scc);
    EXPECT_TRUE(r.condition.taken_on_zero);
    ASSERT_EQ(r.children.size(), 1u);
    EXPECT_EQ(b.tree.regions[r.children[0]].first_block, 1u);
}

TEST(IrRegionTree, ThenArmEndingInSBranchIsIfElse) {
    // 0: vccnz -> 3; 1: s_mov; 2: s_branch -> 4; 3: s_mov; 4: s_endpgm
    const Built b =
        build({sopp(kVccnz, 2), kSMovS0Zero, sopp(kBranch, 1), kSMovS0Zero, kEndpgm}, graphics());
    const auto found = ifs(b.tree);
    ASSERT_EQ(found.size(), 1u);
    const Region& r = *found[0];
    EXPECT_EQ(r.kind, Kind::IfElse);
    EXPECT_EQ(r.else_jump, 1u);
    EXPECT_EQ(r.taken, 2u);
    EXPECT_EQ(r.merge, 3u);
    EXPECT_EQ(r.condition.flag, BranchCondition::Flag::Vcc);
    EXPECT_FALSE(r.condition.taken_on_zero);
    ASSERT_EQ(r.children.size(), 2u);
}

TEST(IrRegionTree, NestedIfsNest) {
    // 0: scc0 -> 4; 1: scc1 -> 3; 2: s_mov; 3: s_mov; 4: v_nop; 5: s_endpgm
    const Built b = build(
        {sopp(kScc0, 3), sopp(kScc1, 1), kSMovS0Zero, kSMovS0Zero, kVNop, kEndpgm}, graphics());
    const auto found = ifs(b.tree);
    ASSERT_EQ(found.size(), 2u);
    // Children are pushed before their parent: the inner if comes first.
    EXPECT_EQ(found[0]->header, 1u);
    EXPECT_EQ(found[1]->header, 0u);
    const Region& outer_then = b.tree.regions[found[1]->children[0]];
    EXPECT_EQ(outer_then.first_block, 1u);
    EXPECT_EQ(outer_then.end_block, found[1]->merge);
}

TEST(IrRegionTree, EarlyOutPastEndpgmEndsTheProgram) {
    // 0: execz -> 3, which is a second s_endpgm past the first (pc 2)
    const Built b = build({sopp(kExecz, 2), kVNop, kEndpgm, kEndpgm}, graphics());
    const auto found = ifs(b.tree);
    ASSERT_EQ(found.size(), 1u);
    EXPECT_EQ(found[0]->kind, Kind::IfThen);
    EXPECT_EQ(found[0]->merge, kNone);
    EXPECT_EQ(found[0]->taken, kNone);
}

TEST(IrRegionTree, EarlyOutToANonTerminatingTargetIsRefused) {
    const Built b = build({sopp(kExecz, 2), kVNop, kEndpgm, kVNop, kEndpgm}, graphics());
    expect_unstructured(b, "nonterminal early target");
}

TEST(IrRegionTree, LinearizedBranchIsNotAnIf) {
    const std::vector<uint32_t> words = {sopp(kScc0, 1), kVNop, kEndpgm};
    const std::unordered_set<uint32_t> safe = {0};
    RegionPolicy policy = graphics();
    policy.linearized = &safe;
    const Built b = build(words, policy);
    EXPECT_FALSE(b.tree.has_unstructured());
    EXPECT_TRUE(ifs(b.tree).empty());
}

TEST(IrRegionTree, ComputeExeczWithBarrierIsRefused) {
    const Built b = build({sopp(kExecz, 2), kBarrier, kVNop, kEndpgm}, compute());
    expect_unstructured(b, "contains barrier");
    const Built clean = build({sopp(kExecz, 1), kVNop, kEndpgm}, compute());
    ASSERT_EQ(ifs(clean.tree).size(), 1u);
    EXPECT_EQ(ifs(clean.tree)[0]->condition.flag, BranchCondition::Flag::Exec);
}

TEST(IrRegionTree, ComputeExeczScalarVccWriteLiveAtMergeIsRefused) {
    // s_mov vcc_lo inside the arm, then a vccz after the merge reads it.
    const Built b = build({sopp(kExecz, 1), kSMovVccLoZero, sopp(kVccz, 0), kEndpgm}, compute());
    expect_unstructured(b, "leaves vcc-half");
}

TEST(IrRegionTree, ComputeVccBranchIsAWaveVoteAndRefusesBarriers) {
    const Built ok = build({sopp(kVccz, 1), kVNop, kEndpgm}, compute());
    ASSERT_EQ(ifs(ok.tree).size(), 1u);
    EXPECT_TRUE(ifs(ok.tree)[0]->condition.wave_vote);
    EXPECT_FALSE(ifs(ok.tree)[0]->condition.uniform_workgroup);
    const Built barrier = build({sopp(kVccz, 2), kBarrier, kVNop, kEndpgm}, compute());
    expect_unstructured(barrier, "synchronized op");
}

TEST(IrRegionTree, StageWithoutVoteModelRefusesVccAndExec) {
    RegionPolicy none;
    none.allow_vcc = false;
    expect_unstructured(build({sopp(kVccz, 1), kVNop, kEndpgm}, none), "unsupported vcc branch");
    expect_unstructured(build({sopp(kExecz, 1), kVNop, kEndpgm}, none), "unsupported execz");
}

TEST(IrRegionTree, ForwardExecnzIsRefused) {
    expect_unstructured(build({sopp(kExecnz, 1), kVNop, kEndpgm}, graphics()), "unclaimed execnz");
}

TEST(IrRegionTree, BackwardBranchIsUnstructured) {
    expect_unstructured(build({kVNop, sopp(kScc1, -2), kEndpgm}, graphics()), "nonforward");
    expect_unstructured(build({kVNop, sopp(kBranch, -2), kEndpgm}, graphics()),
                        "unclaimed s_branch");
}

TEST(IrRegionTree, UnclaimedForwardSBranchIsUnstructured) {
    expect_unstructured(build({sopp(kBranch, 1), kVNop, kEndpgm}, graphics()),
                        "unclaimed s_branch");
}

TEST(IrRegionTree, CallIsUnstructured) {
    expect_unstructured(build({call(1), kVNop, kEndpgm}, graphics()), "indirect");
    expect_unstructured(build({kSwappc, kVNop, kEndpgm}, graphics()), "indirect");
}

TEST(IrRegionTree, IndirectJumpEndingTheStreamLeavesTheProgram) {
    // The merged NGG ES prolog shape: no s_endpgm, s_setpc_b64 then s_code_end padding.
    const Built b = build({sopp(kScc0, 1), kVNop, kVNop, kSetpc, kCodeEnd, kCodeEnd}, graphics());
    ASSERT_FALSE(b.tree.has_unstructured()) << b.tree.regions[b.tree.root].reason;
    EXPECT_EQ(ifs(b.tree).size(), 1u);
}

TEST(IrRegionTree, TrapIsAPlainBlock) {
    const Built b = build({kTrap, kVNop, kEndpgm}, graphics());
    EXPECT_FALSE(b.tree.has_unstructured());
    EXPECT_EQ(b.tree.regions[b.tree.root].children.size(), 2u);
}

TEST(IrRegionTree, CascadeElseMergeMayEscapeItsRange) {
    // Outer if/else 0 -> 5 (else jump at 4 -> 7). Its then-arm holds an inner if/else
    // 1 -> 3 whose then-arm jumps straight to the OUTER merge 7.
    // 0: scc0 -> 5; 1: scc1 -> 3; 2: s_branch -> 7; 3: v_nop; 4: s_branch -> 7;
    // 5: v_nop; 6: v_nop; 7: s_endpgm
    const Built b = build({sopp(kScc0, 4), sopp(kScc1, 1), sopp(kBranch, 4), kVNop,
                           sopp(kBranch, 2), kVNop, kVNop, kEndpgm},
                          graphics());
    ASSERT_FALSE(b.tree.has_unstructured()) << b.tree.regions[b.tree.root].reason;
    const auto found = ifs(b.tree);
    ASSERT_EQ(found.size(), 2u);
    const Region& inner = *found[0];
    EXPECT_EQ(inner.kind, Kind::IfElse);
    EXPECT_EQ(b.cfg.blocks[inner.merge].pc, 7u);
    EXPECT_LT(inner.end_block, inner.merge);   // clamped to the outer then-arm
}

TEST(IrRegionTree, IfLeavingItsArmIsUnstructured) {
    // The inner if at 1 targets 5, inside the outer if/else's ELSE arm [4, 6).
    // 0: scc0 -> 4; 1: scc1 -> 5; 2: v_nop; 3: s_branch -> 6; 4: v_nop; 5: v_nop; 6: s_endpgm
    const Built b =
        build({sopp(kScc0, 3), sopp(kScc1, 3), kVNop, sopp(kBranch, 2), kVNop, kVNop, kEndpgm},
              graphics());
    expect_unstructured(b, "leaves its region");
}

}   // namespace
