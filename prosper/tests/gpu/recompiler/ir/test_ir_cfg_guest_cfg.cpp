// test_ir_cfg_guest_cfg -- build_guest_cfg on synthetic streams: one case per Terminator, plus
// preds, rpo, dominators, back edges and reachability (IR_CONTROL_FLOW_PLAN step 1).
#include "gpu/recompiler/ir/cfg/guest_cfg.hpp"

#include "gpu/recompiler/ir/cfg_streams.hpp"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

using namespace prosper::gpu::ir::cfg;
using namespace prosper::test::cfg_streams;

namespace {

GuestCfg cfg_of(const std::vector<uint32_t>& words) {
    const auto ins = walk(words);
    return build_guest_cfg(ins, SplitPolicy{});
}

TEST(IrGuestCfg, StraightLineIsOneEndBlock) {
    const GuestCfg cfg = cfg_of({kSMovS0Zero, kVNop, kEndpgm});
    ASSERT_EQ(cfg.blocks.size(), 1u);
    EXPECT_EQ(cfg.blocks[0].term, Terminator::End);
    EXPECT_EQ(cfg.blocks[0].first, 0u);
    EXPECT_EQ(cfg.blocks[0].last, 2u);
    EXPECT_EQ(cfg.blocks[0].succ[0], kNone);
    EXPECT_EQ(cfg.blocks[0].succ[1], kNone);
    EXPECT_EQ(cfg.rpo, std::vector<uint32_t>{0});
    EXPECT_EQ(cfg.idom[0], 0u);
    EXPECT_TRUE(cfg.back_edges.empty());
}

TEST(IrGuestCfg, EveryConditionalFormSplitsIntoFallthroughAndTaken) {
    const std::pair<uint32_t, Terminator> forms[] = {
        {kScc0, Terminator::CondScc0},   {kScc1, Terminator::CondScc1},
        {kVccz, Terminator::CondVccz},   {kVccnz, Terminator::CondVccnz},
        {kExecz, Terminator::CondExecz}, {kExecnz, Terminator::CondExecnz},
    };
    for (const auto& [opcode, term] : forms) {
        // 0: branch -> 2; 1: s_mov; 2: s_endpgm
        const GuestCfg cfg = cfg_of({sopp(opcode, 1), kSMovS0Zero, kEndpgm});
        ASSERT_EQ(cfg.blocks.size(), 3u) << opcode;
        EXPECT_EQ(cfg.blocks[0].term, term);
        EXPECT_TRUE(terminator_is_conditional(term));
        EXPECT_EQ(cfg.blocks[0].succ[0], 1u);
        EXPECT_EQ(cfg.blocks[0].succ[1], 2u);
        EXPECT_EQ(cfg.blocks[0].taken_pc, 2u);
        EXPECT_EQ(cfg.blocks[1].term, Terminator::Fallthrough);
        EXPECT_EQ(cfg.blocks[1].succ[0], 2u);
        EXPECT_EQ(cfg.preds[2], (std::vector<uint32_t>{0, 1}));
        EXPECT_EQ(cfg.idom[1], 0u);
        EXPECT_EQ(cfg.idom[2], 0u);
    }
}

TEST(IrGuestCfg, IfElseDiamondDominators) {
    // 0: scc0 -> 3; 1: s_mov; 2: s_branch -> 4; 3: s_mov; 4: s_endpgm
    const GuestCfg cfg =
        cfg_of({sopp(kScc0, 2), kSMovS0Zero, sopp(kBranch, 1), kSMovS0Zero, kEndpgm});
    ASSERT_EQ(cfg.blocks.size(), 4u);
    EXPECT_EQ(cfg.blocks[1].first, 1u);
    EXPECT_EQ(cfg.blocks[1].last, 2u);
    EXPECT_EQ(cfg.blocks[1].term, Terminator::Branch);
    EXPECT_EQ(cfg.blocks[1].succ[0], kNone);   // s_branch has no fallthrough
    EXPECT_EQ(cfg.blocks[1].succ[1], 3u);
    EXPECT_EQ(cfg.blocks[2].succ[0], 3u);
    EXPECT_EQ(cfg.preds[3], (std::vector<uint32_t>{1, 2}));
    EXPECT_EQ(cfg.idom[3], 0u);
    EXPECT_EQ(cfg.block_at_pc(3), 2u);
    EXPECT_EQ(cfg.block_at_pc(2), kNone);   // inside block 1
    ASSERT_EQ(cfg.rpo.size(), 4u);
    EXPECT_EQ(cfg.rpo.front(), 0u);
    EXPECT_EQ(cfg.rpo.back(), 3u);
}

TEST(IrGuestCfg, BackwardBranchIsABackEdge) {
    // 0: s_mov; 1: execnz -> 0; 2: s_endpgm
    const GuestCfg cfg = cfg_of({kSMovS0Zero, sopp(kExecnz, -2), kEndpgm});
    ASSERT_EQ(cfg.blocks.size(), 2u);
    EXPECT_EQ(cfg.blocks[0].term, Terminator::CondExecnz);
    EXPECT_EQ(cfg.blocks[0].succ[1], 0u);
    ASSERT_EQ(cfg.back_edges.size(), 1u);
    EXPECT_EQ(cfg.back_edges[0], std::make_pair(0u, 0u));
    EXPECT_EQ(cfg.preds[0], std::vector<uint32_t>{0});
}

TEST(IrGuestCfg, TrapEndsTheWaveAndLeavesTheRestUnreachable) {
    const GuestCfg cfg = cfg_of({kTrap, kSMovS0Zero, kEndpgm});
    ASSERT_EQ(cfg.blocks.size(), 2u);
    EXPECT_EQ(cfg.blocks[0].term, Terminator::Trap);
    EXPECT_EQ(cfg.blocks[0].succ[0], kNone);
    EXPECT_TRUE(cfg.reachable[0]);
    EXPECT_FALSE(cfg.reachable[1]);
    EXPECT_EQ(cfg.idom[1], kNone);
}

TEST(IrGuestCfg, IndirectJumpHasNoSuccessorAndACallReturns) {
    const GuestCfg jump = cfg_of({kSetpc, kSMovS0Zero, kEndpgm});
    ASSERT_EQ(jump.blocks.size(), 2u);
    EXPECT_EQ(jump.blocks[0].term, Terminator::IndirectUnknown);
    EXPECT_EQ(jump.blocks[0].succ[0], kNone);
    EXPECT_EQ(jump.blocks[0].succ[1], kNone);
    EXPECT_FALSE(jump.reachable[1]);
    for (uint32_t word : {call(1), kSwappc}) {
        const GuestCfg cfg = cfg_of({word, kSMovS0Zero, kEndpgm});
        ASSERT_EQ(cfg.blocks.size(), 2u);
        EXPECT_EQ(cfg.blocks[0].term, Terminator::IndirectUnknown);
        EXPECT_EQ(cfg.blocks[0].succ[0], 1u);   // the return point
        EXPECT_EQ(cfg.blocks[0].succ[1], kNone);
        EXPECT_TRUE(cfg.reachable[1]);
    }
}

TEST(IrGuestCfg, EndVariantEndsABlockMidStream) {
    const GuestCfg cfg = cfg_of({kEndpgmSaved, kSMovS0Zero, kEndpgm});
    ASSERT_EQ(cfg.blocks.size(), 2u);
    EXPECT_EQ(cfg.blocks[0].term, Terminator::End);
    EXPECT_EQ(cfg.blocks[1].term, Terminator::End);
}

TEST(IrGuestCfg, TargetOutsideTheStreamKeepsItsPc) {
    // 0: scc1 -> 3 (past s_endpgm at 2, so not a decoded instruction)
    const GuestCfg cfg = cfg_of({sopp(kScc1, 2), kSMovS0Zero, kEndpgm, kEndpgm});
    ASSERT_EQ(cfg.blocks.size(), 2u);
    EXPECT_EQ(cfg.blocks[0].succ[0], 1u);
    EXPECT_EQ(cfg.blocks[0].succ[1], kNone);
    EXPECT_EQ(cfg.blocks[0].taken_pc, 3u);
}

TEST(IrGuestCfg, StreamWithoutEndpgmFallsOffTheLastBlock) {
    const GuestCfg cfg = cfg_of({kSMovS0Zero, kVNop});
    ASSERT_EQ(cfg.blocks.size(), 1u);
    EXPECT_EQ(cfg.blocks[0].term, Terminator::Fallthrough);
    EXPECT_EQ(cfg.blocks[0].succ[0], kNone);
}

TEST(IrGuestCfg, BranchToTheNextInstructionHasOnePredecessorEdge) {
    const GuestCfg cfg = cfg_of({sopp(kVccz, 0), kEndpgm});
    ASSERT_EQ(cfg.blocks.size(), 2u);
    EXPECT_EQ(cfg.blocks[0].succ[0], 1u);
    EXPECT_EQ(cfg.blocks[0].succ[1], 1u);
    EXPECT_EQ(cfg.preds[1], std::vector<uint32_t>{0});
}

TEST(IrGuestCfg, ServiceSplitIsNotImplemented) {
    const auto ins = walk({kSMovS0Zero, kEndpgm});
    SplitPolicy policy;
    policy.split_at_services = true;
    EXPECT_TRUE(build_guest_cfg(ins, policy).blocks.empty());
    EXPECT_TRUE(build_guest_cfg({}, SplitPolicy{}).blocks.empty());
}

}   // namespace
