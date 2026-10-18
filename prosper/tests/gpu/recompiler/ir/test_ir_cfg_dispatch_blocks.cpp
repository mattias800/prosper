// test_ir_cfg_dispatch_blocks -- IR_CONTROL_FLOW_PLAN step 5: build_guest_cfg's service split
// (SplitPolicy::split_at_services) and the dispatcher's view of it (dispatch_blocks.hpp).
//
// The synthetic cases pin each edge the plan flags: services, s_trap, linearized branches, the
// indirect and call forms, targets past s_endpgm and inside an instruction. The shadow case
// recompiles every committed tests/data fixture under each stage and requires every CFG dispatcher
// region to have taken its blocks from GuestCfg; the guard cases are the positive controls that a
// difference is seen and refused.
#include "gpu/recompiler/ir/cfg/dispatch_blocks.hpp"

#include "gpu/recompiler/float32_rte_support.hpp"
#include "gpu/recompiler/float64_support.hpp"
#include "gpu/recompiler/ir/cfg_fixtures.hpp"
#include "gpu/recompiler/ir/cfg_streams.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>

#include <unordered_set>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir::cfg;
using namespace prosper::test::cfg_streams;
using namespace prosper::test::cfg_fixtures;

namespace {

inline constexpr uint32_t kSMovS0Literal = 0xBE8003FFu;   // s_mov_b32 s0, <literal>

using Successors = std::vector<std::vector<uint32_t>>;

GuestCfg service_cfg(const std::vector<Rdna2Inst>& ins,
                     const std::unordered_set<uint32_t>& services = {},
                     const std::unordered_set<uint32_t>& linearized = {}) {
    SplitPolicy policy;
    policy.split_at_services = true;
    policy.services = &services;
    policy.linearized = &linearized;
    return build_guest_cfg(ins, policy);
}

std::vector<uint32_t> starts_of(const GuestCfg& cfg) {
    return dispatch_blocks(cfg).starts;
}

TEST(IrDispatchBlocks, AServiceIsABlockOfItsOwn) {
    const auto ins = walk({kSMovS0Zero, kVNop, kSMovS0Zero, kEndpgm});
    const GuestCfg cfg = service_cfg(ins, {1});
    EXPECT_EQ(starts_of(cfg), (std::vector<uint32_t>{0, 1, 2, 3}));
    EXPECT_EQ(cfg.blocks[1].service_event, 1u);
    EXPECT_EQ(cfg.blocks[1].term, Terminator::Fallthrough);
    EXPECT_EQ(cfg.blocks[0].service_event, kNone);
    EXPECT_EQ(dispatch_blocks(cfg).successors, (Successors{{1}, {2}, {3}, {}}));
    // The control-flow split ignores services.
    EXPECT_EQ(build_guest_cfg(ins, SplitPolicy{}).blocks.size(), 1u);
}

TEST(IrDispatchBlocks, TrapEndsTheWaveInABlockOfItsOwn) {
    const GuestCfg cfg = service_cfg(walk({kSMovS0Zero, kTrap, kSMovS0Zero, kEndpgm}));
    EXPECT_EQ(starts_of(cfg), (std::vector<uint32_t>{0, 1, 2, 3}));
    EXPECT_EQ(cfg.blocks[1].term, Terminator::Trap);
    EXPECT_EQ(dispatch_blocks(cfg).successors, (Successors{{1}, {}, {3}, {}}));
    EXPECT_FALSE(cfg.reachable[2]);
}

TEST(IrDispatchBlocks, ALinearizedBranchIsAnOrdinaryInstruction) {
    const auto ins = walk({sopp(kVccz, 1), kSMovS0Zero, kEndpgm});
    const GuestCfg linear = service_cfg(ins, {}, {0});
    EXPECT_EQ(starts_of(linear), (std::vector<uint32_t>{0, 2}));
    EXPECT_EQ(linear.blocks[0].term, Terminator::Fallthrough);
    const GuestCfg branch = service_cfg(ins);
    EXPECT_EQ(starts_of(branch), (std::vector<uint32_t>{0, 1, 2}));
    EXPECT_EQ(dispatch_blocks(branch).successors, (Successors{{2, 1}, {2}, {}}));
}

TEST(IrDispatchBlocks, IndirectCallAndOtherEndFormsStayInsideTheirBlock) {
    // The dispatcher's terminators are s_endpgm, s_trap and its branches only.
    const GuestCfg cfg = service_cfg(walk({kSetpc, kSwappc, kEndpgmSaved, kCodeEnd, kEndpgm}));
    EXPECT_EQ(starts_of(cfg), (std::vector<uint32_t>{0, 4}));
    EXPECT_EQ(cfg.blocks[0].term, Terminator::Fallthrough);
    EXPECT_EQ(cfg.blocks[1].term, Terminator::End);
}

TEST(IrDispatchBlocks, SuccessorsListTheTakenEdgeFirstWithoutRepeats) {
    EXPECT_EQ(dispatch_blocks(service_cfg(walk({sopp(kScc0, 1), kSMovS0Zero, kEndpgm}))).successors,
              (Successors{{2, 1}, {2}, {}}));
    EXPECT_EQ(dispatch_blocks(service_cfg(walk({sopp(kScc0, 0), kEndpgm}))).successors,
              (Successors{{1}, {}}));
    EXPECT_EQ(
        dispatch_blocks(service_cfg(walk({sopp(kBranch, 1), kSMovS0Zero, kEndpgm}))).successors,
        (Successors{{2}, {2}, {}}));
}

TEST(IrDispatchBlocks, CodeAfterTheFirstEndpgmIsOutsideTheGraph) {
    // The branch targets the s_mov after s_endpgm: no edge (a proven early exit is the per-case
    // emission's business), and no block there. rdna2_walk stops at s_endpgm, so the tail is
    // decoded separately and appended, as a caller's region may carry one.
    auto ins = walk({sopp(kVccz, 1), kEndpgm});
    for (Rdna2Inst in : walk({kSMovS0Zero, kEndpgm})) {
        in.pc += 2;
        ins.push_back(in);
    }
    const GuestCfg cfg = service_cfg(ins);
    EXPECT_EQ(starts_of(cfg), (std::vector<uint32_t>{0, 1}));
    EXPECT_EQ(cfg.blocks[0].succ[1], kNone);
    EXPECT_EQ(cfg.blocks[0].taken_pc, 2u);
    EXPECT_EQ(cfg.blocks.back().last, 1u);
    EXPECT_EQ(dispatch_blocks(cfg).successors, (Successors{{1}, {}}));
}

TEST(IrDispatchBlocks, NoEndpgmIsNoGraph) {
    EXPECT_TRUE(service_cfg(walk({kSMovS0Zero, kVNop})).blocks.empty());
    EXPECT_TRUE(service_cfg({}).blocks.empty());
}

// A target inside an instruction: the dispatcher starts a case at that pc, which no decoded
// instruction begins, so the graph has no block there. adopt_guest_cfg_blocks then keeps the
// dispatcher's own list (the guard), as the next test shows.
TEST(IrDispatchBlocks, ATargetInsideAnInstructionIsNotABlock) {
    const GuestCfg cfg = service_cfg(walk({sopp(kScc0, 1), kSMovS0Literal, 7u, kEndpgm}));
    EXPECT_EQ(starts_of(cfg), (std::vector<uint32_t>{0, 1, 3}));
    EXPECT_EQ(cfg.blocks[0].taken_pc, 2u);
    EXPECT_EQ(cfg.blocks[0].succ[1], kNone);
}

TEST(IrDispatchBlocks, TheGuardAdoptsOnlyAnIdenticalList) {
    const auto ins = walk({sopp(kScc0, 1), kSMovS0Literal, 7u, kEndpgm});
    SplitPolicy policy;
    policy.split_at_services = true;
    DispatchBlocksCapture capture;
    // The dispatcher's own answer for this program: a case at pc 2, inside the literal.
    std::vector<uint32_t> starts{0, 1, 2, 3};
    Successors successors{{2, 1}, {2}, {3}, {}};
    EXPECT_FALSE(adopt_guest_cfg_blocks(ins, policy, starts, successors));
    EXPECT_EQ(starts, (std::vector<uint32_t>{0, 1, 2, 3}));
    EXPECT_EQ(successors, (Successors{{2, 1}, {2}, {3}, {}}));
    // Its answer without the inner target, which the graph does reproduce.
    std::vector<uint32_t> equal_starts{0, 1, 3};
    Successors equal_successors{{1}, {2}, {}};
    EXPECT_TRUE(adopt_guest_cfg_blocks(ins, policy, equal_starts, equal_successors));
    ASSERT_EQ(capture.records().size(), 2u);
    EXPECT_FALSE(capture.records()[0].adopted);
    EXPECT_EQ(capture.records()[0].difference, "block 2 starts at pc 2, guest cfg pc 3");
    EXPECT_TRUE(capture.records()[1].adopted);
    EXPECT_EQ(capture.records()[1].blocks, 3u);
}

TEST(IrDispatchBlocks, TheDifferenceNamesTheFirstSuccessorMismatch) {
    const DispatchBlocks a{{0, 2}, {{1}, {}}};
    const DispatchBlocks b{{0, 2}, {{}, {}}};
    EXPECT_EQ(dispatch_blocks_difference(a, a), "");
    EXPECT_EQ(dispatch_blocks_difference(a, b), "block 0 (pc 0) successors [1], guest cfg []");
    EXPECT_EQ(dispatch_blocks_difference(a, DispatchBlocks{{0}, {{}}}), "blocks 2, guest cfg 1");
}

// The same shape through the real dispatcher: an irreducible loop (block 2 is entered from the
// entry as well as from its header) sends the program to the CFG dispatcher, and the VCCZ branch
// targets pc 6, the literal of the s_mov at pc 5. The dispatcher starts a case there; the graph
// cannot, so the guard keeps the dispatcher's own blocks and the recompile is unchanged.
TEST(IrDispatchBlocks, TheDispatcherKeepsItsOwnBlocksForATargetInsideALiteral) {
    const std::vector<uint32_t> words{sopp(kScc1, 1), kSMovS0Zero,    kSMovS0Zero, sopp(kScc0, -3),
                                      sopp(kVccz, 1), kSMovS0Literal, 7u,          kEndpgm};
    DispatchBlocksCapture capture;
    const auto spirv =
        recompile_compute(words.data(), words.size(), nullptr, ComputeShaderConfig{});
    EXPECT_FALSE(spirv.empty());
    ASSERT_EQ(capture.records().size(), 1u);
    EXPECT_FALSE(capture.records()[0].adopted);
    EXPECT_EQ(capture.records()[0].blocks, 7u);
    EXPECT_EQ(capture.records()[0].cfg_blocks, 6u);
    EXPECT_EQ(capture.records()[0].difference, "block 5 starts at pc 6, guest cfg pc 7");
}

// Every committed fixture, recompiled the way shader_inspect --stage does. Each CFG dispatcher
// region must have adopted the GuestCfg's blocks.
TEST(IrDispatchBlocks, EveryCommittedFixtureDispatcherRegionAdoptsTheGuestCfg) {
    publish_float64_support(true, true, true);
    publish_float32_rte_support(true, true, true);
    const auto fixtures = guest_program_fixtures(__FILE__);
    ASSERT_GE(fixtures.size(), 9u) << "tests/data lost guest-program fixtures";
    const ComputeShaderConfig compute{};
    size_t regions = 0, services = 0;
    for (const auto& path : fixtures) {
        const std::vector<uint32_t> words = load_words(path);
        ASSERT_FALSE(words.empty()) << path;
        for (int stage = 0; stage < 3; ++stage) {
            DispatchBlocksCapture capture;
            if (stage == 0) recompile_compute(words.data(), words.size(), nullptr, compute);
            if (stage == 1) recompile_fragment(words.data(), words.size());
            if (stage == 2) recompile_vertex(words.data(), words.size());
            for (const auto& record : capture.records()) {
                EXPECT_TRUE(record.adopted) << path.filename() << " stage " << stage << " entry "
                                            << record.entry_pc << ": " << record.difference;
                ++regions;
                services += record.services;
            }
        }
    }
    // Positive control: the fixtures reach the dispatcher, services included.
    EXPECT_GE(regions, 8u);
    EXPECT_GT(services, 0u);
}

}   // namespace
