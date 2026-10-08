// ADR 0028 migration step 3, analysis half: classify every cross-lane operation of a guest compute
// program and decide which route it may take on a host whose compute subgroup is narrower than the
// guest wave. CPU-only; nothing here dispatches.
//
// The programs are hand-assembled RDNA2 words (encodings cross-checked against the fixtures in
// test_rdna2_to_spirv.cpp and test_workgroup_uniformity.cpp), so the analysis is not graded on
// instructions it assembled itself. Route-selection arms that need a context the decoder cannot
// reach cheaply (a loop, a launch constraint) build the facts by hand instead.
#include "gpu/recompiler/compute_wave_route.hpp"

#include <gtest/gtest.h>

#include <vector>

#include "gpu/recompiler/rdna2_decode.hpp"

using namespace prosper::gpu;

namespace {

constexpr uint32_t kEnd = 0xbf810000u;
constexpr uint32_t kCmpEqVcc = 0x7d840100u;        // v_cmp_eq_u32 vcc, v0, v0 (every lane)
constexpr uint32_t kBcnt1Vcc = 0xbe84106au;        // s_bcnt1_i32_b64 s4, vcc
constexpr uint32_t kBcnt1Exec = 0xbe84107eu;       // s_bcnt1_i32_b64 s4, exec
constexpr uint32_t kReadFirstLaneS20 =
    0x7e280500u;  // v_readfirstlane_b32 s20, v0 (per-wave value)
constexpr uint32_t kCmpEq0S20 = 0xbf061480u;       // s_cmp_eq_u32 0, s20
constexpr uint32_t kMovS20Zero = 0xbe940380u;      // s_mov_b32 s20, 0 (uniform constant)
constexpr uint32_t kBranchScc1Over2 = 0xbf850002u; // s_cbranch_scc1 +2

ComputeWaveOpFacts analyze(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> ins;
    EXPECT_EQ(rdna2_walk(code.data(), code.size(), ins), code.size())
        << "fixture must decode fully";
    return analyze_compute_wave_ops(ins, code.data(), code.size());
}

ComputeWaveHost nvidia(uint32_t local_x = 64) {
    ComputeWaveHost host;
    host.guest_wave = 64;
    host.host_subgroup_min = host.host_subgroup_max = 32;
    host.local_x = local_x;
    host.max_shared_bytes = 32768;
    host.max_workgroup_invocations = 1024;
    return host;
}

// A ballot popcount of the whole wave, guarded by a branch on `condition_words`. The guard is
// `s_cbranch_scc1 +2`, so the two instructions it skips (the compare and the popcount) are the
// region.
std::vector<uint32_t> guarded_ballot(uint32_t scalar_definition) {
    return {scalar_definition, kCmpEq0S20, kBranchScc1Over2, kCmpEqVcc, kBcnt1Vcc, kEnd};
}

}  // namespace

TEST(ComputeWaveRoute, ProgramWithoutCrossLaneOperationIsWidthIndependent) {
    const auto facts = analyze({0x7e020287u /* v_mov_b32 v1, 7 */, kEnd});
    ASSERT_TRUE(facts.analyzed);
    EXPECT_TRUE(facts.ops.empty());
    const auto decision = select_compute_wave_route(facts, nvidia());
    EXPECT_EQ(decision.route, ComputeWaveRoute::WidthIndependent);
    EXPECT_STREQ(decision.reason, "no-cross-lane-op");
}

TEST(ComputeWaveRoute, UniformBallotIsExchangeAdmissible) {
    // v_cmp vcc ; s_bcnt1 s4, vcc ; s_endpgm: a ballot popcount at the top level, EXEC untouched.
    const auto facts = analyze({kCmpEqVcc, kBcnt1Vcc, kEnd});
    ASSERT_EQ(facts.ops.size(), 1u);
    EXPECT_EQ(facts.ops[0].kind, ComputeCrossLaneKind::Ballot);
    EXPECT_EQ(facts.ops[0].context, ComputeWaveContext::TopLevel);
    EXPECT_TRUE(facts.ops[0].exec_full);
    EXPECT_EQ(facts.count[static_cast<size_t>(ComputeCrossLaneKind::Ballot)], 1u);

    const auto decision = select_compute_wave_route(facts, nvidia());
    EXPECT_EQ(decision.route, ComputeWaveRoute::WorkgroupExchange);
    EXPECT_STREQ(decision.reason, "cross-lane-in-uniform-flow");
    EXPECT_EQ(decision.cross_lane_ops, 1u);
    EXPECT_EQ(decision.exchange_scratch_bytes, 528u)
        << "two 64-lane planes + wave result + 3 control slots";
}

TEST(ComputeWaveRoute, ReadLaneAndMbcntAreInventoried) {
    const auto facts = analyze({0xd7600006u, 0x00010b1fu /* v_readlane s6, v31, 5 */, 0xd7650001u,
                                0x0001007eu /* v_mbcnt_lo v1, exec_lo, 0 */, kEnd});
    ASSERT_EQ(facts.ops.size(), 2u);
    EXPECT_EQ(facts.ops[0].kind, ComputeCrossLaneKind::ReadLane);
    EXPECT_EQ(facts.ops[1].kind, ComputeCrossLaneKind::Mbcnt);
    EXPECT_EQ(select_compute_wave_route(facts, nvidia()).route,
              ComputeWaveRoute::WorkgroupExchange);
}

TEST(ComputeWaveRoute, MbcntOverAllOnesIsALaneIndexNotACrossLaneOperation) {
    // `v_mbcnt_lo_u32_b32 v2, -1, 0` is the lane id: exact one lane per invocation.
    const auto facts = analyze({0xd7650002u, 0x000100c1u, kEnd});
    EXPECT_TRUE(facts.ops.empty());
}

TEST(ComputeWaveRoute, ReadFirstLaneIsInventoried) {
    const auto facts = analyze({kReadFirstLaneS20, kEnd});
    ASSERT_EQ(facts.ops.size(), 1u);
    EXPECT_EQ(facts.ops[0].kind, ComputeCrossLaneKind::ReadFirstLane);
}

TEST(ComputeWaveRoute, ExecWrittenBeforeAnOperationIsRecorded) {
    // s_and_saveexec_b64 s[8:9], vcc narrows EXEC; the popcount after it no longer has a full EXEC.
    const auto facts =
        analyze({kCmpEqVcc, 0xbe88246au /* s_and_saveexec_b64 s[8:9], vcc */, kBcnt1Exec, kEnd});
    ASSERT_EQ(facts.ops.size(), 2u) << "the saveexec is itself a mask-SCC site";
    EXPECT_EQ(facts.ops[1].kind, ComputeCrossLaneKind::Ballot);
    EXPECT_FALSE(facts.ops[1].exec_full);
}

TEST(ComputeWaveRoute, BallotUnderAProvenUniformBranchStaysAdmissibleInAMultiWaveWorkgroup) {
    // Positive arm for the mutation below: the branch condition is built from a constant, so every
    // wave of the workgroup decides it identically and the region cannot make a barrier divergent.
    const auto facts = analyze(guarded_ballot(kMovS20Zero));
    ASSERT_EQ(facts.ops.size(), 1u);
    EXPECT_EQ(facts.ops[0].context, ComputeWaveContext::UniformRegion);
    const auto decision = select_compute_wave_route(facts, nvidia(128));
    EXPECT_EQ(decision.route, ComputeWaveRoute::WorkgroupExchange) << decision.reason;
}

TEST(ComputeWaveRoute, MutationBallotUnderADivergentBranchIsNotExchangeAdmissible) {
    // Same ballot, same region; only the branch condition changes to a per-wave value
    // (v_readfirstlane of lane data differs between the waves of a workgroup). A scratch vote's
    // barriers would then be reached by some waves and not others, so exchange is NOT exact.
    const auto facts = analyze(guarded_ballot(kReadFirstLaneS20));
    ASSERT_EQ(facts.ops.size(), 2u)
        << "the readfirstlane that feeds the branch is itself an operation";
    const auto ballot = facts.ops[1];
    EXPECT_EQ(ballot.kind, ComputeCrossLaneKind::Ballot);
    EXPECT_EQ(ballot.context, ComputeWaveContext::UnprovenRegion);

    const auto decision = select_compute_wave_route(facts, nvidia(128));
    EXPECT_EQ(decision.route, ComputeWaveRoute::NeedsNLanes);
    EXPECT_STREQ(decision.reason, "cross-lane-in-nonuniform-region");
    EXPECT_EQ(decision.blocker_pc, ballot.pc);
    EXPECT_EQ(decision.blocker_kind, ComputeCrossLaneKind::Ballot);
}

TEST(ComputeWaveRoute, SingleWaveWorkgroupMakesEveryScalarBranchWorkgroupUniform) {
    // The identical divergent-looking program in a 64-invocation workgroup has one guest wave, and
    // a scalar branch cannot split one wave. Only the launch shape differs from the mutation arm.
    const auto facts = analyze(guarded_ballot(kReadFirstLaneS20));
    EXPECT_EQ(select_compute_wave_route(facts, nvidia(64)).route,
              ComputeWaveRoute::WorkgroupExchange);
}

TEST(ComputeWaveRoute, BackwardBranchNoLoopModelCoversIsRefusedNotAdmitted) {
    // An unconditional backward s_branch around a ballot: no loop shape covers it, so nothing can be
    // proved about where the operation sits, whatever the workgroup size.
    const std::vector<uint32_t> code = {kCmpEqVcc, kBcnt1Vcc, 0xbf82fffdu /* s_branch -3 */, kEnd};
    const auto facts = analyze(code);
    ASSERT_FALSE(facts.ops.empty());
    EXPECT_TRUE(facts.control_flow_unmodelled);
    const auto decision = select_compute_wave_route(facts, nvidia(64));
    EXPECT_EQ(decision.route, ComputeWaveRoute::Refused);
    EXPECT_STREQ(decision.reason, "control-flow-unmodelled");
}

// ---- hand-built facts: contexts and constraints constructed outside the analysis ----

namespace {
ComputeWaveOpFacts one_op(ComputeCrossLaneKind kind, ComputeWaveContext context,
                          uint32_t native_lanes = 0) {
    ComputeWaveOpFacts facts;
    facts.analyzed = true;
    ComputeCrossLaneOp op;
    op.pc = 7;
    op.kind = kind;
    op.context = context;
    op.native_lanes = native_lanes;
    facts.ops.push_back(op);
    ++facts.count[static_cast<size_t>(kind)];
    return facts;
}
}   // namespace

TEST(ComputeWaveRoute, LoopedCrossLaneOperationNeedsNLanes) {
    const auto decision = select_compute_wave_route(
        one_op(ComputeCrossLaneKind::ReadLane, ComputeWaveContext::Loop), nvidia(64));
    EXPECT_EQ(decision.route, ComputeWaveRoute::NeedsNLanes);
    EXPECT_STREQ(decision.reason, "cross-lane-in-loop");
    EXPECT_EQ(decision.blocker_pc, 7u);
}

TEST(ComputeWaveRoute, NativeHostNeverAnalyses) {
    ComputeWaveHost host = nvidia();
    host.host_subgroup_min = host.host_subgroup_max = 64;
    EXPECT_EQ(select_compute_wave_route(
                  one_op(ComputeCrossLaneKind::ReadLane, ComputeWaveContext::Loop), host)
                  .route,
              ComputeWaveRoute::Native);
    host = nvidia();
    host.native_contract = true;
    EXPECT_EQ(select_compute_wave_route(
                  one_op(ComputeCrossLaneKind::Ballot, ComputeWaveContext::Loop), host)
                  .route,
              ComputeWaveRoute::Native);
}

TEST(ComputeWaveRoute, NoFactsIsUnanalyzedNeverWidthIndependent) {
    const auto decision = select_compute_wave_route(ComputeWaveOpFacts{}, nvidia());
    EXPECT_EQ(decision.route, ComputeWaveRoute::Refused);
    EXPECT_STREQ(decision.reason, "unanalyzed");
}

TEST(ComputeWaveRoute, WorkgroupMustHoldWholeGuestWaves) {
    const auto facts = one_op(ComputeCrossLaneKind::Ballot, ComputeWaveContext::TopLevel);
    EXPECT_STREQ(select_compute_wave_route(facts, nvidia(96)).reason,
                 "workgroup-not-guest-wave-multiple");
    EXPECT_STREQ(select_compute_wave_route(facts, nvidia(32)).reason,
                 "workgroup-not-guest-wave-multiple");
    EXPECT_EQ(select_compute_wave_route(facts, nvidia(256)).route,
              ComputeWaveRoute::WorkgroupExchange);
}

TEST(ComputeWaveRoute, SubgroupMustDivideTheGuestWave) {
    ComputeWaveHost host = nvidia();
    host.host_subgroup_min = host.host_subgroup_max = 24;
    EXPECT_STREQ(select_compute_wave_route(
                     one_op(ComputeCrossLaneKind::Ballot, ComputeWaveContext::TopLevel), host)
                     .reason,
                 "subgroup-does-not-divide-guest-wave");
}

TEST(ComputeWaveRoute, SharedMemoryBudgetIncludesTheGuestsOwnLds) {
    const auto facts = one_op(ComputeCrossLaneKind::Ballot, ComputeWaveContext::TopLevel);
    ComputeWaveHost host = nvidia();
    host.max_shared_bytes = 16384;
    host.guest_lds_bytes = 16384 - 528;   // exactly fits the 528-byte upper bound for one wave
    EXPECT_EQ(select_compute_wave_route(facts, host).route, ComputeWaveRoute::WorkgroupExchange);
    host.guest_lds_bytes += 4;
    const auto over = select_compute_wave_route(facts, host);
    EXPECT_EQ(over.route, ComputeWaveRoute::Refused);
    EXPECT_STREQ(over.reason, "shared-memory-budget");
    host.max_shared_bytes = 0;
    EXPECT_STREQ(select_compute_wave_route(facts, host).reason, "shared-memory-limit-unknown");
}

TEST(ComputeWaveRoute, ExchangeScratchIsTheUpperBoundOverBothLoweringFamilies) {
    // Dispatcher: two per-lane planes over the padded wave count + one result slot per wave + 3.
    // Scratch votes: one slot per invocation + one result slot per wave. The bound is the larger.
    EXPECT_EQ(compute_exchange_scratch_bytes(64, 64), (2u * 64u + 1u + 3u) * 4u);
    EXPECT_EQ(compute_exchange_scratch_bytes(128, 64), (2u * 128u + 2u + 3u) * 4u);
    EXPECT_EQ(compute_exchange_scratch_bytes(96, 64), (2u * 128u + 2u + 3u) * 4u)
        << "a partial last wave pads";
    EXPECT_EQ(compute_exchange_scratch_bytes(64, 32), (2u * 64u + 2u + 3u) * 4u);
    EXPECT_GE(compute_exchange_scratch_bytes(64, 64), (64u + 1u) * 4u)
        << "never below the vote helpers' array";
}

TEST(ComputeWaveRoute, PartialWorkgroupAndDeviceLimitRefuse) {
    const auto facts = one_op(ComputeCrossLaneKind::Ballot, ComputeWaveContext::TopLevel);
    ComputeWaveHost host = nvidia();
    host.partial_workgroup = true;
    EXPECT_STREQ(select_compute_wave_route(facts, host).reason, "partial-workgroup-barrier");
    host = nvidia(2048);
    EXPECT_STREQ(select_compute_wave_route(facts, host).reason, "workgroup-exceeds-device-limit");
}

TEST(ComputeWaveRoute, ShuffleOnlyOperationsNeedASubgroupThatHoldsTheirDomain) {
    // A quad shuffle (4 lanes) is exact inside a 32-lane subgroup; a row shuffle (16) is not
    // inside an 8-lane one, and there is no exchange form to fall back on: visible refusal.
    const auto quad = one_op(ComputeCrossLaneKind::Dpp, ComputeWaveContext::TopLevel, 4);
    const auto row = one_op(ComputeCrossLaneKind::Dpp, ComputeWaveContext::TopLevel, 16);
    EXPECT_EQ(select_compute_wave_route(quad, nvidia()).route, ComputeWaveRoute::Native);
    EXPECT_STREQ(select_compute_wave_route(quad, nvidia()).reason,
                 "cross-lane-within-host-subgroup");
    ComputeWaveHost lavapipe = nvidia();
    lavapipe.host_subgroup_min = lavapipe.host_subgroup_max = 8;
    const auto refused = select_compute_wave_route(row, lavapipe);
    EXPECT_EQ(refused.route, ComputeWaveRoute::Refused);
    EXPECT_STREQ(refused.reason, "exchange-lowering-unavailable");
    EXPECT_EQ(refused.blocker_kind, ComputeCrossLaneKind::Dpp);
    // permlanex16 needs a 32-lane row pair: fine on 32, refused on 16.
    const auto x16 = one_op(ComputeCrossLaneKind::PermLane, ComputeWaveContext::TopLevel, 32);
    EXPECT_EQ(select_compute_wave_route(x16, nvidia()).route, ComputeWaveRoute::Native);
    lavapipe.host_subgroup_min = lavapipe.host_subgroup_max = 16;
    EXPECT_EQ(select_compute_wave_route(x16, lavapipe).route, ComputeWaveRoute::Refused);
}

TEST(ComputeWaveRoute, WaveCollectiveCounterHasNoExchangeForm) {
    const auto decision = select_compute_wave_route(
        one_op(ComputeCrossLaneKind::DsAppend, ComputeWaveContext::TopLevel), nvidia());
    EXPECT_EQ(decision.route, ComputeWaveRoute::Refused);
    EXPECT_STREQ(decision.reason, "exchange-lowering-unavailable");
}

TEST(ComputeWaveRoute, PermLaneIsRecognisedFromEncodings) {
    // v_permlane16 / permlanex16 (VOP3 0x377 / 0x378): operands are irrelevant to the inventory.
    const auto facts = analyze({0xd7770001u, 0x00000100u, 0xd7780002u, 0x00000100u, kEnd});
    ASSERT_EQ(facts.ops.size(), 2u);
    EXPECT_EQ(facts.ops[0].kind, ComputeCrossLaneKind::PermLane);
    EXPECT_EQ(facts.ops[0].native_lanes, 16u);
    EXPECT_EQ(facts.ops[1].native_lanes, 32u);
}

TEST(ComputeWaveRoute, LaunchRefusalIsTheSameFunctionTheBackendApplies) {
    ComputeWaveHost host = nvidia();
    EXPECT_EQ(compute_exchange_launch_refusal(host), nullptr);
    host.partial_workgroup = true;
    EXPECT_STREQ(compute_exchange_launch_refusal(host), "partial-workgroup-barrier");
    host = nvidia(192);
    host.max_workgroup_invocations = 128;
    EXPECT_STREQ(compute_exchange_launch_refusal(host), "workgroup-exceeds-device-limit");
    host = nvidia();
    host.guest_lds_bytes = 32768;
    EXPECT_STREQ(compute_exchange_launch_refusal(host), "shared-memory-budget");
}

// ---- decoded arms for every inventory path (each is red if its detection is deleted) ----

namespace {
bool has_kind(const ComputeWaveOpFacts& f, ComputeCrossLaneKind k) {
    return f.count[static_cast<size_t>(k)] != 0;
}
}   // namespace

TEST(ComputeWaveRoute, AWholeWaveSccVoteIsNotWidthIndependent) {
    // v_cmp vcc ; s_cmp_lg_u64 vcc, 0 ; s_cbranch_scc1 +1: SCC reads the whole wave's mask.
    const auto facts = analyze({kCmpEqVcc, 0xbf13806au, 0xbf850001u, 0x7e020287u, kEnd});
    EXPECT_TRUE(has_kind(facts, ComputeCrossLaneKind::MaskScc));
    EXPECT_NE(select_compute_wave_route(facts, nvidia()).route, ComputeWaveRoute::WidthIndependent);
}

TEST(ComputeWaveRoute, ASaveExecSccVoteIsNotWidthIndependent) {
    const auto facts = analyze({kCmpEqVcc, 0xbe88246au /* s_and_saveexec_b64 s[8:9], vcc */,
                                0xbf840001u, 0x7e020287u, 0xbefe0408u, kEnd});
    EXPECT_TRUE(has_kind(facts, ComputeCrossLaneKind::MaskScc));
}

TEST(ComputeWaveRoute, ASixtyFourBitMaskAndIsAMaskScc) {
    // s_and_b64 s[4:5], vcc, exec -> SCC = (result != 0)
    const auto facts = analyze({kCmpEqVcc, 0x87847e6au, kEnd});
    EXPECT_TRUE(has_kind(facts, ComputeCrossLaneKind::MaskScc));
}

TEST(ComputeWaveRoute, WriteLaneWithADynamicSelectorIsInventoriedAndRefused) {
    const auto dynamic = analyze({0xd7610001u, 0x00000602u /* v_writelane v1, s2, s3 */, kEnd});
    ASSERT_TRUE(has_kind(dynamic, ComputeCrossLaneKind::WriteLane));
    const auto decision = select_compute_wave_route(dynamic, nvidia());
    EXPECT_EQ(decision.route, ComputeWaveRoute::Refused);
    EXPECT_STREQ(decision.reason, "exchange-lowering-unavailable");
    // The inline-selector form is the lane-slot spill and crosses no lane.
    EXPECT_FALSE(has_kind(analyze({0xd7610001u, 0x00010a02u /* v_writelane v1, s2, 5 */, kEnd}),
                          ComputeCrossLaneKind::WriteLane));
}

TEST(ComputeWaveRoute, WaveSynchronousLdsIsNamedNotWidthIndependent) {
    // ds_write_b32 v0, v1 ; ds_read_b32 v2, v0 with no s_barrier between.
    const std::vector<uint32_t> synchronous = {0xd8340000u, 0x00000100u, 0xd8d80000u, 0x02000000u,
                                               kEnd};
    const auto facts = analyze(synchronous);
    ASSERT_TRUE(has_kind(facts, ComputeCrossLaneKind::LdsWaveSync));
    const auto decision = select_compute_wave_route(facts, nvidia());
    EXPECT_EQ(decision.route, ComputeWaveRoute::Refused);
    EXPECT_EQ(decision.blocker_kind, ComputeCrossLaneKind::LdsWaveSync);
    // A barrier between the store and the load makes it ordinary workgroup LDS.
    const std::vector<uint32_t> barriered = {0xd8340000u, 0x00000100u, 0xbf8a0000u,
                                             0xd8d80000u, 0x02000000u, kEnd};
    EXPECT_FALSE(has_kind(analyze(barriered), ComputeCrossLaneKind::LdsWaveSync));
}

TEST(ComputeWaveRoute, EveryDecodedKindIsInventoried) {
    EXPECT_TRUE(
        has_kind(analyze({kCmpEqVcc, 0xbf860001u /* s_cbranch_vccz +1 */, 0x7e020287u, kEnd}),
                 ComputeCrossLaneKind::WaveVote));
    EXPECT_TRUE(has_kind(analyze({0xbe841006u /* s_bcnt1_i32_b64 s4, s[6:7] */, kEnd}),
                         ComputeCrossLaneKind::MaskConsumer));
    EXPECT_TRUE(has_kind(analyze({0xdacc0000u, 0x02000100u /* ds_bpermute_b32 */, kEnd}),
                         ComputeCrossLaneKind::DsBpermute));
    EXPECT_TRUE(has_kind(analyze({0xd8f40000u, 0x01000000u /* ds_append */, kEnd}),
                         ComputeCrossLaneKind::DsAppend));
    EXPECT_TRUE(has_kind(analyze({0xdac80000u, 0x02000100u /* ds_permute_b32 */, kEnd}),
                         ComputeCrossLaneKind::DsPermute));
    const auto quad = analyze({0xd8d48000u, 0x02000000u /* ds_swizzle quad form */, kEnd});
    ASSERT_TRUE(has_kind(quad, ComputeCrossLaneKind::DsSwizzle));
    EXPECT_EQ(quad.ops[0].native_lanes, 4u);
    const auto group = analyze({0xd8d4001fu, 0x02000000u /* ds_swizzle group32 form */, kEnd});
    ASSERT_TRUE(has_kind(group, ComputeCrossLaneKind::DsSwizzle));
    EXPECT_EQ(group.ops[0].native_lanes, 32u);
}

TEST(ComputeWaveRoute, DppOperationsReportTheirShuffleDomain) {
    const auto quad = analyze({0x7e0202fau, 0xff00e400u /* v_mov_b32 v1, v0 quad_perm */, kEnd});
    ASSERT_TRUE(has_kind(quad, ComputeCrossLaneKind::Dpp));
    EXPECT_EQ(quad.ops[0].native_lanes, 4u);
    const auto row = analyze({0x7e0202fau, 0xff011100u /* v_mov_b32 v1, v0 row_shr:1 */, kEnd});
    ASSERT_TRUE(has_kind(row, ComputeCrossLaneKind::Dpp));
    EXPECT_EQ(row.ops[0].native_lanes, 16u);
}

TEST(ComputeWaveRoute, ADecodedCountedLoopPutsItsOperationInTheLoopContext) {
    // s_mov s0,4 ; loop: v_readlane s6,v1,5 ; s_sub s0,s0,1 ; s_cmp_lg s0,0 ; s_cbranch_scc1 loop
    const auto facts = analyze(
        {0xbe800384u, 0xd7600006u, 0x00010b01u, 0x80808100u, 0xbf078000u, 0xbf85fffbu, kEnd});
    ASSERT_EQ(facts.ops.size(), 1u);
    EXPECT_EQ(facts.ops[0].context, ComputeWaveContext::Loop);
    EXPECT_EQ(select_compute_wave_route(facts, nvidia()).route, ComputeWaveRoute::NeedsNLanes);
}

TEST(ComputeWaveRoute, AWaterfallLoopIsALoopEvenBesideARecognisedLoop) {
    // A counted loop, then a waterfall (readfirstlane ; s_andn2_b64 s[6:7],s[6:7],exec ; scc1 back).
    // Its trip count is per wave, so a barrier inside it diverges in a multi-wave workgroup.
    const std::vector<uint32_t> program = {0xbe800383u, 0x80808100u, 0xbf078000u,
                                           0xbf85fffdu,   // counted loop, 3 trips
                                           0x7e080500u, 0x8a867e06u, 0xbf85fffdu,   // waterfall
                                           kEnd};
    const auto facts = analyze(program);
    const ComputeCrossLaneOp* waterfall_op = nullptr;
    for (const auto& op : facts.ops)
        if (op.kind == ComputeCrossLaneKind::ReadFirstLane) waterfall_op = &op;
    ASSERT_NE(waterfall_op, nullptr);
    EXPECT_EQ(waterfall_op->context, ComputeWaveContext::Loop);
    EXPECT_NE(select_compute_wave_route(facts, nvidia(128)).route,
              ComputeWaveRoute::WorkgroupExchange);
}

TEST(ComputeWaveRoute, ABallotInTheElseArmOfADivergentBranchIsUnproven) {
    // cond = per-wave value ; scc0 -> else. then: v_mov ; s_branch merge. else: ballot. merge: end.
    const std::vector<uint32_t> program = {kReadFirstLaneS20,
                                           kCmpEq0S20,
                                           0xbf840002u /* s_cbranch_scc0 else (+2) */,
                                           0x7e020287u /* then: v_mov v1, 7 */,
                                           0xbf820002u /* s_branch merge (+2) */,
                                           kCmpEqVcc,
                                           kBcnt1Vcc,
                                           /* else */ kEnd};
    const auto facts = analyze(program);
    const ComputeCrossLaneOp* ballot = nullptr;
    for (const auto& op : facts.ops)
        if (op.kind == ComputeCrossLaneKind::Ballot) ballot = &op;
    ASSERT_NE(ballot, nullptr);
    EXPECT_EQ(ballot->context, ComputeWaveContext::UnprovenRegion)
        << "the else arm is inside the region";
}
