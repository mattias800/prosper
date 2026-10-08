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
    ASSERT_EQ(facts.ops.size(), 1u);
    EXPECT_FALSE(facts.ops[0].exec_full);
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
    // s_mov s20,0 ; loop: popcount ; s_cbranch_scc0 loop. A shape the region detectors do not
    // model proves nothing, whatever the workgroup size.
    const std::vector<uint32_t> code = {
        kMovS20Zero, kCmpEqVcc, kBcnt1Vcc, kCmpEq0S20, 0xbf84fffcu /* s_cbranch_scc0 -4 */, kEnd};
    const auto facts = analyze(code);
    ASSERT_FALSE(facts.ops.empty());
    const auto decision = select_compute_wave_route(facts, nvidia(64));
    EXPECT_NE(decision.route, ComputeWaveRoute::WorkgroupExchange) << decision.reason;
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
