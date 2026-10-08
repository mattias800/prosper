// ADR 0028: the frontend glue that names a Wave64 route CANDIDATE on a refusal line. A stand-in
// context supplies the fields live_compute.cpp's real one has, so nothing here needs a Vulkan device.
#include <gtest/gtest.h>

#include <cstdio>
#include <vector>

#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/compute_wave_route.hpp"
#include "fixtures/wave64_exchange_fixture.hpp"
#include "shared/live/compute_wave_admission.hpp"

using namespace prosper::frontend;
using namespace prosper::gpu;
using prosper::diagnostics::perf::Wave64Candidate;

namespace {

struct FakeContext {
    uint32_t min_native_subgroup_size = 32, max_native_subgroup_size = 32, subgroup_size = 32;
    bool native_subgroup_contract = true;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
};

ComputeItem plain_item() {
    ComputeItem item;
    item.recompile_config_available = true;
    item.recompile_config.local_x = 64;
    item.recompile_config.wave_size = 64;
    item.recompile_config.threads_x = 64;
    return item;
}

ComputeWaveOpFacts one(ComputeCrossLaneKind kind, ComputeWaveContext context) {
    ComputeWaveOpFacts facts;
    facts.analyzed = true;
    facts.ops.push_back({3, kind, context, true, 0});
    return facts;
}

}   // namespace

TEST(ComputeWaveAdmission, TheCandidateNamesTheVocabularyRouteAndTheReason) {
    const auto item = plain_item();
    const auto loop = compute_wave_candidate(
        FakeContext{}, item, one(ComputeCrossLaneKind::ReadLane, ComputeWaveContext::Loop),
        {32768, 1024});
    EXPECT_STREQ(loop.route, "n-lanes");
    EXPECT_STREQ(loop.reason, "cross-lane-in-loop");
    const auto top = compute_wave_candidate(
        FakeContext{}, item, one(ComputeCrossLaneKind::Ballot, ComputeWaveContext::TopLevel),
        {32768, 1024});
    EXPECT_STREQ(top.route, "workgroup-exchange");
    // Unknown device limits are refused, not assumed.
    const auto unknown = compute_wave_candidate(
        FakeContext{}, item, one(ComputeCrossLaneKind::Ballot, ComputeWaveContext::TopLevel),
        {0, 0});
    EXPECT_STREQ(unknown.route, "refused");
    EXPECT_STREQ(unknown.reason, "shared-memory-limit-unknown");
}

TEST(ComputeWaveAdmission, ANativeCandidateIsNeverPrintedOnADeclinedDispatch) {
    FakeContext ctx;
    ctx.min_native_subgroup_size = ctx.max_native_subgroup_size = 64;
    auto item = plain_item();
    item.required_subgroup_size = 64;
    const auto out = compute_wave_candidate(
        ctx, item, one(ComputeCrossLaneKind::ReadLane, ComputeWaveContext::Loop), {32768, 1024});
    EXPECT_STREQ(out.route, "") << "`native` beside `dispatch skipped` would contradict itself";
    EXPECT_STREQ(out.reason, "");
}

TEST(ComputeWaveAdmission, APartialWorkgroupIsRefused) {
    auto item = plain_item();
    item.recompile_config.exact_thread_extent = true;
    item.recompile_config.threads_x = 40;
    item.recompile_config.threads_y = item.recompile_config.threads_z = 1;
    const auto out = compute_wave_candidate(
        FakeContext{}, item, one(ComputeCrossLaneKind::Ballot, ComputeWaveContext::TopLevel),
        {32768, 1024});
    EXPECT_STREQ(out.reason, "partial-workgroup-barrier");
}

TEST(ComputeWaveAdmission, TheThunkAnalysesTheProgramFromItsGuestAddress) {
    // The callback note_unsupported_wave64 runs after its dedupe: a ballot popcount at the top level.
    // The physical device is null, so the device limits are unknown and the candidate says so.
    static const std::vector<uint32_t> code{0x7d840100u, 0xbe84106au, 0xbf810000u};
    auto item = plain_item();
    item.code_addr = reinterpret_cast<uintptr_t>(code.data());
    item.code_dwords = static_cast<uint32_t>(code.size());
    const FakeContext ctx;
    const ComputeWaveCandidateArg<FakeContext> arg{&ctx, &item};
    const auto out = compute_wave_candidate_thunk<FakeContext>(&arg);
    EXPECT_STREQ(out.route, "refused");
    EXPECT_STREQ(out.reason, "shared-memory-limit-unknown");
}

TEST(ComputeWaveAdmission, AnItemWithoutAProgramLengthIsUnanalyzedNotWidthIndependent) {
    const auto item = plain_item();
    const FakeContext ctx;
    const ComputeWaveCandidateArg<FakeContext> arg{&ctx, &item};
    const auto out = compute_wave_candidate_thunk<FakeContext>(&arg);
    EXPECT_STREQ(out.route, "refused");
    EXPECT_STREQ(out.reason, "unanalyzed");
}

// ---- the exchange admission: each refusal names the condition that fired ----

namespace {
namespace fx = prosper::test::wave64_exchange;

ComputeItem exchange_item(const fx::Case& c, ComputeWaveOpFacts& facts) {
    ComputeItem item = plain_item();
    item.spirv = fx::compile(c, 32);
    item.recompile_config.local_x = c.local;
    item.recompile_config.threads_x = c.local;
    item.code_addr = 0x5028;
    const auto code = fx::program(c);
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    facts = analyze_compute_wave_ops(ins, code.data(), code.size());
    return item;
}
}   // namespace

TEST(ComputeWaveAdmission, AnExchangeModuleIsAdmittedWhenTheAnalysisAndTheLimitsAgree) {
    ComputeWaveOpFacts facts;
    const auto item = exchange_item({128, fx::Trips::Constant3}, facts);
    ASSERT_TRUE(compute_spirv_wave64_exchange(item.spirv));
    // For the dispatcher a loop is admissible (per-wave trip counts execute exactly); the refusal
    // line's candidate still names the analysis's own, stricter route, labelled as a candidate.
    EXPECT_EQ(exchange_limit(FakeContext{}, item, &facts, {32768, 1024}), nullptr);
    EXPECT_STREQ(compute_wave_candidate(FakeContext{}, item, facts, {32768, 1024}).route,
                 "n-lanes");
}

TEST(ComputeWaveAdmission, EachExchangeRefusalNamesItsOwnCondition) {
    ComputeWaveOpFacts facts;
    auto item = exchange_item({128, fx::Trips::Constant3}, facts);
    EXPECT_STREQ(exchange_limit(FakeContext{}, item, &facts, {512, 1024}), "shared-memory-budget");
    EXPECT_STREQ(exchange_limit(FakeContext{}, item, &facts, {0, 1024}),
                 "shared-memory-limit-unknown");
    EXPECT_STREQ(exchange_limit(FakeContext{}, item, &facts, {32768, 64}),
                 "workgroup-exceeds-device-limit");
    EXPECT_STREQ(exchange_limit(FakeContext{}, item, nullptr, {32768, 1024}), "unanalyzed");
    auto partial = item;
    partial.recompile_config.exact_thread_extent = true;
    partial.recompile_config.threads_x = 100;
    partial.recompile_config.threads_y = partial.recompile_config.threads_z = 1;
    EXPECT_STREQ(exchange_limit(FakeContext{}, partial, &facts, {32768, 1024}),
                 "partial-workgroup-barrier");
    auto ragged = item;
    ragged.recompile_config.local_x = 96;
    EXPECT_STREQ(exchange_limit(FakeContext{}, ragged, &facts, {32768, 1024}),
                 "workgroup-not-guest-wave-multiple");
}

TEST(ComputeWaveAdmission, TheAnalysisGatesAdmissionNotJustTheLimits) {
    ComputeWaveOpFacts facts;
    auto item = exchange_item({128, fx::Trips::Constant3}, facts);
    facts.control_flow_unmodelled = true;
    EXPECT_STREQ(exchange_limit(FakeContext{}, item, &facts, {32768, 1024}),
                 "control-flow-unmodelled");
    const ComputeWaveOpFacts writelane =
        one(ComputeCrossLaneKind::WriteLane, ComputeWaveContext::TopLevel);
    EXPECT_STREQ(exchange_limit(FakeContext{}, item, &writelane, {32768, 1024}),
                 "exchange-lowering-unavailable");
}

TEST(ComputeWaveAdmission, AnOrdinaryModuleIsNeverHeldToTheExchangeRules) {
    ComputeWaveOpFacts facts;
    auto item = exchange_item({128, fx::Trips::Constant3}, facts);
    item.spirv = fx::compile({128, fx::Trips::Constant3}, 0);
    ASSERT_FALSE(compute_spirv_wave64_exchange(item.spirv));
    EXPECT_EQ(exchange_limit(FakeContext{}, item, nullptr, {0, 0}), nullptr);
}

namespace {
int g_candidate_calls = 0;
Wave64Candidate counting_candidate(const void*) {
    ++g_candidate_calls;
    Wave64Candidate out;
    std::snprintf(out.route, sizeof out.route, "n-lanes");
    std::snprintf(out.reason, sizeof out.reason, "cross-lane-in-loop");
    return out;
}
}   // namespace

TEST(ComputeWaveAdmission, TheCandidateIsComputedOnlyWhenTheLineWillPrint) {
    using prosper::diagnostics::perf::note_unsupported_wave64;
    using prosper::diagnostics::perf::Wave64Refusal;
    g_candidate_calls = 0;
    for (int i = 0; i < 5; ++i)
        note_unsupported_wave64(Wave64Refusal::ComputeSubgroup, 64, 0x7a11e0a1, 0, UINT32_MAX, 32,
                                32, {}, &counting_candidate, nullptr);
    EXPECT_EQ(g_candidate_calls, 1) << "five refusals of one identity run the analysis once";
}
