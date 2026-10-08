// ADR 0028: the frontend glue that names a Wave64 route CANDIDATE on a refusal line. A stand-in
// context supplies the fields live_compute.cpp's real one has, so nothing here needs a Vulkan device.
#include <gtest/gtest.h>

#include <vector>

#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/compute_wave_route.hpp"
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
