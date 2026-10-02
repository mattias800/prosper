// compute_program_facts: the per-dispatch program facts (decode, native-multiwave probe, GDS use)
// are memoized per exact program bytes, and a memoized answer is indistinguishable from a fresh one
// -- including the probe's side effect on the terminal reject-reason map.
//
// Arms:
//  1. memo hit: a second dispatch of the same program runs the probe zero more times (fails if the
//     cache is removed or the executor keeps re-deriving per dispatch).
//  2. equality: every cached fact equals a direct evaluation, for a policy-positive and a
//     policy-negative program.
//  3. rewrite: new bytes at the same address are re-analyzed, never served from the old entry.
//  4. replay: a program whose probe records a terminal reject reason re-records it on a HIT after
//     the map entry was overwritten, exactly as a fresh probe would (positive control included).
//  5. may_specialize: the scratch-copy gate is true exactly for the resource shapes the two
//     specializers act on.
#include "gpu/execute/compute_program_facts.hpp"
#include <gtest/gtest.h>
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <cstdio>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

using namespace prosper::gpu;

namespace {
int failures = 0;
void check(bool ok, const char* message) { EXPECT_TRUE(ok) << message; }

bool same_stream(const std::vector<Rdna2Inst>& a, const std::vector<Rdna2Inst>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].pc != b[i].pc || a[i].fmt != b[i].fmt || a[i].opcode != b[i].opcode ||
            a[i].len_dwords != b[i].len_dwords ||
            std::memcmp(a[i].words, b[i].words, sizeof(a[i].words)) != 0)
            return false;
    return true;
}

// Four sequential top-level wave votes: the native-multiwave policy says yes.
const uint32_t kVoteHeavy[] = {
    0xBE800380u, 0xBE810380u, 0x7DA20200u, 0xBF880001u, 0xBE800381u, 0xBEFE04C1u, 0xBF8A0000u,
    0x7DA20600u, 0xBF880001u, 0xBE810382u, 0xBEFE04C1u, 0xBF8A0000u, 0x7DA20200u, 0xBF880001u,
    0xBE800383u, 0xBEFE04C1u, 0xBF8A0000u, 0x7DA20600u, 0xBF880001u, 0xBE810384u, 0xBEFE04C1u,
    0xBF8A0000u, 0x80020100u, 0x7E040C02u, 0xBF810000u,
};
// Guarded work after a uniform barrier: linearized, so the policy says no.
const uint32_t kLinearized[] = {
    0xBF8A0000u, 0x7DA20200u, 0xBF880007u, 0x7DA20200u, 0xBF880005u, 0x7DA20200u,
    0xBF880003u, 0x7DA20200u, 0xBF880001u, 0x7E040C00u, 0xBF810000u,
};
// A barrier plus an unclaimed forward s_cbranch_execnz: the probe's structured-branch scan
// declines it and records a terminal reject reason for the program.
const uint32_t kProbeRejects[] = {
    0xBF8A0000u,   // s_barrier
    0xBF890001u,   // s_cbranch_execnz +1
    0x7E040C00u,   // v_cvt_f32_u32 v2, s0
    0xBF810000u,   // s_endpgm
};

RecompileDiagnosticContext at(uint64_t address) {
    return {RecompileDiagnosticStage::Compute, address};
}
} // namespace

TEST(ComputeProgramFacts, Contract) {
    // 1 + 2: hit counting and equality with a direct evaluation.
    reset_compute_program_facts_for_test();
    std::vector<uint32_t> vote(std::begin(kVoteHeavy), std::end(kVoteHeavy));
    std::vector<uint32_t> linear(std::begin(kLinearized), std::end(kLinearized));
    const auto first = compute_program_facts(vote.data(), vote.size(), at(0x5000));
    const auto second = compute_program_facts(vote.data(), vote.size(), at(0x5000));
    const ComputeProgramFactsStats after_two = compute_program_facts_stats();
    check(after_two.probe_evaluations == 1 && after_two.hits == 1 && after_two.misses == 1,
          "a repeated dispatch of one program evaluates the probe once");
    check(first == second, "the hit returns the analyzed entry");

    std::vector<Rdna2Inst> direct;
    rdna2_walk(vote.data(), vote.size(), direct);
    check(same_stream(first->decoded, direct), "cached decode equals rdna2_walk");
    check(first->prefers_native_multiwave ==
              compute_shader_prefers_native_multiwave(vote.data(), vote.size()) &&
              first->prefers_native_multiwave,
          "cached probe equals a direct probe (policy-positive program)");
    const auto linear_facts = compute_program_facts(linear.data(), linear.size(), at(0x6000));
    check(!linear_facts->prefers_native_multiwave &&
              linear_facts->prefers_native_multiwave ==
                  compute_shader_prefers_native_multiwave(linear.data(), linear.size()),
          "cached probe equals a direct probe (policy-negative program)");
    check(!first->uses_gds && !linear_facts->uses_gds, "no GDS access in either program");

    // 3: the same address with rewritten bytes (a recycled or patched program).
    std::vector<uint32_t> buffer = vote;
    const auto before = compute_program_facts(buffer.data(), buffer.size(), at(0x7000));
    buffer.assign(linear.begin(), linear.end());
    buffer.resize(vote.size(), 0xBF810000u);   // keep the dword count: same cache key
    const uint64_t evals_before = compute_program_facts_stats().probe_evaluations;
    const auto rewritten = compute_program_facts(buffer.data(), buffer.size(), at(0x7000));
    check(compute_program_facts_stats().probe_evaluations == evals_before + 1,
          "rewritten bytes at a cached address are re-analyzed");
    check(before->prefers_native_multiwave && !rewritten->prefers_native_multiwave &&
              rewritten->prefers_native_multiwave ==
                  compute_shader_prefers_native_multiwave(buffer.data(), buffer.size()),
          "the re-analysis answers for the new bytes, not the old entry");

    // 4: reject-reason replay. Positive control first: a fresh probe records a reason.
    std::vector<uint32_t> rejecting(std::begin(kProbeRejects), std::end(kProbeRejects));
    constexpr uint64_t kReject = 0x8000;
    const auto reject_facts = compute_program_facts(rejecting.data(), rejecting.size(), at(kReject));
    const std::string recorded = last_terminal_reject_reason(kReject);
    std::fprintf(stderr, "  probe reason: '%s' (%zu captured)\n", recorded.c_str(),
                 reject_facts->probe_reject_reasons.size());
    check(!recorded.empty() && !reject_facts->probe_reject_reasons.empty(),
          "positive control: the probe records a terminal reject reason for this program");
    // Another site overwrites the program's reason (as the recompile does after the probe).
    record_recompile_reject_reason_for_test(at(kReject), "other-site", "terminal", "overwritten");
    check(last_terminal_reject_reason(kReject) == "other-site overwritten",
          "the overwrite took effect");
    const uint64_t evals_replay = compute_program_facts_stats().probe_evaluations;
    compute_program_facts(rejecting.data(), rejecting.size(), at(kReject));
    check(compute_program_facts_stats().probe_evaluations == evals_replay,
          "the replay dispatch was a cache hit");
    check(last_terminal_reject_reason(kReject) == recorded,
          "a hit re-records the probe's reason exactly as a fresh probe would");

    // 5: the specialization gate.
    ShaderResourceTable table;
    ShaderResource ordinary;
    ordinary.fetch_pc = 3;
    table.resources.push_back(ordinary);
    check(!compute_resource_paths_may_specialize(table, 64) &&
              !compute_resource_paths_may_specialize(table, 32),
          "an ordinary table cannot specialize");
    static uint8_t null_bvh_backing[256];
    ShaderResource null_bvh;
    null_bvh.cls = ResourceClass::ConstantBuffer;
    null_bvh.format = DataFormat::Uint32;
    null_bvh.num_components = 1;
    null_bvh.gpu_addr = 0;
    null_bvh.size = 256;
    null_bvh.stride = 0;
    null_bvh.fetch_pc = 7;
    null_bvh.host_data = null_bvh_backing;
    null_bvh.host_data_size = sizeof(null_bvh_backing);
    check(is_proven_null_bvh(null_bvh), "fixture is a proven-null BVH marker");
    ShaderResourceTable bvh_table = table;
    bvh_table.resources.push_back(null_bvh);
    check(compute_resource_paths_may_specialize(bvh_table, 32) &&
              compute_resource_paths_may_specialize(bvh_table, 64),
          "a proven-null BVH table may specialize");
    ShaderResource zero_record;
    zero_record.cls = ResourceClass::ConstantBuffer;
    zero_record.format = DataFormat::Unknown;
    zero_record.num_components = 0;
    zero_record.gpu_addr = 0;
    zero_record.size = 0;
    zero_record.stride = 0;
    zero_record.srt_offset = 0xFFFFFFFFu;
    zero_record.sgpr_base = 0xFFFFFFFFu;
    zero_record.fetch_pc = 9;
    zero_record.host_data = nullptr;
    zero_record.host_data_size = 0;
    check(is_zero_record_raw_buffer(zero_record), "fixture is a zero-record marker");
    ShaderResourceTable zero_table = table;
    zero_table.resources.push_back(zero_record);
    check(compute_resource_paths_may_specialize(zero_table, 64) &&
              !compute_resource_paths_may_specialize(zero_table, 32),
          "a zero-record table may specialize only at wave64");

    if (failures) std::fprintf(stderr, "%d FAILED\n", failures);
    else std::fprintf(stderr, "all compute_program_facts checks passed\n");
    EXPECT_EQ(failures, 0);
}
