// test_compute_scalar_pair_mask -- a Wave64 program's scalar DATA words never become a per-lane mask
// bit when one of them may be the structured emitter's fabricated zero (#4714, GPU-5/FAIL-1; the
// fragment pair projection is #4711). Compile-level only: lane selection itself is executed by
// test_fragment_scalar_pair_mask.
//
// The one-path programs are assembled by hand here, outside the recompiler code that sets the mark:
// s[4:5] is written on ONE arm of a scalar if whose bound no fold knows (v_readfirstlane), so both
// words are merge-marked after the join; a probe op then turns that pair into a lane mask. Every
// SITE FAMILY that derives a lane bit from scalar data is probed on both stages:
//   cselect_b64 -> vcc, cselect_b32 -> vcc_lo, pack -> vcc_lo, lshl_b64 -> vcc/exec,
//   bitreplicate -> vcc, s_mov_b64 -> vcc/exec, and the s_and_b64 pair projection.
// Each negative arm pins the refusal REASON (the shared guard's own diagnostic), so a refusal that
// merely happens for another reason cannot satisfy it. The controls keep admitting a fully defined
// pair, and a pair written on BOTH arms of the if.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

using namespace prosper::gpu;

namespace {

using Words = std::vector<uint32_t>;

constexpr uint64_t kAddress = 0xa4714001ull;
constexpr const char* kReason = "scalar-fabricated-lane-mask";

// v_mov v5,0 | v_readfirstlane s0,v5 (no fold knows s0) | v_cmp_eq_u32 s[2:3],0,0
const Words kPrefix = {0x7e0a0280u, 0x7e000505u, 0xd4c20002u, 0x00010080u};
// s_cmp_eq_u32 s0,0 | s_cbranch_scc1 +2 | s_mov_b32 s4,-1 | s_mov_b32 s5,-1   <- the one path
const Words kOnePath = {0xbf068000u, 0xbf850002u, 0xbe8403c1u, 0xbe8503c1u};
// The same if with BOTH arms writing s[4:5]: then s4=s5=-1; else s4=0, s5=-1.
const Words kBothPaths = {0xbf068000u, 0xbf850003u, 0xbe8403c1u, 0xbe8503c1u,
                          0xbf820002u, 0xbe840380u, 0xbe8503c1u};
// No branch: s[4:5] = -1, fully defined.
const Words kDefined = {0xbe8403c1u, 0xbe8503c1u};
// s_cmp_eq_u32 s0,0: the SCC the cselect forms consume.
const Words kScc = {0xbf068000u};

Words cat(std::initializer_list<const Words*> parts) {
    Words out;
    for (const Words* part : parts) out.insert(out.end(), part->begin(), part->end());
    return out;
}

// v_cndmask_b32_e64 v3,0,1.0,vcc | buffer_store_dword v3 -> out[x] | s_endpgm
const Words kComputeTail = {0xd5010003u, 0x01a9e480u, 0xe0702000u, 0x80020300u, 0xbf810000u};
// v_cndmask_b32_e64 v1,0,1.0,vcc | v0=0, v2=0, v3=1.0 | exp mrt0 v0..v3 | s_endpgm
const Words kFragmentTail = {0xd5010001u, 0x01a9e480u, 0x7e000280u, 0x7e040280u,
                             0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u};

ShaderResourceTable output_table() {
    ShaderResourceTable table;
    ShaderResource out{};
    out.cls = ResourceClass::ConstantBuffer;
    out.format = DataFormat::Uint32;
    out.num_components = 1;
    out.binding = 3;
    out.stride = 4;
    out.sgpr_base = 8;
    table.resources.push_back(out);
    return table;
}

enum class Stage { Compute, Fragment };

Words compile_whole(Stage stage, const Words& code) {
    if (stage == Stage::Fragment)
        return recompile_fragment(code.data(), code.size(), nullptr, nullptr, UINT32_MAX, nullptr,
                                  false, {RecompileDiagnosticStage::Fragment, kAddress});
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = 64;
    const ShaderResourceTable table = output_table();
    return recompile_compute(code.data(), code.size(), &table, config,
                             {RecompileDiagnosticStage::Compute, kAddress});
}

Words compile(Stage stage, const Words& body) {
    return compile_whole(stage,
                         cat({&body, stage == Stage::Fragment ? &kFragmentTail : &kComputeTail}));
}

const char* name(Stage stage) {
    return stage == Stage::Fragment ? "fragment" : "compute";
}

// prefix | <path> | s_cmp_eq_u32 s0,0 | <probe ops>
Words program(const Words& path, const Words& probe) {
    return cat({&kPrefix, &path, &kScc, &probe});
}

// One probe per site family. The one-path pair s[4:5] is the marked source.
struct Probe {
    const char* what;
    Words ops;
};

std::vector<Probe> site_probes() {
    return {
        {"s_cselect_b64 vcc, s[4:5], -1", {0x85eac104u}},
        {"s_cselect_b32 vcc_lo, s4, -1", {0x856ac104u}},
        {"s_pack_ll_b32_b16 vcc_lo, s4, s5", {0x996a0504u}},
        {"s_lshl_b64 vcc, s[4:5], 1", {0x8fea8104u}},
        {"s_lshl_b64 exec, s[4:5], 1", {0x8ffe8104u}},
        {"s_bitreplicate_b64_b32 vcc, s4", {0xbeea3b04u}},
        {"s_mov_b64 vcc, s[4:5]", {0xbeea0404u}},
        {"s_mov_b64 exec, s[4:5]", {0xbefe0404u}},
        {"s_and_saveexec_b64 s[10:11], s[4:5]", {0xbe8a2404u}},
    };
}

void expect_refused_for_the_mark(Stage stage, const Words& code, const std::string& what) {
    EXPECT_TRUE(compile(stage, code).empty()) << name(stage) << ": " << what;
    const std::string reason = last_terminal_reject_reason(kAddress);
    EXPECT_NE(reason.find(kReason), std::string::npos)
        << name(stage) << ": " << what << ": refused for another reason: " << reason;
}

}   // namespace

TEST(ScalarPairMask, DefinedPairIsAdmittedOnBothStages) {
    // The reviewer's control: s[4:5] defined, no branch, then the cselect probe.
    for (Stage stage : {Stage::Compute, Stage::Fragment})
        EXPECT_FALSE(compile(stage, program(kDefined, {0x85eac104u})).empty())
            << name(stage) << ": a fully defined pair is a per-lane bit, not a refusal";
}

TEST(ScalarPairMask, PairWrittenOnBothArmsIsAdmitted) {
    // Over-refusal control: every path writes both words, so there is no fabricated zero.
    for (Stage stage : {Stage::Compute, Stage::Fragment})
        EXPECT_FALSE(compile(stage, program(kBothPaths, {0x85eac104u})).empty())
            << name(stage) << ": a pair defined on both arms of the if is definite";
}

TEST(ScalarPairMask, EverySiteFamilyRefusesAOnePathPair) {
    for (Stage stage : {Stage::Compute, Stage::Fragment})
        for (const Probe& probe : site_probes())
            expect_refused_for_the_mark(stage, program(kOnePath, probe.ops), probe.what);
}

TEST(ScalarPairMask, ReviewersExecutedProbeRefuses) {
    // The probe words as the review ran them (compute form).
    const Words probe = {0x7e0a0280u, 0x7e000505u, 0xd4c20002u, 0x00010080u, 0xbf068000u,
                         0xbf850002u, 0xbe8403c1u, 0xbe8503c1u, 0xbf068000u, 0x85eac104u,
                         0xd5010003u, 0x01a9e480u, 0xe0702000u, 0x80020300u, 0xbf810000u};
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = 64;
    const ShaderResourceTable table = output_table();
    EXPECT_TRUE(recompile_compute(probe.data(), probe.size(), &table, config,
                                  {RecompileDiagnosticStage::Compute, kAddress})
                    .empty());
    EXPECT_NE(last_terminal_reject_reason(kAddress).find(kReason), std::string::npos);
}

TEST(ScalarPairMask, AndPairProjectionRefusesAOnePathPair) {
    // The original #4714 site, whose destination is plain scalar scratch so only the pair
    // projection can refuse: s_and_b64 s[6:7], s[4:5], s[2:3], consumed by v_cndmask through s[6:7].
    const Words and_pair = {0x87860204u};
    for (Stage stage : {Stage::Compute, Stage::Fragment}) {
        Words code = cat({&kPrefix, &kOnePath, &and_pair});
        const Words select =
            stage == Stage::Compute
                ? Words{0xd5010003u, 0x0019e480u, 0xe0702000u, 0x80020300u, 0xbf810000u}
                : Words{0xd5010001u, 0x0019e480u, 0x7e000280u, 0x7e040280u,
                        0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u};
        code.insert(code.end(), select.begin(), select.end());
        EXPECT_TRUE(compile_whole(stage, code).empty()) << name(stage);
    }
}
