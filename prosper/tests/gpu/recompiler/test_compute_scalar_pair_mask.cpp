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

constexpr uint64_t kAddress = 0xa4714001ULL;
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

// The program: prefix, then the path, then s_cmp_eq_u32 s0,0, then the probe ops.
Words program(const Words& path, const Words& probe) {
    return cat({&kPrefix, &path, &kScc, &probe});
}

// One probe per site family. The one-path pair s[4:5] is the marked source.
struct Probe {
    const char* what;
    Words ops;
    bool vcc_only;   // writes VCC only: the data is kept and the refusal is at the first mask read
};

std::vector<Probe> site_probes() {
    return {
        {"s_cselect_b64 vcc, s[4:5], -1", {0x85eac104u}, true},
        {"s_cselect_b32 vcc_lo, s4, -1", {0x856ac104u}, true},
        {"s_pack_ll_b32_b16 vcc_lo, s4, s5", {0x996a0504u}, true},
        {"s_lshl_b64 vcc, s[4:5], 1", {0x8fea8104u}, true},
        {"s_lshl_b64 exec, s[4:5], 1", {0x8ffe8104u}, false},
        {"s_bitreplicate_b64_b32 vcc, s4", {0xbeea3b04u}, true},
        {"s_mov_b64 vcc, s[4:5]", {0xbeea0404u}, true},
        {"s_mov_b64 exec, s[4:5]", {0xbefe0404u}, false},
        {"s_and_saveexec_b64 s[10:11], s[4:5]", {0xbe8a2404u}, false},
    };
}

// A probe that writes EXEC is refused AT the write, naming the guard. One that writes only VCC keeps
// the data words and publishes no lane view, so it is refused at the first instruction that reads
// VCC as a mask (the v_cndmask appended by compile()), as an unresolved operand.
void expect_refused_for_the_mark(Stage stage, const Words& code, const Probe& probe) {
    EXPECT_TRUE(compile(stage, code).empty()) << name(stage) << ": " << probe.what;
    const std::string reason = last_terminal_reject_reason(kAddress);
    if (probe.vcc_only) {
        // At the mask read, or (for the B32 cselect, which wave64 compute also gates on a whole-CFG
        // low-only proof of its own) at the probe itself; never earlier.
        const std::string consumer = "pc=" + std::to_string(code.size()) + " ";
        const std::string at_probe = "pc=" + std::to_string(code.size() - 1) + " ";
        EXPECT_NE(reason.find("mode=unresolved-operand"), std::string::npos)
            << name(stage) << ": " << probe.what << ": " << reason;
        EXPECT_TRUE(reason.find(consumer) != std::string::npos ||
                    reason.find(at_probe) != std::string::npos)
            << name(stage) << ": " << probe.what << ": not refused at the mask read: " << reason;
    } else {
        EXPECT_NE(reason.find(kReason), std::string::npos)
            << name(stage) << ": " << probe.what << ": refused for another reason: " << reason;
    }
}

// The same VCC-writing probes with VCC consumed only as scalar DATA (v_mov v3, vcc_lo / v1 for the
// fragment): no mask read, so the marked source is irrelevant and the program must compile.
Words compile_vcc_as_data(Stage stage, const Words& body) {
    const Words tail = stage == Stage::Fragment
                           ? Words{0x7e02026au, 0x7e000280u, 0x7e040280u, 0x7e0602f2u,
                                   0xf800180fu, 0x03020100u, 0xbf810000u}
                           : Words{0x7e06026au, 0xe0702000u, 0x80020300u, 0xbf810000u};
    return compile_whole(stage, cat({&body, &tail}));
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
            expect_refused_for_the_mark(stage, program(kOnePath, probe.ops), probe);
}

TEST(ScalarPairMask, VccUsedAsScalarScratchIsAdmitted) {
    // Over-refusal control (Black Flag: a loop counter / SMEM word shifted into VCC and read back as
    // an address): the pair is fabricated but never read as a mask.
    for (Stage stage : {Stage::Compute, Stage::Fragment})
        for (const Probe& probe : site_probes())
            if (probe.vcc_only)
                EXPECT_FALSE(compile_vcc_as_data(stage, program(kOnePath, probe.ops)).empty())
                    << name(stage) << ": " << probe.what;
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
    const std::string reason = last_terminal_reject_reason(kAddress);
    EXPECT_NE(reason.find("mode=unresolved-operand"), std::string::npos) << reason;
    EXPECT_NE(reason.find("pc=10 "), std::string::npos) << "refused at the v_cndmask: " << reason;
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

TEST(ScalarPairMask, ANeverWrittenSourceIsFabricatedToo) {
    // No branch: s[4:5] is defined, s20/s21 are NEVER written, and operand_bits reads their
    // absence as uconst(0). Each probe turns one into lane bits. VCC writers keep the data and are
    // refused at the first mask read; the program is otherwise unchanged.
    const std::vector<Probe> rows = {
        {"s_mov_b64 vcc, s[20:21]", {0xbeea0414u}, true},
        {"s_mov_b32 vcc_hi, s20", {0xbeeb0314u}, true},
        {"s_lshl_b64 vcc, s[4:5], s20", {0x8fea1404u}, true},
        {"s_mov_b64 exec, s[20:21]", {0xbefe0414u}, false},
    };
    for (Stage stage : {Stage::Compute, Stage::Fragment})
        for (const Probe& row : rows)
            expect_refused_for_the_mark(stage, program(kDefined, row.ops), row);
}

TEST(ScalarPairMask, ADirectDescriptorWordStillProjects) {
    // s[8:9] is the output V# the compute harness binds directly: real driver user data kept in
    // sreg_input, not a fabricated word, so projecting it must not be over-refused.
    const Words body = program(kDefined, {0xbeea0408u});
    const Words code = cat({&body, &kComputeTail});
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = 64;
    config.user_sgprs.assign(12, 0u);   // s0..s11 are launch user data; s[8:11] is the V#
    ShaderResourceTable table = output_table();
    table.resources[0].srt_offset = 0xFFFFFFFFu;   // a direct (inline) descriptor
    EXPECT_FALSE(recompile_compute(code.data(), code.size(), &table, config,
                                   {RecompileDiagnosticStage::Compute, kAddress})
                     .empty());
}

TEST(ScalarPairMask, AFabricatedVccHiSiblingIsNotAMask) {
    // vcc_hi is copied from the never-written s20 (so marked, with no branch in the program);
    // s_cselect_b32 vcc_lo then combines it with vcc_lo to form the lane bit. The marked sibling
    // must not become that bit (compute: this is the sibling check's own refusal).
    const Words fabricated_hi = {0xbe840380u, 0xbe8503c1u,
                                 0xbeeb0314u};   // s4=0, s5=-1, vcc_hi=s20
    const Words code = program(fabricated_hi, {0x856ac1c1u});
    EXPECT_TRUE(compile(Stage::Compute, code).empty());
    const std::string reason = last_terminal_reject_reason(kAddress);
    EXPECT_NE(reason.find("mode=unresolved-operand"), std::string::npos) << reason;
    EXPECT_NE(reason.find("pc=" + std::to_string(code.size() + 0u) + " "), std::string::npos)
        << reason;
}

TEST(ScalarPairMask, OtherSaveexecFormsHaveNoEmitterYet) {
    // Only s_and_saveexec_b64 has an emitter (and the guard). s_or_saveexec_b64 refuses as an
    // unresolved operand even for a fully defined pair, so a new emitter must bring the guard.
    for (Stage stage : {Stage::Compute, Stage::Fragment}) {
        EXPECT_TRUE(compile(stage, program(kDefined, {0xbe8a2504u})).empty()) << name(stage);
        EXPECT_NE(last_terminal_reject_reason(kAddress).find("mode=unresolved-operand"),
                  std::string::npos);
    }
}

namespace {

// The compute harness with real launch data: s0..s15 are user SGPRs, s[8:11] is the output V#
// (binding 3) and s[12:15] a second, unused direct V# (binding 4). Both are direct (inline)
// descriptors, so their words live in sreg_input rather than sreg.
Words compile_with_direct_descriptors(const Words& code) {
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = 64;
    config.user_sgprs.assign(16, 0u);
    ShaderResourceTable table = output_table();
    table.resources[0].srt_offset = 0xFFFFFFFFu;
    ShaderResource second = table.resources[0];
    second.binding = 4;
    second.sgpr_base = 12;
    table.resources.push_back(second);
    return recompile_compute(code.data(), code.size(), &table, config,
                             {RecompileDiagnosticStage::Compute, kAddress});
}

// s_mov_b32 s6, s8 | s_mov_b32 s7, s9 | s_mov_b64 vcc, s[6:7]: a register copy of a descriptor pair.
const Words kCopiedDescriptorPair = {0xbe860308u, 0xbe870309u, 0xbeea0406u};

}   // namespace

TEST(ScalarPairMask, ACopiedDirectDescriptorWordStillProjects) {
    // The review's executed row: the same driver words that project when read directly must not be
    // refused after a register copy. The copy's source marks are decided by the same
    // sreg_word_may_be_fabricated() call as a direct read, which is where sreg_input is exempted.
    const Words body = program(kDefined, kCopiedDescriptorPair);
    EXPECT_FALSE(compile_with_direct_descriptors(cat({&body, &kComputeTail})).empty())
        << last_terminal_reject_reason(kAddress);
}

TEST(ScalarPairMask, ADescriptorWordOverwrittenOnOnePathIsFabricated) {
    // The exemption must not survive a merge. s12 (a word of the unused direct V#) is overwritten on
    // ONE arm of the if; the skipped edge's phi input is sget()'s uconst(0), not the driver's word,
    // so after the join s12 is the fabricated zero even though the edge still held it in sreg_input.
    // s_cbranch_scc1 +1 | s_mov_b32 s12, -1 | s_mov_b64 vcc, s[12:13]
    const Words one_path = {0xbf068000u, 0xbf850001u, 0xbe8c03c1u};
    const Words probe = {0xbeea040cu};
    const Words code = cat({&kPrefix, &one_path, &probe, &kComputeTail});
    EXPECT_TRUE(compile_with_direct_descriptors(code).empty());
    const std::string reason = last_terminal_reject_reason(kAddress);
    EXPECT_NE(reason.find("mode=unresolved-operand"), std::string::npos) << reason;
    EXPECT_NE(reason.find("pc=" + std::to_string(kPrefix.size() + one_path.size() + 1u) + " "),
              std::string::npos)
        << "refused at the v_cndmask: " << reason;
    // Control: the same if writing s12 on the arm and leaving s13 alone, with the pair read only
    // after s12 is redefined on every path, is admitted.
    const Words redefined = {0xbe8c03c1u};
    EXPECT_FALSE(compile_with_direct_descriptors(
                     cat({&kPrefix, &one_path, &redefined, &probe, &kComputeTail}))
                     .empty())
        << last_terminal_reject_reason(kAddress);
}

TEST(ScalarPairMask, AFragmentUserDataWordReadsAsTheFabricatedZero) {
    // A fragment shell seeds no user data: operand_bits reads s13 / s[8:9] as uconst(0), so turning
    // one into lane bits, directly or through a copy, refuses at the mask read (#4725's contract,
    // unchanged by #4714). (s13, not Messenger's s3: this harness's prefix writes s[2:3] as a mask.)
    const Words vcc_hi_from_s13 = {0x876bff0du, 0x0000ffffu};   // s_and_b32 vcc_hi, s13, 0xffff
    const std::vector<Probe> rows = {
        {"s_and_b32 vcc_hi, s13, 0xffff", vcc_hi_from_s13, true},
        {"s_mov_b64 vcc, s[8:9]", {0xbeea0408u}, true},
        {"s_mov_b32 s6, s8 ; s_mov_b32 s7, s9 ; s_mov_b64 vcc, s[6:7]", kCopiedDescriptorPair,
         true},
    };
    for (const Probe& row : rows)
        expect_refused_for_the_mark(Stage::Fragment, program(kDefined, row.ops), row);
    // The same user-data word used as VCC scratch DATA (Messenger's NGG-preamble shape) compiles:
    // v_mov_b32 v1, vcc_hi | v0=0, v2=0, v3=1.0 | exp mrt0 v0..v3 | s_endpgm
    const Words data_tail = {0x7e02026bu, 0x7e000280u, 0x7e040280u, 0x7e0602f2u,
                             0xf800180fu, 0x03020100u, 0xbf810000u};
    const Words body = program(kDefined, vcc_hi_from_s13);
    EXPECT_FALSE(compile_whole(Stage::Fragment, cat({&body, &data_tail})).empty())
        << last_terminal_reject_reason(kAddress);
}

TEST(ScalarPairMask, CoverageDoesNotCallLaunchDataFabricated) {
    // recompile_coverage() seeds no launch state for any stage, so a VS's launch word is not a
    // fabricated zero there. The Messenger VS fixture's first two VCC writes, then a mask read:
    // s_and_b32 vcc_hi, s3, 0xffff | s_and_b32 vcc_lo, s11, 0x1 | v_cndmask_b32 v3, 0, 1.0, vcc
    const Words code = {0x876bff03u, 0x0000ffffu, 0x876aff0bu, 0x00000001u,
                        0xd5010003u, 0x01a9e480u, 0xbf810000u};
    const RecompileCoverage coverage = recompile_coverage(code.data(), code.size());
    EXPECT_EQ(coverage.unsupported, 0u)
        << "first_bad fmt=" << coverage.first_bad_fmt << " op=0x" << std::hex
        << coverage.first_bad_op << " pc=" << std::dec << coverage.first_bad_pc;
}

namespace {

// Kena's level-load pixel program 0x5007ad0000, reduced (#4749 review). EXEC is saved into s[20:21]
// and both halves are spilled to v20 lanes 0/1; s20 is then reused for data (Kena reloads s100 from
// memory), and the loop re-spills the high half on every trip (Kena's pc 853), so that slot is
// loop-carried. After the loop both halves are reloaded and EXEC is restored from them. (Not
// s[8:11]: that is the compute harness's output V#.)
//   s_mov_b64 s[20:21], exec | v_writelane v20, <pre>, 1 | v_writelane v20, s20, 0
//   s_mov_b32 s20, 5 | s_mov_b32 s22, 0 | s_mov_b32 s23, 3
//   L: s_cmp_lt_u32 s22, s23 | s_cbranch_scc0 X | v_writelane v20, <back>, 1
//      s_add_u32 s22, s22, 1 | s_branch L
//   X: v_readlane s20, v20, 0 | v_readlane s21, v20, 1 | s_mov_b64 exec, s[20:21]
Words spill_restore_through_loop(uint32_t preheader_hi, uint32_t backedge_hi) {
    return {0xbe94047eu, 0xd7610014u, 0x00010200u | preheader_hi,
            0xd7610014u, 0x00010014u, 0xbe940385u,
            0xbe960380u, 0xbe970383u, 0xbf0a1716u,
            0xbf840004u, 0xd7610014u, 0x00010200u | backedge_hi,
            0x80168116u, 0xbf82fffau, 0xd7600014u,
            0x00010114u, 0xd7600015u, 0x00010314u,
            0xbefe0414u};
}
constexpr uint32_t kS21 = 21;   // the saved EXEC_HI
constexpr uint32_t kS24 = 24;   // never written: operand_bits reads it as the fabricated zero

const Probe kRestore = {"s_mov_b64 exec, s[20:21] after the reload", {}, false};

}   // namespace

TEST(ScalarPairMask, AnExecSpillRestoreThroughALoopCompiles) {
    // (a) Every word is real: the high half survives s20's reuse (it is materialized from the mask
    // before s20 is overwritten), and the loop-carried slot's exit value is defined because both
    // its preheader and its back-edge values are.
    const Words body = spill_restore_through_loop(kS21, kS21);
    for (Stage stage : {Stage::Compute, Stage::Fragment})
        EXPECT_FALSE(compile(stage, cat({&kPrefix, &body})).empty())
            << name(stage) << ": " << last_terminal_reject_reason(kAddress);
}

TEST(ScalarPairMask, AFabricatedWordSpilledThroughALoopRefusesAtTheRestore) {
    // (b) The same shape with one fabricated input: the back edge re-spills the never-written s24,
    // or the preheader spills it and the back edge spills the real s21. Either way the loop's exit
    // value may be the zero, so the EXEC restore refuses at the write.
    const Words backedge = spill_restore_through_loop(kS21, kS24);
    const Words seed = spill_restore_through_loop(kS24, kS21);
    expect_refused_for_the_mark(Stage::Fragment, cat({&kPrefix, &backedge}), kRestore);
    expect_refused_for_the_mark(Stage::Fragment, cat({&kPrefix, &seed}), kRestore);
}

TEST(ScalarPairMask, AOnePathWordSpilledRefusesAtTheRestore) {
    // (b) s25 is written on ONE arm of an if, then spilled as EXEC_HI and restored.
    //   s_mov_b64 s[20:21], exec | v_writelane v20, s20, 0 | s_cmp_eq_u32 s0, 0
    //   s_cbranch_scc1 +1 | s_mov_b32 s25, -1 | v_writelane v20, s25, 1
    //   v_readlane s20, v20, 0 | v_readlane s21, v20, 1 | s_mov_b64 exec, s[20:21]
    const Words body = {0xbe94047eu, 0xd7610014u, 0x00010014u, 0xbf068000u, 0xbf850001u,
                        0xbe9903c1u, 0xd7610014u, 0x00010219u, 0xd7600014u, 0x00010114u,
                        0xd7600015u, 0x00010314u, 0xbefe0414u};
    expect_refused_for_the_mark(Stage::Fragment, cat({&kPrefix, &body}), kRestore);
}

TEST(ScalarPairMask, ASlotWrittenOnOnePathRefusesAtTheRestore) {
    // (c) EXEC_HI is spilled on ONE arm only, so on the skipped edge the slot holds the merge's
    // placeholder, not a word of EXEC (#4740: a one-edge slot is fabricated after the merge).
    //   s_mov_b64 s[20:21], exec | v_writelane v20, s20, 0 | s_cmp_eq_u32 s0, 0
    //   s_cbranch_scc1 +2 | v_writelane v20, s21, 1
    //   v_readlane s20, v20, 0 | v_readlane s21, v20, 1 | s_mov_b64 exec, s[20:21]
    const Words body = {0xbe94047eu, 0xd7610014u, 0x00010014u, 0xbf068000u,
                        0xbf850002u, 0xd7610014u, 0x00010215u, 0xd7600014u,
                        0x00010114u, 0xd7600015u, 0x00010314u, 0xbefe0414u};
    expect_refused_for_the_mark(Stage::Fragment, cat({&kPrefix, &body}), kRestore);
    // Control: the same if, with EXEC_HI spilled before the branch as well, compiles.
    const Words both = {0xbe94047eu, 0xd7610014u, 0x00010014u, 0xd7610014u, 0x00010215u,
                        0xbf068000u, 0xbf850002u, 0xd7610014u, 0x00010215u, 0xd7600014u,
                        0x00010114u, 0xd7600015u, 0x00010314u, 0xbefe0414u};
    for (Stage stage : {Stage::Compute, Stage::Fragment})
        EXPECT_FALSE(compile(stage, cat({&kPrefix, &both})).empty())
            << name(stage) << ": " << last_terminal_reject_reason(kAddress);
}
