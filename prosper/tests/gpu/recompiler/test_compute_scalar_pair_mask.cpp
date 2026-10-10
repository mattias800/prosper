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

// NOLINTBEGIN(bugprone-throwing-static-initialization): constant instruction words of a test
// fixture; an allocation failure here ends the test binary either way.
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

// NOLINTEND(bugprone-throwing-static-initialization)

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
// NOLINTNEXTLINE(bugprone-throwing-static-initialization): fixture instruction words, as above.
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

// The CFG dispatcher reloads every spill slot at a block entry. A slot written on EVERY path from a
// defined word (here the saved EXEC halves) reads back as defined, so Kena's and GTA V's
// `s_mov_b64 exec, s[6:7]` restore compiles; a slot spilled from a fabricated word still refuses.
// NOLINTBEGIN(bugprone-throwing-static-initialization): fixture instruction words, as above.
namespace {
const Words kDispatcherBody = {
    0x7e040280u, 0x7e060280u, 0x7e080280u, 0x7e0a0280u,   // v2..v5 = 0
    0x7c020300u, 0xbf860003u,   // v_cmp_lt_f32 vcc,v0,v1 ; vccz -> +3
    0x7c020300u, 0xbf860002u,   // v_cmp ; vccz -> +2 (crosses the first)
    0x7e040281u, 0x7e060281u,   // v2 = 1 ; v3 = 1
    0x7c020300u, 0xbf860002u,   // v_cmp ; vccz -> +2
    0x7e080281u, 0xbf820001u, 0x7e0a0281u};   // v4 = 1 ; s_branch +1 ; v5 = 1
const Words kWriteLanes = {0xd7610014u, 0x0001022du, 0xd7610014u, 0x0001002cu};
const Words kReadLanesAndRestore = {0xd7600006u, 0x00010114u, 0xd7600007u, 0x00010314u,
                                    0xbefe0406u};
const Words kExecTail = {0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
                         0xf800000fu, 0x03020100u, 0xbf810000u};
}   // namespace
// NOLINTEND(bugprone-throwing-static-initialization)

TEST(ScalarPairMask, ADispatcherReloadedDefinedExecPairCompiles) {
    const Words saved_exec = {0xbeac047eu};   // s_mov_b64 s[44:45], exec
    const Words code =
        cat({&saved_exec, &kWriteLanes, &kDispatcherBody, &kReadLanesAndRestore, &kExecTail});
    EXPECT_FALSE(compile_whole(Stage::Fragment, code).empty());
}

TEST(ScalarPairMask, ADispatcherReloadedFabricatedExecPairStillRefuses) {
    const Words one_path = {0xbf068000u, 0xbf850002u, 0xbeac03c1u, 0xbead03c1u};
    const Words code = cat(
        {&kPrefix, &one_path, &kWriteLanes, &kDispatcherBody, &kReadLanesAndRestore, &kExecTail});
    EXPECT_TRUE(compile_whole(Stage::Fragment, code).empty());
}

// A v_readlane result is defined only when the slot it reads is (#4749 review): a fabricated word
// spilled into v21, read back into s[44:45] and re-spilled into v20 before the dispatcher would
// otherwise reach the EXEC restore as a defined slot. Each chain row puts a different source in
// front of kWriteLanes, so the restore that refuses is the dispatcher-reloaded one. Fragment only:
// on compute even the EXEC control below refuses at the restore (with and without this rule), so a
// compute refusal there could not tell the relay from that.
// NOLINTBEGIN(bugprone-throwing-static-initialization): fixture instruction words, as above.
namespace {
// v_writelane v21, s50, 0 | v_writelane v21, s51, 1
const Words kSpillS50ToV21 = {0xd7610015u, 0x00010032u, 0xd7610015u, 0x00010233u};
// v_readlane s44, v21, 0 | v_readlane s45, v21, 1
const Words kReadV21ToS44 = {0xd760002cu, 0x00010115u, 0xd760002du, 0x00010315u};
// v_readlane s46, v21, 0 | v_readlane s47, v21, 1 | s_mov_b64 s[44:45], s[46:47]
const Words kReadV21ViaS46 = {0xd760002eu, 0x00010115u, 0xd760002fu, 0x00010315u, 0xbeac042eu};
// s_cmp_eq_u32 s0, 0 | s_cbranch_scc1 +2 | s_mov_b32 s50, -1 | s_mov_b32 s51, -1
const Words kS50OnOnePath = {0xbf068000u, 0xbf850002u, 0xbeb203c1u, 0xbeb303c1u};
const Words kS50FromExec = {0xbeb2047eu};   // s_mov_b64 s[50:51], exec
// v_readlane s6, v21, 0 | v_readlane s7, v21, 1: v21 read back into the pair EXEC is later
// restored from, with no second spill.
const Words kReadV21ToS6 = {0xd7600006u, 0x00010115u, 0xd7600007u, 0x00010315u};
const Words kRestoreS6 = {0xbefe0406u};   // s_mov_b64 exec, s[6:7]
const Words kNothing = {};

// Every chain row's code: `source` + the v21 round trip (`relay`) + kWriteLanes + the dispatcher.
Words relay_chain(const Words& source, const Words& relay) {
    return cat({&kPrefix, &source, &kSpillS50ToV21, &relay, &kWriteLanes, &kDispatcherBody,
                &kReadLanesAndRestore, &kExecTail});
}

// The refusal is the restore itself, at its own pc. Inside a dispatcher case the guard's refusal
// reaches the terminal line as the case's unresolved operand, as for the PR's own dispatcher arm.
void expect_restore_refused(const Words& code, uint32_t restore_word) {
    EXPECT_TRUE(compile_whole(Stage::Fragment, code).empty());
    size_t pc = 0;
    while (pc < code.size() && code[pc] != restore_word) ++pc;
    const std::string reason = last_terminal_reject_reason(kAddress);
    EXPECT_NE(reason.find("mode=unresolved-operand pc=" + std::to_string(pc) + " "),
              std::string::npos)
        << "not refused at the restore: " << reason;
}
}   // namespace
// NOLINTEND(bugprone-throwing-static-initialization)

TEST(ScalarPairMask, ADefinedWordRelayedThroughAReadlaneCompiles) {
    // Control: s[50:51] holds EXEC, so v21's slots and the readlane'd s[44:45] are all defined.
    EXPECT_FALSE(compile_whole(Stage::Fragment, relay_chain(kS50FromExec, kReadV21ToS44)).empty())
        << last_terminal_reject_reason(kAddress);
    EXPECT_FALSE(compile_whole(Stage::Fragment, relay_chain(kS50FromExec, kReadV21ViaS46)).empty())
        << last_terminal_reject_reason(kAddress);
}

TEST(ScalarPairMask, ANeverWrittenWordRelayedThroughAReadlaneRefuses) {
    expect_restore_refused(relay_chain(kNothing, kReadV21ToS44), 0xbefe0406u);
}

TEST(ScalarPairMask, AOnePathWordRelayedThroughAReadlaneRefuses) {
    expect_restore_refused(relay_chain(kS50OnOnePath, kReadV21ToS44), 0xbefe0406u);
}

TEST(ScalarPairMask, ARelayedWordCopiedBeforeTheSpillRefuses) {
    // The readlane result reaches slot B through an s_mov_b64, not directly.
    expect_restore_refused(relay_chain(kNothing, kReadV21ViaS46), 0xbefe0406u);
    expect_restore_refused(relay_chain(kS50OnOnePath, kReadV21ViaS46), 0xbefe0406u);
}

TEST(ScalarPairMask, AReadlanedFabricatedWordCrossingTheDispatcherRefuses) {
    // No second spill: s[6:7] is read back from v21 BEFORE the dispatcher and restored after it.
    const Words never =
        cat({&kPrefix, &kSpillS50ToV21, &kReadV21ToS6, &kDispatcherBody, &kRestoreS6, &kExecTail});
    const Words one_path = cat({&kPrefix, &kS50OnOnePath, &kSpillS50ToV21, &kReadV21ToS6,
                                &kDispatcherBody, &kRestoreS6, &kExecTail});
    expect_restore_refused(never, 0xbefe0406u);
    expect_restore_refused(one_path, 0xbefe0406u);
}

// Kena's per-cone loop (cs 0x50071e0000): an outer s_cmp/s_cbranch_scc1 loop around an inner step loop
// that leaves through s_cbranch_execz after v_cmp writes VCC. Its tail recycles VCC as scalar scratch
// from the loop counter (only blanket-marked at the header). The check block's VCC must stay a mask:
// dropping VCC's lane view for that write left the loop exit unable to name VCC ("check-mask=0
// body-mask=0") and refused a program main compiles.
// NOLINTBEGIN(bugprone-throwing-static-initialization): fixture instruction words, as above.
namespace {
const Words kNestedLoopsWithVccScratchTail = {
    0x7d820082u,   // v_cmp_lt_u32 vcc, 2, v0: VCC holds a lane mask on loop entry
    0xbe9a0380u,   // s_mov_b32 s26, 0
    0xbe9b0380u,   // OUTER: s_mov_b32 s27, 0
    0x7d82001bu,   // INNER: v_cmp_lt_u32 vcc, s27, v0
    0xbe9c246au,   // s_and_saveexec_b64 s[28:29], vcc
    0xbf880004u,   // s_cbranch_execz -> exit of the inner loop
    0x801b811bu,   // s_add_u32 s27, s27, 1
    0x88fe1c7eu,   // s_or_b64 exec, exec, s[28:29]
    0xbf0a841bu,   // s_cmp_lt_u32 s27, 4
    0xbf85fff9u,   // s_cbranch_scc1 INNER
    0x88fe1c7eu,   // s_or_b64 exec, exec, s[28:29]
    0x8f6a821au,   // s_lshl_b32 vcc_lo, s26, 2   (VCC scratch from the counter, after the inner loop)
    0x801a811au,   // s_add_u32 s26, s26, 1
    0xbf0a891au,   // s_cmp_lt_u32 s26, 9
    0xbf85fff4u};   // s_cbranch_scc1 OUTER
}   // namespace
// NOLINTEND(bugprone-throwing-static-initialization)

TEST(ScalarPairMask, ANestedLoopWhoseTailRecyclesVccFromItsCounterCompiles) {
    const Words tail = {0x7e0602f2u, 0xe0702000u, 0x80020300u, 0xbf810000u};   // v3 = 1.0; store
    const Words code = cat({&kNestedLoopsWithVccScratchTail, &tail});
    EXPECT_FALSE(compile_whole(Stage::Compute, code).empty())
        << last_terminal_reject_reason(kAddress);
}

// The loop header's blanket mark is an assumption: a carried word defined on the preheader edge is
// taken as defined on the back edge too, so a lane-bit read in the body admits it before the back
// edge exists. The back edge checks it (#4749 review). s[16:17] = -1 before a counted loop whose
// body first reads s[16:17] as VCC's lane bits (v_cndmask), then rewrites s[16:17] (not s[8:11]:
// that is the harness's output V#):
//   s_mov_b32 s16, -1 | s_mov_b32 s17, -1 | s_mov_b32 s22, 0 | s_mov_b32 s23, 3 | v_mov v3, 0
//   L: s_cmp_lt_u32 s22, s23 | s_cbranch_scc0 X | <read> | <rewrite> | s_add_u32 s22, s22, 1
//      s_branch L
//   X: <tail>
namespace {
// s_mov_b64 vcc, s[16:17] | v_cndmask_b32_e64 v3, 0, 1.0, vcc
const Words kBlanketLaneRead = {0xbeea0410u, 0xd5010003u, 0x01a9e480u};
// v_mov_b32 v3, 1.0 | s_nop | s_nop: the same length, no lane-bit read of s[16:17]
const Words kNoLaneRead = {0x7e0602f2u, 0xbf800000u, 0xbf800000u};
// s_mov_b64 s[16:17], s[24:25]: s24/s25 are never written, so the back edge carries the zero
const Words kRewriteFabricated = {0xbe900418u};
// s_mov_b64 s[16:17], -1
const Words kRewriteDefined = {0xbe9004c1u};
// s_cmp_eq_u32 s0, 0 | s_cbranch_scc1 +2 | s_mov_b32 s16, -1 | s_mov_b32 s17, -1: rewritten on ONE
// path from a defined value, so the back edge merges the header phi with -1 (defined either way)
const Words kRewriteDefinedOnOnePath = {0xbf068000u, 0xbf850002u, 0xbe9003c1u, 0xbe9103c1u};
// store v3 (compute) or export v0..v3 with v0..v2 = 0 (fragment)
const Words kStoreV3 = {0xe0702000u, 0x80020300u, 0xbf810000u};
const Words kExportV3 = {0x7e000280u, 0x7e020280u, 0x7e040280u,
                         0xf800180fu, 0x03020100u, 0xbf810000u};

Words blanket_loop(Stage stage, const Words& read, const Words& rewrite, const Words& pre = {}) {
    const Words head = {0xbe9003c1u, 0xbe9103c1u, 0xbe960380u,
                        0xbe970383u, 0x7e060280u, 0xbf0a1716u};
    const auto body = static_cast<uint32_t>(read.size() + rewrite.size() + 2);   // +add +branch
    const Words exit_branch = {0xbf840000u | body};   // s_cbranch_scc0 X
    const Words latch = {0x80168116u, 0xbf820000u | ((0x10000u - (body + 2u)) & 0xffffu)};
    return cat({&kPrefix, &pre, &head, &exit_branch, &read, &rewrite, &latch,
                stage == Stage::Fragment ? &kExportV3 : &kStoreV3});
}
// s_mov_b32 s20, 7 | s_mov_b32 s21, 7: a second carried pair, defined before the loop
const Words kDefineS20S21 = {0xbe940387u, 0xbe950387u};
}   // namespace

TEST(ScalarPairMask, ALoopCarriedWordFabricatedOnTheBackEdgeRefusesItsLaneRead) {
    for (Stage stage : {Stage::Compute, Stage::Fragment}) {
        EXPECT_TRUE(
            compile_whole(stage, blanket_loop(stage, kBlanketLaneRead, kRewriteFabricated)).empty())
            << name(stage);
        const std::string reason = last_terminal_reject_reason(kAddress);
        EXPECT_NE(reason.find("fabricated on the back edge"), std::string::npos)
            << name(stage) << ": refused for another reason: " << reason;
    }
}

TEST(ScalarPairMask, ALoopCarriedWordDefinedOnTheBackEdgeKeepsItsLaneRead) {
    for (Stage stage : {Stage::Compute, Stage::Fragment})
        EXPECT_FALSE(
            compile_whole(stage, blanket_loop(stage, kBlanketLaneRead, kRewriteDefined)).empty())
            << name(stage) << ": " << last_terminal_reject_reason(kAddress);
}

TEST(ScalarPairMask, ALoopCarriedWordRewrittenOnOnePathFromDefinedDataKeepsItsLaneRead) {
    // The in-body if merges the header phi (blanket-only) with -1: still defined under the header's
    // assumption, so the merge keeps the blanket and the back edge honours it.
    for (Stage stage : {Stage::Compute, Stage::Fragment})
        EXPECT_FALSE(
            compile_whole(stage, blanket_loop(stage, kBlanketLaneRead, kRewriteDefinedOnOnePath))
                .empty())
            << name(stage) << ": " << last_terminal_reject_reason(kAddress);
}

TEST(ScalarPairMask, ALoopCarriedWordFabricatedOnTheBackEdgeButNeverReadAsLaneBitsCompiles) {
    for (Stage stage : {Stage::Compute, Stage::Fragment})
        EXPECT_FALSE(
            compile_whole(stage, blanket_loop(stage, kNoLaneRead, kRewriteFabricated)).empty())
            << name(stage) << ": " << last_terminal_reject_reason(kAddress);
}

namespace {
// Two nested counted loops. s[18:19] is written in the OUTER body by `outer_writes` (two words), then
// an inner loop reads it as VCC's lane bits and rewrites it with -1:
//   s_mov_b32 s22, 0 | s_mov_b32 s23, 3 | v_mov v3, 0
//   OL: s_cmp_lt_u32 s22, s23 | s_cbranch_scc0 OX
//       <outer_writes> | s_mov_b32 s20, 0
//       IL: s_cmp_lt_u32 s20, s23 | s_cbranch_scc0 IX
//           s_mov_b64 vcc, s[18:19] | v_cndmask_b32_e64 v3, 0, 1.0, vcc
//           s_mov_b64 s[18:19], -1 | s_add_u32 s20, s20, 1 | s_branch IL
//       IX: s_add_u32 s22, s22, 1 | s_branch OL
//   OX: <tail>
Words nested_loops(Stage stage, const Words& outer_writes) {
    const Words head = {0xbe960380u, 0xbe970383u, 0x7e060280u, 0xbf0a1716u, 0xbf84000du};
    const Words inner = {0xbe940380u, 0xbf0a1714u, 0xbf840006u, 0xbeea0412u,
                         0xd5010003u, 0x01a9e480u, 0xbe9204c1u, 0x80148114u,
                         0xbf82fff8u, 0x80168116u, 0xbf82fff1u};
    return cat({&kPrefix, &head, &outer_writes, &inner,
                stage == Stage::Fragment ? &kExportV3 : &kStoreV3});
}
}   // namespace

TEST(ScalarPairMask, AnInnerLoopWordDefinedAtItsPreheaderIsNotAbsentFromTheOuterSeed) {
    // s_mov_b32 s18, -1 | s_mov_b32 s19, -1. s[18:19] is written in the outer body, so the outer
    // header seeds it absent; the inner preheader defines it, so the inner header must blanket-mark
    // it (only that header's absent seeds count) and its lane-bit read stays admitted (cc4d1614c).
    const Words defined = {0xbe9203c1u, 0xbe9303c1u};
    for (Stage stage : {Stage::Compute, Stage::Fragment})
        EXPECT_FALSE(compile_whole(stage, nested_loops(stage, defined)).empty())
            << name(stage) << ": " << last_terminal_reject_reason(kAddress);
}

TEST(ScalarPairMask, AnInnerLoopWordDerivedFromTheOuterCounterIsDefinedAtItsPreheader) {
    // s_mov_b32 s18, s22 | s_mov_b32 s19, s22: the outer counter is blanket-only, so s[18:19] is
    // defined under the outer header's assumption (which the outer back edge checks), and the inner
    // header blanket-marks it too. GTA V's workgroup-store kernel has this shape with VCC scratch.
    const Words from_counter = {0xbe920316u, 0xbe930316u};
    for (Stage stage : {Stage::Compute, Stage::Fragment})
        EXPECT_FALSE(compile_whole(stage, nested_loops(stage, from_counter)).empty())
            << name(stage) << ": " << last_terminal_reject_reason(kAddress);
}

TEST(ScalarPairMask, ABottomTestedLoopChecksTheBlanketAtItsBackEdgeToo) {
    // The same read and rewrite in a bottom-tested loop, which the divergent-loop emitter lowers:
    //   s_mov_b32 s16, -1 | s_mov_b32 s17, -1 | s_mov_b32 s22, 0 | v_mov v3, 0
    //   L: s_mov_b64 vcc, s[16:17] | v_cndmask_b32_e64 v3, 0, 1.0, vcc | <rewrite>
    //      s_add_u32 s22, s22, 1 | s_cmp_lt_u32 s22, 3 | s_cbranch_scc1 L
    const auto loop = [](Stage stage, const Words& rewrite) {
        const Words head = {0xbe9003c1u, 0xbe9103c1u, 0xbe960380u, 0x7e060280u};
        const Words latch = {0x80168116u, 0xbf0a8316u, 0xbf85fff9u};
        return cat({&kPrefix, &head, &kBlanketLaneRead, &rewrite, &latch,
                    stage == Stage::Fragment ? &kExportV3 : &kStoreV3});
    };
    for (Stage stage : {Stage::Compute, Stage::Fragment}) {
        EXPECT_FALSE(compile_whole(stage, loop(stage, kRewriteDefined)).empty())
            << name(stage) << ": " << last_terminal_reject_reason(kAddress);
        EXPECT_TRUE(compile_whole(stage, loop(stage, kRewriteFabricated)).empty()) << name(stage);
        const std::string reason = last_terminal_reject_reason(kAddress);
        EXPECT_NE(reason.find("fabricated on the back edge"), std::string::npos)
            << name(stage) << ": refused for another reason: " << reason;
    }
}

TEST(ScalarPairMask, AnUnrelatedCarriedWordFabricatedOnTheBackEdgeDoesNotVoidTheRead) {
    // The loop reads s[16:17] as lane bits, rewrites it with -1 (so it is carried and its read is
    // admitted on the header's blanket), and rewrites s20 from the never-written s24. s20's
    // assumption fails, but no read depended on it: assumptions are per (header, register).
    const Words rewrite = {0xbe9004c1u,
                           0xbe940318u};   // s_mov_b64 s[16:17], -1 | s_mov_b32 s20, s24
    for (Stage stage : {Stage::Compute, Stage::Fragment})
        EXPECT_FALSE(
            compile_whole(stage, blanket_loop(stage, kBlanketLaneRead, rewrite, kDefineS20S21))
                .empty())
            << name(stage) << ": " << last_terminal_reject_reason(kAddress);
}

TEST(ScalarPairMask, ACarriedWordCopiedFromAViolatedOneIsViolatedToo) {
    // s_mov_b64 s[16:17], s[20:21] | s_mov_b64 s[20:21], s[24:25]: s[16:17] leaves the body
    // blanket-marked, but on s20/s21's assumption, which the never-written s[24:25] breaks. The next
    // trip's read of s[16:17] holds that zero, so the read is void.
    const Words rewrite = {0xbe900414u, 0xbe940418u};
    for (Stage stage : {Stage::Compute, Stage::Fragment}) {
        EXPECT_TRUE(
            compile_whole(stage, blanket_loop(stage, kBlanketLaneRead, rewrite, kDefineS20S21))
                .empty())
            << name(stage);
        const std::string reason = last_terminal_reject_reason(kAddress);
        EXPECT_NE(reason.find("fabricated on the back edge"), std::string::npos)
            << name(stage) << ": refused for another reason: " << reason;
    }
}

TEST(ScalarPairMask, ExecCmovSelectsOnlyOnADefiniteScc) {
    // s_cmov_b64 exec, -1 after s_add_u32 s20, s4, 1, which writes SCC from s4 (#4819 review). With
    // s4 written on one arm of an if, SCC carries the merge mark and must not select EXEC; with s4
    // written on both arms, it is definite and the program compiles.
    const Words body = {0x80148104u, 0xbefe06c1u, 0x7e0602f2u};
    EXPECT_FALSE(
        compile_whole(Stage::Compute, cat({&kPrefix, &kBothPaths, &body, &kStoreV3})).empty())
        << "control: a definite SCC selects: " << last_terminal_reject_reason(kAddress);
    EXPECT_TRUE(compile_whole(Stage::Compute, cat({&kPrefix, &kOnePath, &body, &kStoreV3})).empty())
        << "a merge-marked SCC must not select EXEC";
}

TEST(ScalarPairMask, ExecSourceIsAdmittedAsLaneMask) {
    // EXEC (126/127) is the live lane mask, not scalar data, so copying it into VCC
    // (`s_mov_b64 vcc, exec`: 0xbeea047eu) must not reach the fabricated-word check. A pin, not a
    // regression: before #4841 EXEC was admitted too, as 126/127 lay outside the special-data
    // range. A fabricated word cannot reach EXEC in the first place; the restore tests above
    // (AFabricatedWordSpilledThroughALoopRefusesAtTheRestore and its siblings) refuse it there.
    const Words mov_vcc_exec = {0xbeea047eu};
    for (Stage stage : {Stage::Compute, Stage::Fragment}) {
        EXPECT_FALSE(compile(stage, program(kDefined, mov_vcc_exec)).empty())
            << name(stage)
            << ": s_mov_b64 vcc, exec must be admitted: " << last_terminal_reject_reason(kAddress);
    }
}

// #4808: Wave64 compute builds a 64-bit select in the VCC pair one dword at a time and reads the
// halves back as data -- `s_cselect_b32 vcc_hi, 1, 0; s_cselect_b32 vcc_lo, 1, vcc_hi`, the shape
// of PPSA28000's cs 0x258009de00 and PPSA29343's matching programs. VCC_HI is a scalar DATA word
// there, so the second select is an ordinary dword select. Removing the is_wave64_vcc_lo_cselect_reading_vcc_hi
// disjunct from emit_alu's Wave64 VCC_LO cselect gate turns the first arm red.
TEST(ScalarPairMask, Wave64VccLoCselectReadingScalarVccHiCompiles) {
    // s_mov_b32 s0,0 | s_cmp_eq_u32 s0,0 | s_cselect_b32 vcc_hi,1,0 | s_cselect_b32 vcc_lo,1,vcc_hi
    const Words body = {0xbe800380u, 0xbf068000u, 0x856b8081u, 0x856a6b81u};
    EXPECT_FALSE(compile_vcc_as_data(Stage::Compute, body).empty())
        << "VCC_HI holds a scalar dword; selecting it into VCC_LO is plain scalar data";
    // Both operand orders: `s_cselect_b32 vcc_lo, vcc_hi, 64`.
    const Words swapped = {0xbe800380u, 0xbf068000u, 0x856b8081u, 0x856ac06bu};
    EXPECT_FALSE(compile_vcc_as_data(Stage::Compute, swapped).empty());
}

TEST(ScalarPairMask, Wave64VccLoCselectReadingMaskVccHiStillRefuses) {
    // v_cmp_eq_u32 vcc, 0, v0 leaves VCC_HI a MASK half, which has no scalar dword to select:
    // s_mov_b32 s0,0 | v_cmp_eq_u32 vcc,0,v0 | s_cmp_eq_u32 s0,0 | s_cselect_b32 vcc_lo,1,vcc_hi
    const Words body = {0xbe800380u, 0x7d840080u, 0xbf068000u, 0x856a6b81u};
    EXPECT_TRUE(compile_vcc_as_data(Stage::Compute, body).empty())
        << "a mask-domain VCC_HI must not be read back as a scalar dword";
}

// v_mov_b32 v3, vcc_lo | buffer_store_dword v3 -> out[x] | s_endpgm (compile_vcc_as_data's tail).
const Words kVccLoStoreTail = {0x7e06026au, 0xe0702000u, 0x80020300u, 0xbf810000u};   // NOLINT

TEST(ScalarPairMask, Wave64VccLoCselectReadingVccHiAcrossADispatcherEdgeCompiles) {
    // The same select with VCC_HI written in an EARLIER block of a program that only the CFG
    // dispatcher emits (a portable v_readlane, as in test_entry_m0_dispatcher). Each dispatcher block
    // reloads its state with placeholders, so acceptance needs the Wave64 record pass to prove the
    // pair complete at the select, as in PPSA28000 cs 0x29800f9600 pc14. Removing
    // is_wave64_vcc_lo_cselect_reading_vcc_hi from emit_cfg's b32_vcc_scalar_write turns this red.
    // s_mov s15,7 | s_lshr_b32 vcc_hi,s15,1 | s_cmp_eq_u32 s0,0 | s_cbranch_scc0 +1 | s_mov s1,5 |
    // v_readlane_b32 s2,v0,0 | s_cmp_eq_u32 s15,7 | s_cselect_b32 vcc_lo,64,vcc_hi
    const Words body = {0xbe8f0387u, 0x906b810fu, 0xbf068000u, 0xbf840001u, 0xbe810385u,
                        0xd7600002u, 0x00010100u, 0xbf06870fu, 0x856a6bc0u};
    // No native subgroup: the readlane is portable, which routes the stream to the dispatcher.
    const Words code = cat({&kPrefix, &body, &kVccLoStoreTail});
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = 0;
    const ShaderResourceTable table = output_table();
    EXPECT_FALSE(recompile_compute(code.data(), code.size(), &table, config,
                                   {RecompileDiagnosticStage::Compute, kAddress})
                     .empty());
}

TEST(ScalarPairMask, Wave64VccLoCselectFromAOnePathVccHiRefusesItsLaneRead) {
    // VCC_HI written on ONE path only is the structured emitter's fabricated zero at the select
    // (#4714). As DATA it is VCC-as-scratch, like any one-path word; what must never happen is the
    // selected pair being read back as this lane's VCC bit. kComputeTail is that lane read.
    // kOnePath's if, writing VCC_HI instead: s_cmp_eq_u32 s0,0 | s_cbranch_scc1 +1 |
    // s_mov_b32 vcc_hi,5 ; then kScc and s_cselect_b32 vcc_lo,1,vcc_hi.
    const Words one_path_vcc_hi = {0xbf068000u, 0xbf850001u, 0xbeeb0385u};
    const Words both_paths_vcc_hi = {0xbeeb0385u};
    const Words select = {0x856a6b81u};
    EXPECT_TRUE(compile(Stage::Compute, program(one_path_vcc_hi, select)).empty())
        << "a merge-marked VCC_HI must not reach a VCC lane read through the #4808 select";
    // Control: VCC_HI written unconditionally, the same select and lane read compile.
    EXPECT_FALSE(compile(Stage::Compute, program(both_paths_vcc_hi, select)).empty())
        << "a defined scalar VCC_HI selects and projects: "
        << last_terminal_reject_reason(kAddress);
}
