// ADR 0028 migration step 1: the `s_cbranch_execz` any-vote certificate, under today's ProvenVotes
// policy. Two halves that must agree: the GUEST-level classifier of the skipped region (scalar
// live-out, scalar memory effect, wave-level side effect, foreign exit) and the SPIR-V proof that
// consumes its verdict. Positive controls are hand-built here, outside the code that admits them.
#include <gtest/gtest.h>
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_exec_skip_region.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/spirv_fragment_neutral_selection.hpp"
#include "gpu/recompiler/spirv_fragment_vote_lowering.hpp"
#include "../../fixtures/spirv_fragment_neutral_fixtures.hpp"
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace prosper::gpu;
namespace f = prosper::test::fragment_neutral;
namespace base = prosper::test::fragment_votes;

constexpr uint32_t kVote = 60; // the OpGroupNonUniformAny result id in make_module

std::vector<uint32_t> with_marker(std::vector<uint32_t> words, const std::string& text) {
    std::vector<uint32_t> metadata;
    base::marker(metadata, text);
    base::insert(words, base::find(words, 71), metadata);
    return words;
}
std::vector<uint32_t> marked(f::Shape shape) {
    return with_marker(f::make_module(shape),
                       "Prosper.FragmentExecSkipVote=" + std::to_string(kVote));
}

// ---- SPIR-V half ---------------------------------------------------------------------------

TEST(FragmentExecSkipVote, MaskedVaryingStateIsAdmittedOnlyWithGuestEvidence) {
    // Positive arm: the region does VALU work only. Every lane's state is rewritten through a
    // Select on the vote's own predicate, so under P=false the executed value IS the skipped value.
    const auto with = lower_fragment_votes(marked(f::Shape::VaryingMasked));
    EXPECT_EQ(with.refusal, FragmentVoteRefusal::None);
    EXPECT_EQ(with.neutral_votes, 1u);
    EXPECT_TRUE(!with.words.empty());
    EXPECT_EQ(base::find(with.words, 335), with.words.size()) << "the Any is rewritten away";

    // Red-without-the-guest-evidence: the identical module with no marker is today's refusal. The
    // old proof needs a constant or frozen-leaf skipped value, and a varying one is neither.
    const auto without = lower_fragment_votes(f::make_module(f::Shape::VaryingMasked));
    EXPECT_EQ(without.refusal, FragmentVoteRefusal::UnprovedVote);
    EXPECT_TRUE(without.words.empty());
}

TEST(FragmentExecSkipVote, SgprLiveOutReadAfterMergeIsRefused) {
    // ADR mutation arm, by name. A value computed in the skipped region reaches a merge Phi that
    // is read afterwards: the executed value is not the skipped one, so forcing the vote TRUE would
    // publish a scalar write the all-false wave never made. Refused even WITH guest evidence.
    const auto lowered = lower_fragment_votes(marked(f::Shape::SgprLiveOut));
    EXPECT_EQ(lowered.refusal, FragmentVoteRefusal::UnprovedVote);
    EXPECT_TRUE(lowered.words.empty());
    EXPECT_EQ(lowered.neutral_votes, 0u);
}

TEST(FragmentExecSkipVote, MaskNotTiedToTheVotePredicateIsRefused) {
    // Masking by some OTHER condition does not make the region neutral under P=false.
    const auto lowered = lower_fragment_votes(marked(f::Shape::UnrelatedMask));
    EXPECT_EQ(lowered.refusal, FragmentVoteRefusal::UnprovedVote);
}

TEST(FragmentExecSkipVote, EvidenceMustNameThisVoteAndBeWellFormed) {
    const auto module = f::make_module(f::Shape::VaryingMasked);
    const char* refused[] = {
        "Prosper.FragmentExecSkipVote=61",   // a different id
        "Prosper.FragmentExecSkipVote=0",   // no such result
        "Prosper.FragmentExecSkipVote=",   // empty
        "Prosper.FragmentExecSkipVote=60x",   // trailing garbage
        "Prosper.FragmentExecSkipVote=-60",   // sign
        "Prosper.FragmentExecSkipVote=060",   // leading zero: not the canonical spelling
        "Prosper.FragmentExecSkipVote= 60",   // space
        "Prosper.FragmentExecSkipVote=99999999999",   // overflow
        "Prosper.FragmentExecSkipVoteX=60",   // a different key
    };
    for (const char* text : refused) {
        const auto lowered = lower_fragment_votes(with_marker(module, text));
        EXPECT_EQ(lowered.refusal, FragmentVoteRefusal::UnprovedVote) << text;
    }
    // Malformed evidence is ignored, not a contract error: it can only cost an admission.
    EXPECT_EQ(lower_fragment_votes(with_marker(module, "Prosper.FragmentExecSkipVote=60")).refusal,
              FragmentVoteRefusal::None);
}

TEST(FragmentExecSkipVote, EvidenceDoesNotWidenTheOlderCertificate) {
    // The evidence never replaces the neutral proof: a body that is not UB-free is still refused
    // with it present, and the older constant/atom exports behave as before.
    for (const auto shape :
         {f::Shape::BodyStore, f::Shape::DeadLoad, f::Shape::DeadDivide, f::Shape::DeadDerivative,
          f::Shape::ScalarExport, f::Shape::LiveUnequalPhi}) {
        const auto lowered = lower_fragment_votes(marked(shape));
        EXPECT_EQ(lowered.refusal, FragmentVoteRefusal::UnprovedVote) << static_cast<int>(shape);
    }
    EXPECT_EQ(lower_fragment_votes(marked(f::Shape::Masked)).refusal, FragmentVoteRefusal::None);
}

TEST(FragmentExecSkipVote, GraphProofRequiresTheFlagForVaryingIdentity) {
    using Type = FragmentNeutralType;
    // %10 = P, %20 varying skipped state, %21 = Select(P, new, %20), %22 = new.
    FragmentNeutralSelection graph;
    graph.predicate = 10;
    graph.values = {
        {10, {Type::Boolean, 61, 0, {}}},
        {20, {Type::Int32, 61, 0, {}}},
        {22, {Type::Int32, 128, 0, {20, 20}}},
        {21, {Type::Int32, 169, 0, {10, 22, 20}}},
    };
    graph.body = {22, 21};
    graph.exports = {{20, 21}};
    const std::unordered_set<uint32_t> frozen;
    EXPECT_FALSE(prove_fragment_neutral_selection(graph, frozen));
    graph.guest_scalar_effects_absent = true;
    EXPECT_TRUE(prove_fragment_neutral_selection(graph, frozen));
    // Arithmetic is never identity, flag or not.
    graph.exports = {{20, 22}};
    EXPECT_FALSE(prove_fragment_neutral_selection(graph, frozen));
    // A Select whose condition is not the predicate is not identity.
    graph.exports = {{20, 21}};
    graph.values[30] = {Type::Boolean, 61, 0, {}};
    graph.values[21].operands = {30, 22, 20};
    EXPECT_FALSE(prove_fragment_neutral_selection(graph, frozen));
}

// ---- guest half ----------------------------------------------------------------------------

Rdna2Inst decode(uint32_t word, uint32_t pc, Rdna2Format expected_fmt, uint32_t expected_opcode) {
    Rdna2Inst in = rdna2_decode_one(&word, 1);
    EXPECT_EQ(in.fmt, expected_fmt) << std::hex << word;
    EXPECT_EQ(in.opcode, expected_opcode) << std::hex << word;
    in.pc = pc;
    return in;
}
Operand sgpr(int n) {
    return {OperandKind::SGPR, n};
}
Rdna2Inst decode2(uint32_t word0, uint32_t word1, uint32_t pc, Rdna2Format expected_fmt,
                  uint32_t expected_opcode) {
    const uint32_t words[2] = {word0, word1};
    Rdna2Inst in = rdna2_decode_one(words, 2);
    EXPECT_EQ(in.fmt, expected_fmt) << std::hex << word0;
    EXPECT_EQ(in.opcode, expected_opcode) << std::hex << word0;
    EXPECT_EQ(in.len_dwords, 2u) << std::hex << word0;
    in.pc = pc;
    return in;
}

constexpr uint32_t kBranch = 0xBF880000u;   // s_cbranch_execz
constexpr uint32_t kEnd = 0xBF810000u;   // s_endpgm
constexpr uint32_t kNop = 0xBF800000u;
constexpr uint32_t kVAdd = 0x06020702u;   // v_add_f32 v1, v2, v3
constexpr uint32_t kVMov = 0x7E020302u;   // v_mov_b32 v1, v2
constexpr uint32_t kVCmp = 0x7C020501u;   // v_cmp_lt_f32 vcc, v1, v2
constexpr uint32_t kVCmpx = 0x7C220501u;   // v_cmpx_lt_f32 vcc, v1, v2
constexpr uint32_t kReadFirstLaneS5 = 0x7E0A0501u;   // v_readfirstlane_b32 s5, v1
constexpr uint32_t kSMovS6S5 = 0xBE860305u;   // s_mov_b32 s6, s5
constexpr uint32_t kSMovS5Zero = 0xBE850380u;   // s_mov_b32 s5, 0
constexpr uint32_t kSAddU32 = 0x80050201u;   // s_add_u32 s5, s1, s2
constexpr uint32_t kSendMsg = 0xBF900000u;
constexpr uint32_t kBarrier = 0xBF8A0000u;
constexpr uint32_t kSleep = 0xBF8E0001u;
constexpr uint32_t kSBranch = 0xBF820001u;
// Encodings below are llvm-mc -mcpu=gfx1030 -mattr=+wavefrontsize64 round-trips, except the VOP3
// spelling of v_readfirstlane_b32 (0x182), which llvm-mc refuses to assemble and is built by hand
// from the VOP3 field layout (cf. v_mov_b32_e64 v5, v1 = 0xD5810005 0x00000101).
// v_cmp_lt_f32_e64 s[6:7], v1, v2
constexpr uint32_t kVCmpE64S6 = 0xD4010006u;
constexpr uint32_t kVCmpE64S6Hi = 0x00020501u;
constexpr uint32_t kSMovS8S6 = 0xBE880306u;   // s_mov_b32 s8, s6
constexpr uint32_t kSMovS6Zero = 0xBE860380u;   // s_mov_b32 s6, 0
constexpr uint32_t kSMovS7Zero = 0xBE870380u;   // s_mov_b32 s7, 0
constexpr uint32_t kVCndmaskE32 = 0x02020702u;   // v_cndmask_b32_e32 v1, v2, v3, vcc (reads VCC)
constexpr uint32_t kVAddS5 = 0x06040405u;   // v_add_f32_e32 v2, s5, v2
constexpr uint32_t kVCmpGtU32Region = 0x7d880284u;   // v_cmp_gt_u32 vcc, 4, v1

// pc 0: s_cbranch_execz -> merge; the region follows; the merge is `tail` then s_endpgm.
struct Program {
    std::vector<Rdna2Inst> ins;
    uint32_t target = 0;
};
Program program(const std::vector<Rdna2Inst>& region, const std::vector<Rdna2Inst>& tail = {}) {
    Program p;
    p.ins.push_back(
        decode(kBranch | static_cast<uint32_t>(region.size()), 0, Rdna2Format::SOPP, 0x08));
    uint32_t pc = 1;
    for (auto in : region) {
        in.pc = pc;
        pc += in.len_dwords;
        p.ins.push_back(in);
    }
    p.target = pc;
    for (auto in : tail) {
        in.pc = pc;
        pc += in.len_dwords;
        p.ins.push_back(in);
    }
    auto end = decode(kEnd, pc, Rdna2Format::SOPP, 0x01);
    p.ins.push_back(end);
    return p;
}
ExecSkipRegionEffects classify(const Program& p) {
    return classify_exec_skip_region(p.ins, 0, p.target);
}
Rdna2Inst vadd() {
    return decode(kVAdd, 0, Rdna2Format::VOP2, 0x03);
}
Rdna2Inst vmov() {
    return decode(kVMov, 0, Rdna2Format::VOP1, 0x01);
}
Rdna2Inst smov(uint32_t word) {
    return decode(word, 0, Rdna2Format::SOP1, 0x03);
}
// s_mov_b32 s6, 0 ; s_mov_b32 s7, 0 ; s_mov_b32 s8, s6: s[6:7] dead at the merge, then read.
std::vector<Rdna2Inst> redefine_s6_s7_then_read() {
    return {smov(kSMovS6Zero), smov(kSMovS7Zero), smov(kSMovS8S6)};
}
Rdna2Inst cndmask_reads_vcc() {
    return decode(kVCndmaskE32, 0, Rdna2Format::VOP2, 0x01);
}
Rdna2Inst smem(uint32_t opcode, int dst) {
    Rdna2Inst in;
    in.fmt = Rdna2Format::SMEM;
    in.opcode = opcode;
    in.dst = sgpr(dst);
    in.src[0] = sgpr(0);
    in.src[1] = {OperandKind::InlineInt, 0};
    in.n_src = 2;
    in.len_dwords = 1;
    return in;
}

TEST(FragmentExecSkipRegion, ValuOnlyRegionIsClean) {
    // Positive arm: VALU work only, including a v_cmp whose VCC result is dead at the merge.
    const auto cmp = decode(kVCmp, 0, Rdna2Format::VOPC, 0x01);
    EXPECT_TRUE(classify(program({vadd(), vmov()})).clean());
    EXPECT_TRUE(classify(program({decode(kNop, 0, Rdna2Format::SOPP, 0x00), vadd()})).clean());
    EXPECT_TRUE(classify(program({vadd(), cmp, vmov()})).clean())
        << "VCC is overwritten or unread after the merge";
}

TEST(FragmentExecSkipRegion, SgprLiveOutIsRefusedAndDeadOneIsNot) {
    // ADR mutation arm, by name: a scalar written in the region and read after the merge. The
    // writer is an e64 compare into s[6:7], which the decoder reclassifies as VOPC.
    const auto write = decode2(kVCmpE64S6, kVCmpE64S6Hi, 1, Rdna2Format::VOPC, 0x01);
    ASSERT_EQ(write.dst.kind, OperandKind::SGPR);
    ASSERT_EQ(write.dst.value, 6);
    auto fx = classify(program({vadd(), write}, {smov(kSMovS8S6)}));
    EXPECT_TRUE(fx.scalar_live_out);
    EXPECT_FALSE(fx.clean());
    // Control: the same write, redefined before any read, is not observable.
    EXPECT_TRUE(classify(program({vadd(), write}, redefine_s6_s7_then_read())).clean());
}

TEST(FragmentExecSkipRegion, ScalarAluAndExecWritesAreLiveOuts) {
    EXPECT_TRUE(classify(program({decode(kSAddU32, 0, Rdna2Format::SOP2, 0x00)})).scalar_live_out)
        << "SCC has no liveness proof, so any scalar ALU op is refused";
    EXPECT_TRUE(classify(program({decode(kVCmpx, 0, Rdna2Format::VOPC, 0x11)})).scalar_live_out)
        << "v_cmpx writes EXEC";
    // VCC read after the merge by a v_cndmask-style consumer: a v_cmp whose mask survives.
    const auto cmp = decode(kVCmp, 0, Rdna2Format::VOPC, 0x01);
    Rdna2Inst cndmask;
    cndmask.fmt = Rdna2Format::VOP2;
    cndmask.opcode = 0x01;
    cndmask.dst = {OperandKind::VGPR, 1};
    cndmask.src[0] = {OperandKind::VGPR, 2};
    cndmask.src[1] = {OperandKind::VGPR, 3};
    cndmask.n_src = 2;
    cndmask.len_dwords = 1;
    EXPECT_TRUE(classify(program({cmp}, {cndmask})).scalar_live_out);
}

TEST(FragmentExecSkipRegion, ScalarLoadReadAfterTheMergeIsRefused) {
    // A scalar load whose result is read after the merge.
    EXPECT_TRUE(classify(program({smem(0x8, 4)}, {smov(0xBE860304u)})).scalar_memory_effect)
        << "s_mov_b32 s6, s4 reads the loaded value";
    // Control: the loaded register is redefined before it is read.
    EXPECT_TRUE(classify(program({smem(0x8, 5)}, {smov(kSMovS5Zero), smov(kSMovS6S5)})).clean());
}

TEST(FragmentExecSkipRegion, OnlyPlainScalarLoadsAreCleanAcrossEverySmemOpcode) {
    // RDNA2 has no scalar store or atomic, so the scalar-memory condition is a whitelist of the
    // loads: s_load_dword{,x2,x4,x8,x16} 0x00-0x04 and s_buffer_load_* 0x08-0x0C. Everything else
    // in the 6-bit opcode space -- s_gl1_inv 0x1F, s_dcache_inv 0x20, s_memtime 0x24,
    // s_memrealtime 0x25, s_atc_probe{,_buffer} 0x26/0x27 and every undefined slot -- is refused,
    // with its destination dead (nothing follows the merge), so only the opcode decides.
    for (uint32_t op = 0; op <= 0x3F; ++op) {
        // word0: SDATA = s4, SBASE = s[0:1]; word1: SOFFSET = null, offset 0.
        const auto in =
            decode2(0xF4000000u | (op << 18) | (4u << 6), 0xFA000000u, 1, Rdna2Format::SMEM, op);
        const bool plain_load = op <= 0x04 || (op >= 0x08 && op <= 0x0C);
        const auto fx = classify(program({in}));
        EXPECT_EQ(fx.clean(), plain_load) << "SMEM opcode 0x" << std::hex << op;
        EXPECT_EQ(fx.scalar_memory_effect, !plain_load) << "SMEM opcode 0x" << std::hex << op;
    }
}

TEST(FragmentExecSkipRegion, WaveLevelSideEffectsAreRefused) {
    EXPECT_TRUE(
        classify(program({vadd(), decode(kSendMsg, 0, Rdna2Format::SOPP, 0x10)})).wave_side_effect);
    EXPECT_TRUE(classify(program({decode(kBarrier, 0, Rdna2Format::SOPP, 0x0a)})).wave_side_effect);
    EXPECT_TRUE(classify(program({decode(kSleep, 0, Rdna2Format::SOPP, 0x0e)})).wave_side_effect);
    Rdna2Inst setreg;
    setreg.fmt = Rdna2Format::SOPK;
    setreg.opcode = 0x13;
    setreg.len_dwords = 1;
    EXPECT_TRUE(classify(program({setreg})).wave_side_effect);
    Rdna2Inst dpp = vmov();
    dpp.has_dpp = true;
    EXPECT_TRUE(classify(program({dpp})).wave_side_effect) << "cross-lane read";
}

TEST(FragmentExecSkipRegion, ExitsOtherThanTheMergeAreRefused) {
    EXPECT_TRUE(classify(program({decode(kSBranch, 0, Rdna2Format::SOPP, 0x02)})).foreign_exit);
    EXPECT_TRUE(classify(program({decode(kEnd, 0, Rdna2Format::SOPP, 0x01)})).foreign_exit);
    EXPECT_TRUE(classify(program({decode(kBranch | 1u, 0, Rdna2Format::SOPP, 0x08)})).foreign_exit)
        << "a nested skip is not a fall-through to the merge";
}

TEST(FragmentExecSkipRegion, UnknownFormatsAndBackwardRegionsFailClosed) {
    Rdna2Inst buffer;
    buffer.fmt = Rdna2Format::MUBUF;
    buffer.len_dwords = 2;
    EXPECT_TRUE(classify(program({buffer})).unclassified);
    Rdna2Inst unknown;
    unknown.fmt = Rdna2Format::Unknown;
    EXPECT_TRUE(classify(program({unknown})).unclassified);
    auto p = program({vadd()});
    EXPECT_TRUE(classify_exec_skip_region(p.ins, p.target, 0).foreign_exit);
    EXPECT_TRUE(classify_exec_skip_region(p.ins, 0, 0).foreign_exit);
}

// ---- one arm per classifier condition, each with the control that isolates it --------------

TEST(FragmentExecSkipRegion, Vop2CarryOutToVccIsALiveOutOnlyWhenRead) {
    // v_add/sub/subrev_co_ci_u32_e32 (0x28-0x2A) write their carry-out to VCC implicitly; the
    // decoded operands name no scalar destination. Read after the merge by v_cndmask_b32_e32.
    for (const uint32_t word : {0x50020702u, 0x52020702u, 0x54020702u}) {
        const auto carry = decode(word, 0, Rdna2Format::VOP2, (word >> 25) & 0x3Fu);
        EXPECT_TRUE(classify(program({carry}, {cndmask_reads_vcc()})).scalar_live_out)
            << std::hex << word;
        // Control: a v_cmp kills the VCC pair before the same reader.
        EXPECT_TRUE(classify(program({carry}, {decode(kVCmp, 0, Rdna2Format::VOPC, 0x01),
                                               cndmask_reads_vcc()}))
                        .clean())
            << std::hex << word;
    }
}

TEST(FragmentExecSkipRegion, Vop3bScalarDestinationIsALiveOutOnlyWhenRead) {
    // VOP3B carry/flag SDST in s[6:7]: v_add_co_u32, v_mad_u64_u32, v_div_scale_f32.
    const struct {
        uint32_t lo, hi, opcode;
    } forms[] = {
        {0xD70F0601u, 0x00020702u, 0x30F},   // v_add_co_u32 v1, s[6:7], v2, v3
        {0xD5760601u, 0x04120702u, 0x176},   // v_mad_u64_u32 v[1:2], s[6:7], v2, v3, v[4:5]
        {0xD56D0601u, 0x04120702u, 0x16D},   // v_div_scale_f32 v1, s[6:7], v2, v3, v4
    };
    for (const auto& form : forms) {
        const auto in = decode2(form.lo, form.hi, 1, Rdna2Format::VOP3, form.opcode);
        ASSERT_EQ(in.sdst.kind, OperandKind::SGPR) << std::hex << form.opcode;
        ASSERT_EQ(in.sdst.value, 6) << std::hex << form.opcode;
        EXPECT_TRUE(classify(program({in}, {smov(kSMovS8S6)})).scalar_live_out)
            << std::hex << form.opcode;
        EXPECT_TRUE(classify(program({in}, redefine_s6_s7_then_read())).clean())
            << std::hex << form.opcode;
    }
}

TEST(FragmentExecSkipRegion, ReadFirstLaneIntoADeadSgprIsStillRefused) {
    // The failing scenario this certificate must not admit: VCC dead, s5 never read after the
    // merge, but s5 feeds an EXEC-masked v_add. Guest wave64 adds the FIRST active lane's v1 to
    // every lane; the fragment emitter adds each lane's own v1. Liveness of s5 is irrelevant.
    const auto readfirstlane = decode(kReadFirstLaneS5, 0, Rdna2Format::VOP1, 0x02);
    const auto fx = classify(program({decode(kVCmpGtU32Region, 0, Rdna2Format::VOPC, 0xC4),
                                      readfirstlane, decode(kVAddS5, 0, Rdna2Format::VOP2, 0x03)}));
    EXPECT_TRUE(fx.wave_side_effect);
    EXPECT_FALSE(fx.scalar_live_out) << "nothing after the merge reads s5 or VCC";
    EXPECT_FALSE(fx.clean());
    // Control: the same region with an ordinary move in place of the cross-lane read is clean.
    EXPECT_TRUE(
        classify(program({decode(kVCmpGtU32Region, 0, Rdna2Format::VOPC, 0xC4), vmov(), vadd()}))
            .clean());
}

TEST(FragmentExecSkipRegion, Vop3LaneAccessIsRefusedWhateverItsDestination) {
    // VOP3 0x182 v_readfirstlane_b32 and 0x360 v_readlane_b32 read another lane; 0x361
    // v_writelane_b32 writes one lane with EXEC ignored. Every destination is dead here.
    const struct {
        uint32_t lo, hi, opcode;
    } forms[] = {
        {0xD5820005u, 0x00000101u, 0x182},   // v_readfirstlane_b32 s5, v1 (VOP3, hand-encoded)
        {0xD7600005u, 0x00010701u, 0x360},   // v_readlane_b32 s5, v1, 3
        {0xD7610001u, 0x00010605u, 0x361},   // v_writelane_b32 v1, s5, 3
    };
    for (const auto& form : forms) {
        const auto in = decode2(form.lo, form.hi, 1, Rdna2Format::VOP3, form.opcode);
        const auto fx = classify(program({in}));
        EXPECT_TRUE(fx.wave_side_effect) << std::hex << form.opcode;
        EXPECT_FALSE(fx.clean()) << std::hex << form.opcode;
    }
    // Control: the neighbouring VOP3 opcode 0x181, v_mov_b32_e64 v5, v1, is clean.
    EXPECT_TRUE(classify(program({decode2(0xD5810005u, 0x00000101u, 1, Rdna2Format::VOP3, 0x181)}))
                    .clean());
}

TEST(FragmentExecSkipRegion, UnmodelledVopModifiersAreWaveSideEffects) {
    // VOP2 DPP16 (row/quad permutations read other lanes).
    const auto dpp = decode2(0x060206FAu, 0xFF00E402u, 1, Rdna2Format::VOP2, 0x03);
    ASSERT_TRUE(dpp.has_dpp || dpp.has_modifier);
    EXPECT_TRUE(classify(program({dpp})).wave_side_effect) << "v_add_f32_dpp quad_perm";
    EXPECT_TRUE(classify(program({vadd()})).clean()) << "control: v_add_f32_e32";
    // VOPC SDWA with a float WORD_1 select: a form the decoder leaves has_modifier on.
    const auto sdwa = decode2(0x7C0204F9u, 0x06050001u, 1, Rdna2Format::VOPC, 0x01);
    ASSERT_TRUE(sdwa.has_modifier);
    EXPECT_TRUE(classify(program({sdwa})).wave_side_effect) << "v_cmp_lt_f32_sdwa WORD_1";
    EXPECT_TRUE(classify(program({decode(kVCmp, 0, Rdna2Format::VOPC, 0x01)})).clean())
        << "control: v_cmp_lt_f32_e32";
    // VOP3: gfx10 VOP3 has no DPP/SDWA spelling, so the decoder sets neither flag today. Pin the
    // fail-closed rule anyway, on a real v_fma_f32 with the flag set, against a future decoder.
    auto fma = decode2(0xD54B0001u, 0x04120702u, 1, Rdna2Format::VOP3, 0x14B);
    EXPECT_TRUE(classify(program({fma})).clean()) << "control: v_fma_f32 v1, v2, v3, v4";
    fma.has_modifier = true;
    EXPECT_TRUE(classify(program({fma})).wave_side_effect);
    fma.has_modifier = false;
    fma.has_dpp = true;
    EXPECT_TRUE(classify(program({fma})).wave_side_effect);
}

TEST(FragmentExecSkipRegion, InstructionsThatLeaveTheDecodedCfgAreForeignExits) {
    // rdna2_escapes_decoded_effects: without that check each of these would read only as scalar
    // ALU (SOP1) or a scalar write (SOPK), so the arm asserts the foreign exit specifically.
    EXPECT_TRUE(classify(program({decode(0xBE802004u, 0, Rdna2Format::SOP1, 0x20)})).foreign_exit)
        << "s_setpc_b64 s[4:5]";
    EXPECT_TRUE(classify(program({decode(0xBE853006u, 0, Rdna2Format::SOP1, 0x30)})).foreign_exit)
        << "s_movreld_b32 s5, s6";
    EXPECT_TRUE(classify(program({decode(0xBD840001u, 0, Rdna2Format::SOPK, 0x1B)})).foreign_exit)
        << "s_subvector_loop_begin s4, 1";
    // Control: an ordinary SOP1 move is a scalar live-out, not an exit.
    const auto fx = classify(program({smov(kSMovS6Zero)}));
    EXPECT_FALSE(fx.foreign_exit);
    EXPECT_TRUE(fx.scalar_live_out);
}

TEST(FragmentExecSkipRegion, WaitcntVscntIsAWaveSideEffectAndCounterWaitsAreNot) {
    Rdna2Inst vscnt = decode(0xBBFD0000u, 0, Rdna2Format::SOPK, 0x17);   // s_waitcnt_vscnt null, 0
    const auto fx = classify(program({vscnt}));
    EXPECT_TRUE(fx.wave_side_effect);
    EXPECT_FALSE(fx.scalar_live_out) << "refused as a barrier, not as a scalar write";
    // Control: s_waitcnt_lgkmcnt null, 0 is a counter the synchronous model never waits on.
    EXPECT_TRUE(classify(program({decode(0xBD7D0000u, 0, Rdna2Format::SOPK, 0x1A)})).clean());
}

TEST(FragmentExecSkipRegion, Sop1AndSopcAreScalarLiveOutsEvenWithADeadDestination) {
    // SCC has no liveness proof, so a dead SGPR destination does not clear a scalar ALU op.
    const auto smov_dead = classify(program({smov(kSMovS6Zero)}));   // s6 never read
    EXPECT_TRUE(smov_dead.scalar_live_out);
    EXPECT_FALSE(smov_dead.unclassified);
    const auto scmp = classify(program({decode(0xBF060201u, 0, Rdna2Format::SOPC, 0x06)}));
    EXPECT_TRUE(scmp.scalar_live_out) << "s_cmp_eq_u32 s1, s2 writes SCC";
    EXPECT_FALSE(scmp.unclassified);
}

TEST(FragmentExecSkipRegion, LdsIsUnclassified) {
    // ds_read_b32 v1, v2
    const auto ds = decode2(0xD8D80000u, 0x01000002u, 1, Rdna2Format::DS, 0x36);
    EXPECT_TRUE(classify(program({ds})).unclassified);
}

// ---- emitter half: the recompiler's own evidence ---------------------------------------------

// Wave64 pixel shader: EXEC narrowed by a v_cmpx on the interpolation input, an execz skip over
// `region`, then EXEC restored and an MRT0 export. v0 enters as the barycentric `i`, a varying
// value; the vote takes no lane-id (MBCNT) dependency, so the module's wave contract is the
// vote alone, which is the only contract lower_fragment_votes admits.
std::vector<uint32_t> execz_shader(const std::vector<uint32_t>& region) {
    std::vector<uint32_t> code = {
        0xbe80047eu,   // s_mov_b64 s[0:1], exec
        0x7e0202f2u,   // v_mov_b32 v1, 1.0
        0x7e040280u,   // v_mov_b32 v2, 0
        0x7e0602f2u,   // v_mov_b32 v3, 1.0
        0x7da800f0u,   // v_cmpx_gt_u32 0.5(inline), v0 : EXEC &= varying
        0xbf880000u | static_cast<uint32_t>(region.size()),   // s_cbranch_execz merge
    };
    code.insert(code.end(), region.begin(), region.end());
    code.insert(code.end(), {
                                0xbefe0400u,   // s_mov_b64 exec, s[0:1]
                                0xf800180fu,
                                0x03020100u,   // exp mrt0 v0, v1, v2, v3 done vm
                                0xbf810000u,   // s_endpgm
                            });
    return code;
}
struct Evidence {
    std::vector<uint32_t> votes, marked;
};
Evidence evidence(const std::vector<uint32_t>& spirv) {
    Evidence out;
    for (size_t at = 5; at < spirv.size(); at += spirv[at] >> 16) {
        const uint32_t op = spirv[at] & 0xffffu;
        if (op == 335) out.votes.push_back(spirv[at + 2]);
        if (op != 330) continue;
        const char* text = reinterpret_cast<const char*>(&spirv[at + 1]);
        const std::string_view view(text);
        constexpr std::string_view prefix = "Prosper.FragmentExecSkipVote=";
        if (view.starts_with(prefix))
            out.marked.push_back(
                static_cast<uint32_t>(std::stoul(std::string(view.substr(prefix.size())))));
    }
    return out;
}

// A region of only EXEC-predicated VALU is linearized by the recompiler with no vote at all, so a
// vote exists only for a region the older linearizer refuses. A v_cmp is that: its VCC write is not
// EXEC-predicated. The classifier accepts it because VCC is dead at the merge.
constexpr uint32_t kVCmpGtU32 = 0x7d880284u;   // v_cmp_gt_u32 vcc, 4, v1
constexpr uint32_t kVMovOne = 0x7e0002f2u;   // v_mov_b32 v0, 1.0

Evidence compile_evidence(const std::vector<uint32_t>& region) {
    const auto code = execz_shader(region);
    const auto spirv = recompile_fragment(code.data(), code.size());
    EXPECT_FALSE(spirv.empty());
    return evidence(spirv);
}

TEST(FragmentExecSkipEmitter, CleanRegionCarriesEvidenceForItsOwnVote) {
    const auto e = compile_evidence({kVCmpGtU32, kVMovOne});
    ASSERT_EQ(e.votes.size(), 1u);
    EXPECT_EQ(e.marked, e.votes) << "the evidence names the vote that guards the execz skip";
}

TEST(FragmentExecSkipEmitter, RegionWithAnInvisibleWaveEffectCarriesNone) {
    // s_sendmsg is lowered to nothing in a pixel shader, so the SPIR-V cannot show it. Only the
    // guest-level classifier can tell, and it must withhold the evidence.
    const auto e = compile_evidence({kVCmpGtU32, 0xbf900000u /* s_sendmsg */, kVMovOne});
    ASSERT_EQ(e.votes.size(), 1u);
    EXPECT_TRUE(e.marked.empty());
}

TEST(FragmentExecSkipEmitter, RegionWithScalarWorkCarriesNone) {
    // s_mov_b32 s5, 7 followed by the merge reading s5 would be an SGPR live-out; here it is simply
    // scalar ALU, which the VALU-only certificate does not cover.
    const auto e = compile_evidence({kVCmpGtU32, 0xbe850387u /* s_mov_b32 s5, 7 */, kVMovOne});
    ASSERT_EQ(e.votes.size(), 1u);
    EXPECT_TRUE(e.marked.empty());
}

TEST(FragmentExecSkipEmitter, RegionWithReadFirstLaneCarriesNone) {
    // The classifier's readfirstlane refusal, end to end: VCC is dead and s5 is never read after
    // the merge, but s5 feeds an EXEC-masked v_add whose value is per-lane on the host.
    const auto e = compile_evidence({kVCmpGtU32, kReadFirstLaneS5, kVAddS5});
    ASSERT_EQ(e.votes.size(), 1u);
    EXPECT_TRUE(e.marked.empty());
}

// Cross-lane work the classifier does not itself refuse rests on lower_fragment_votes' exact
// `Prosper.FragmentSubgroupWhy=2` contract: the evidence names the vote, and the module is still
// refused because the emitter recorded a lane-id reason beside the vote.
TEST(FragmentExecSkipEmitter, LaneIdInTheRegionIsRefusedDespiteTheEvidence) {
    const auto lower = [](const std::vector<uint32_t>& region) {
        const auto code = execz_shader(region);
        return lower_fragment_votes(recompile_fragment(code.data(), code.size()));
    };
    // v_mbcnt_lo_u32_b32 v1, -1, 0: a lane id the 32-lane host numbers differently.
    const std::vector<uint32_t> mbcnt = {kVCmpGtU32, 0xD7650001u, 0x000100C1u, kVMovOne};
    const auto e = compile_evidence(mbcnt);
    ASSERT_EQ(e.votes.size(), 1u);
    EXPECT_EQ(e.marked, e.votes) << "the guest classifier does not see lane ids";
    const auto lowered = lower(mbcnt);
    EXPECT_EQ(lowered.refusal, FragmentVoteRefusal::InconsistentContract);
    EXPECT_TRUE(lowered.words.empty());
    // v_readlane_b32 s5, v1, 40: a lane the 32-lane host does not have. Refused twice over.
    const std::vector<uint32_t> readlane = {kVCmpGtU32, 0xD7600005u, 0x00015101u, kVMovOne};
    EXPECT_TRUE(compile_evidence(readlane).marked.empty());
    EXPECT_NE(lower(readlane).refusal, FragmentVoteRefusal::None);
    // Control: the clean region, carrying the same evidence, lowers.
    EXPECT_EQ(lower({kVCmpGtU32, kVMovOne}).refusal, FragmentVoteRefusal::None);
}
}   // namespace
