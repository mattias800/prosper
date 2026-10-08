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

TEST(FragmentExecSkipVote, ScalarMemoryStoreInTheRegionIsRefused) {
    // ADR mutation arm, by name. The state is correctly masked, but the region also stores.
    const auto lowered = lower_fragment_votes(marked(f::Shape::ScalarStore));
    EXPECT_EQ(lowered.refusal, FragmentVoteRefusal::UnprovedVote);
    EXPECT_TRUE(lowered.words.empty());
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
    // ADR mutation arm, by name: a scalar written in the region and read after the merge.
    const auto write = decode(kReadFirstLaneS5, 0, Rdna2Format::VOP1, 0x02);
    auto fx = classify(program({vadd(), write}, {smov(kSMovS6S5)}));
    EXPECT_TRUE(fx.scalar_live_out);
    EXPECT_FALSE(fx.clean());
    // Control: the same write, redefined before any read, is not observable.
    EXPECT_TRUE(classify(program({vadd(), write}, {smov(kSMovS5Zero), smov(kSMovS6S5)})).clean());
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

TEST(FragmentExecSkipRegion, ScalarMemoryEffectIsRefused) {
    // ADR mutation arm, by name: a scalar memory store in the region.
    EXPECT_TRUE(classify(program({vadd(), smem(0x18, 4)})).scalar_memory_effect);
    // A scalar load whose result is read after the merge.
    EXPECT_TRUE(classify(program({smem(0x8, 4)}, {smov(0xBE860304u)})).scalar_memory_effect)
        << "s_mov_b32 s6, s4 reads the loaded value";
    // Control: the loaded register is redefined before it is read.
    EXPECT_TRUE(classify(program({smem(0x8, 5)}, {smov(kSMovS5Zero), smov(kSMovS6S5)})).clean());
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
}   // namespace
