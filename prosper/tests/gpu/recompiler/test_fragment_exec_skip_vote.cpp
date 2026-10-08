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
    // ADR mutation arm, by name, at the SPIR-V level: the state is correctly masked, but the
    // region also stores. It is refused by the existing body-purity rule (the stored composite is
    // itself outside the executable domain); the guest-level scalar-memory condition is tested
    // by EveryScalarMemoryOpcodeIsALoadOrRefused.
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
        "Prosper.FragmentExecSkipVote=060",     // leading zero
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

// Decode hand-assembled gfx10 words and check the decoder produced the instruction meant.
Rdna2Inst decode(std::vector<uint32_t> words, Rdna2Format expected_fmt, uint32_t expected_opcode) {
    Rdna2Inst in = rdna2_decode_one(words.data(), words.size());
    EXPECT_EQ(in.fmt, expected_fmt) << std::hex << words[0];
    EXPECT_EQ(in.opcode, expected_opcode) << std::hex << words[0];
    return in;
}
Rdna2Inst decode(uint32_t word, Rdna2Format fmt, uint32_t opcode) {
    return decode(std::vector<uint32_t>{word}, fmt, opcode);
}
Operand sgpr(int n) { return {OperandKind::SGPR, n}; }

constexpr uint32_t kBranch = 0xBF880000u; // s_cbranch_execz
constexpr uint32_t kEnd = 0xBF810000u;    // s_endpgm
constexpr uint32_t kVAdd = 0x06020702u;   // v_add_f32 v1, v2, v3
constexpr uint32_t kVMov = 0x7E020302u;   // v_mov_b32 v1, v2
constexpr uint32_t kVCmp = 0x7C020501u;   // v_cmp_lt_f32 vcc, v1, v2
constexpr uint32_t kVCmpx = 0x7C220501u;  // v_cmpx_lt_f32 vcc, v1, v2
constexpr uint32_t kSMovS6S5 = 0xBE860305u;    // s_mov_b32 s6, s5
constexpr uint32_t kSMovS5Zero = 0xBE850380u;  // s_mov_b32 s5, 0
constexpr uint32_t kSMovS5Seven = 0xBE850387u; // s_mov_b32 s5, 7
constexpr uint32_t kSAddU32 = 0x80050201u;     // s_add_u32 s5, s1, s2
constexpr uint32_t kSCmpEq = 0xBF060201u;      // s_cmp_eq_u32 s1, s2

// pc 0: s_cbranch_execz -> merge; the region follows; the merge is `tail` then s_endpgm.
struct Program {
    std::vector<Rdna2Inst> ins;
    uint32_t target = 0;
};
Program program(const std::vector<Rdna2Inst>& region, const std::vector<Rdna2Inst>& tail = {}) {
    Program p;
    auto branch = decode(kBranch | static_cast<uint32_t>(region.size()), Rdna2Format::SOPP, 0x08);
    branch.pc = 0;
    p.ins.push_back(branch);
    uint32_t pc = 1;
    for (auto in : region) { in.pc = pc; pc += in.len_dwords; p.ins.push_back(in); }
    p.target = pc;
    for (auto in : tail) { in.pc = pc; pc += in.len_dwords; p.ins.push_back(in); }
    auto end = decode(kEnd, Rdna2Format::SOPP, 0x01);
    end.pc = pc;
    p.ins.push_back(end);
    return p;
}
ExecSkipRegionEffects classify(const std::vector<Rdna2Inst>& region,
                               const std::vector<Rdna2Inst>& tail = {}) {
    const auto p = program(region, tail);
    return classify_exec_skip_region(p.ins, 0, p.target);
}
Rdna2Inst vadd() { return decode(kVAdd, Rdna2Format::VOP2, 0x03); }
Rdna2Inst vmov() { return decode(kVMov, Rdna2Format::VOP1, 0x01); }
Rdna2Inst smov(uint32_t word) { return decode(word, Rdna2Format::SOP1, 0x03); }
// A merge-side reader of VCC (v_cndmask_b32 reads it implicitly): makes a region's VCC write live.
Rdna2Inst vcc_reader() {
    Rdna2Inst in;
    in.fmt = Rdna2Format::VOP2;
    in.opcode = 0x01;
    in.dst = {OperandKind::VGPR, 1};
    in.src[0] = {OperandKind::VGPR, 2};
    in.src[1] = {OperandKind::VGPR, 3};
    in.n_src = 2;
    return in;
}
Rdna2Inst smem(uint32_t opcode, int dst) {
    Rdna2Inst in;
    in.fmt = Rdna2Format::SMEM;
    in.opcode = opcode;
    in.dst = sgpr(dst);
    in.src[0] = sgpr(0);
    in.src[1] = {OperandKind::InlineInt, 0};
    in.n_src = 2;
    return in;
}

TEST(FragmentExecSkipRegion, ValuOnlyRegionIsClean) {
    // Positive arm: VALU work only, including a v_cmp whose VCC result is dead at the merge, the
    // pure SOPP hints, and the wait-counter SOPKs.
    const auto cmp = decode(kVCmp, Rdna2Format::VOPC, 0x01);
    EXPECT_TRUE(classify({vadd(), vmov()}).clean());
    EXPECT_TRUE(classify({vadd(), cmp, vmov()}).clean()) << "VCC is unread after the merge";
    EXPECT_TRUE(classify({decode(0xBF800000u, Rdna2Format::SOPP, 0x00),  // s_nop
                          decode(0xBF8C0000u, Rdna2Format::SOPP, 0x0c),  // s_waitcnt
                          decode(0xBFA00000u, Rdna2Format::SOPP, 0x20),  // s_inst_prefetch
                          decode(0xBFA10000u, Rdna2Format::SOPP, 0x21),  // s_clause
                          decode(0xBD7D0000u, Rdna2Format::SOPK, 0x1a),  // s_waitcnt lgkmcnt
                          vadd()})
                    .clean());
    Rdna2Inst interp;
    interp.fmt = Rdna2Format::VINTRP;
    EXPECT_TRUE(classify({interp}).clean());
}

TEST(FragmentExecSkipRegion, SgprLiveOutIsRefusedAndDeadOneIsNot) {
    // ADR mutation arm, by name: a scalar written in the region and read after the merge. The only
    // scalar writers a VALU-only region may hold are the VOPC/carry/VOP3B VCC and SGPR-pair writes,
    // so this arm uses a v_cmp whose pair is read after the merge, with its dead control.
    const auto cmp = decode(kVCmp, Rdna2Format::VOPC, 0x01);
    EXPECT_TRUE(classify({vadd(), cmp}, {vcc_reader()}).scalar_live_out);
    EXPECT_FALSE(classify({vadd(), cmp}, {}).scalar_live_out);
    // The e64 form with an explicit SGPR pair: live while read, dead once the pair is redefined by
    // another compare before any read (the real Black Flag region needs this kill).
    const auto cmp_s2 = decode({0xD4C40002u, 0x00020501u}, Rdna2Format::VOPC, 0xC4);
    EXPECT_EQ(cmp_s2.dst.kind, OperandKind::SGPR);
    EXPECT_EQ(cmp_s2.dst.value, 2);
    const Rdna2Inst read_s2 = smov(0xBE860302u);  // s_mov_b32 s6, s2
    EXPECT_TRUE(classify({cmp_s2}, {read_s2}).scalar_live_out);
    EXPECT_FALSE(classify({cmp_s2}, {cmp_s2, read_s2}).scalar_live_out)
        << "a non-cmpx VOPC redefines both words of its SGPR pair";
}

TEST(FragmentExecSkipRegion, EveryCrossLaneOperationIsRefusedEvenWithADeadDestination) {
    // v_readfirstlane returns the invocation's own lane in a fragment shader, which is wrong on a
    // narrower host whatever its destination; likewise the lane-addressed and permuting forms.
    struct Case {
        const char* name;
        std::vector<uint32_t> words;
        Rdna2Format fmt;
        uint32_t opcode;
    };
    const Case cases[] = {
        {"v_readfirstlane_b32 s5, v1", {0x7E0A0501u}, Rdna2Format::VOP1, 0x02},
        {"v_readfirstlane_b32 s5, v1 (VOP3 form)", {0xD5820005u, 0x00000101u}, Rdna2Format::VOP3, 0x182},
        {"v_readlane_b32 s5, v1, 0", {0xD7600005u, 0x00010101u}, Rdna2Format::VOP3, 0x360},
        {"v_writelane_b32 v1, s5, 0", {0xD7610001u, 0x00010005u}, Rdna2Format::VOP3, 0x361},
        {"v_mbcnt_lo_u32_b32 v4, s0, 0", {0xD7650004u, 0x00010000u}, Rdna2Format::VOP3, 0x365},
        {"v_mbcnt_hi_u32_b32 v4, s1, v4", {0xD7660004u, 0x00020801u}, Rdna2Format::VOP3, 0x366},
        {"v_permlane16_b32 v1, v2, s0, s1", {0xD7770001u, 0x00000102u}, Rdna2Format::VOP3, 0x377},
        {"v_permlanex16_b32 v1, v2, s0, s1", {0xD7780001u, 0x00000102u}, Rdna2Format::VOP3, 0x378},
    };
    for (const auto& c : cases) {
        const auto fx = classify({decode(c.words, c.fmt, c.opcode)});
        EXPECT_TRUE(fx.wave_side_effect) << c.name;
        EXPECT_FALSE(fx.clean()) << c.name;
    }
    // The scenario that motivated the rule: dead destination, consumed inside the region.
    const auto write = decode(0x7E0A0501u, Rdna2Format::VOP1, 0x02);
    EXPECT_TRUE(classify({vadd(), write, vadd()}, {smov(kSMovS5Zero)}).wave_side_effect);
}

TEST(FragmentExecSkipRegion, CarryOutsAndFlagWritesAreLiveOutsOnlyWhenRead) {
    // VOP2 v_addc_co_u32 e32 writes its carry-out to VCC.
    const auto carry = decode(0x50020702u, Rdna2Format::VOP2, 0x28);
    EXPECT_TRUE(classify({carry}, {vcc_reader()}).scalar_live_out);
    EXPECT_FALSE(classify({carry}).scalar_live_out);
    // VOP3B v_add_co_u32 vcc, v1, v2, v3 carries its flag in SDST.
    const auto vop3b = decode({0xD70F6A01u, 0x00020702u}, Rdna2Format::VOP3, 0x30F);
    EXPECT_EQ(vop3b.sdst.kind, OperandKind::SGPR);
    EXPECT_TRUE(classify({vop3b}, {vcc_reader()}).scalar_live_out);
    EXPECT_FALSE(classify({vop3b}).scalar_live_out);
}

TEST(FragmentExecSkipRegion, ScalarAluAndExecWritesAreLiveOuts) {
    EXPECT_TRUE(classify({decode(kSAddU32, Rdna2Format::SOP2, 0x00)}).scalar_live_out);
    EXPECT_TRUE(classify({decode(kSMovS5Seven, Rdna2Format::SOP1, 0x03)}).scalar_live_out);
    EXPECT_TRUE(classify({decode(kSCmpEq, Rdna2Format::SOPC, 0x06)}).scalar_live_out) << "SCC";
    EXPECT_TRUE(classify({decode(0xB0050007u, Rdna2Format::SOPK, 0x00)}).scalar_live_out)
        << "s_movk_i32";
    EXPECT_TRUE(classify({decode(kVCmpx, Rdna2Format::VOPC, 0x11)}).scalar_live_out)
        << "v_cmpx writes EXEC";
}

TEST(FragmentExecSkipRegion, ModifiersAndDppAreWaveLevelOnEveryVectorFormat) {
    // DPP crosses lanes and SDWA/modifier forms are not modelled: refused on each VALU format.
    const Rdna2Inst forms[] = {vmov(), vadd(), decode(kVCmp, Rdna2Format::VOPC, 0x01),
                               decode({0xD5010003u, 0x00110103u}, Rdna2Format::VOP3, 0x101)};
    for (const auto& base : forms) {
        auto dpp = base;
        dpp.has_dpp = true;
        EXPECT_TRUE(classify({dpp}).wave_side_effect) << "dpp, fmt " << static_cast<int>(base.fmt);
        auto modified = base;
        modified.has_modifier = true;
        EXPECT_TRUE(classify({modified}).wave_side_effect)
            << "modifier, fmt " << static_cast<int>(base.fmt);
        EXPECT_FALSE(classify({base}, {vcc_reader()}).wave_side_effect);
    }
}

TEST(FragmentExecSkipRegion, EveryScalarMemoryOpcodeIsALoadOrRefused) {
    // gfx10.3 has no scalar stores. The SMEM opcode space holds the dword loads (0x0-0x4 s_load,
    // 0x8-0xC s_buffer_load) and, beyond them, only cache control, s_memtime/s_memrealtime and
    // s_atc_probe, none of which is a load the region rules can reason about. Every opcode in the
    // 6-bit field is classified: loads with a dead destination are clean, the rest are refused.
    for (uint32_t opcode = 0; opcode < 64; ++opcode) {
        const bool load = opcode <= 0x4 || (opcode >= 0x8 && opcode <= 0xC);
        const auto fx = classify({smem(opcode, 20)});  // s20..: never read after the merge
        EXPECT_EQ(fx.scalar_memory_effect, !load) << "opcode 0x" << std::hex << opcode;
        EXPECT_EQ(fx.clean(), load) << "opcode 0x" << std::hex << opcode;
    }
}

TEST(FragmentExecSkipRegion, ScalarLoadReadAfterTheMergeIsRefused) {
    // The ADR's scalar-memory condition: a load whose result is read after the merge.
    EXPECT_TRUE(classify({smem(0x8, 4)}, {smov(0xBE860304u)}).scalar_memory_effect)
        << "s_mov_b32 s6, s4 reads the loaded value";
    EXPECT_FALSE(
        classify({smem(0x8, 5)}, {smov(kSMovS5Zero), smov(kSMovS6S5)}).scalar_memory_effect)
        << "the loaded register is redefined before it is read";
}

TEST(FragmentExecSkipRegion, WaveLevelSideEffectsAreRefused) {
    for (const uint32_t word : {0xBF900000u /* s_sendmsg */, 0xBF8A0000u /* s_barrier */,
                                0xBF8E0001u /* s_sleep */, 0xBF920000u /* s_trap */,
                                0xBF960000u /* s_ttracedata */}) {
        const Rdna2Inst in = rdna2_decode_one(&word, 1);
        ASSERT_EQ(in.fmt, Rdna2Format::SOPP) << std::hex << word;
        EXPECT_TRUE(classify({vadd(), in}).wave_side_effect) << std::hex << word;
    }
    EXPECT_TRUE(classify({decode(0xBBFD0000u, Rdna2Format::SOPK, 0x17)}).wave_side_effect)
        << "s_waitcnt_vscnt publishes stores";
    Rdna2Inst setreg;
    setreg.fmt = Rdna2Format::SOPK;
    setreg.opcode = 0x13;
    EXPECT_TRUE(classify({setreg}).wave_side_effect);
}

TEST(FragmentExecSkipRegion, ExitsOtherThanTheMergeAreRefused) {
    for (const uint32_t opcode : {0x02u, 0x04u, 0x05u, 0x06u, 0x07u, 0x08u, 0x09u})
        EXPECT_TRUE(
            classify({decode(0xBF800000u | (opcode << 16) | 1u, Rdna2Format::SOPP, opcode)})
                .foreign_exit)
            << opcode;
    EXPECT_TRUE(classify({decode(kEnd, Rdna2Format::SOPP, 0x01)}).foreign_exit);
    // s_setpc_b64 s[0:1]: control goes where the decoded CFG does not.
    EXPECT_TRUE(classify({decode(0xBE802000u, Rdna2Format::SOP1, 0x20)}).foreign_exit);
}

TEST(FragmentExecSkipRegion, UnknownFormatsAndBackwardRegionsFailClosed) {
    for (const auto fmt : {Rdna2Format::DS, Rdna2Format::MUBUF, Rdna2Format::MTBUF,
                           Rdna2Format::MIMG, Rdna2Format::FLAT, Rdna2Format::EXP,
                           Rdna2Format::Unknown}) {
        Rdna2Inst in;
        in.fmt = fmt;
        EXPECT_TRUE(classify({in}).unclassified) << static_cast<int>(fmt);
    }
    const auto p = program({vadd()});
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
