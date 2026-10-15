// #5174: the fabricated-fresh-mask-source guard (#4949) and the VGPRs hardware never loads.
//
// Policy (decision on #5174, 2026-10-15): a VGPR word the stage's explicit launch inventory
// (rdna2_launch_vgprs, #5163) says hardware does not load, and that no path has written, holds an
// earlier wave's leftovers on hardware. A predicate may read it: there is no lane identity to
// fabricate, and the recompiler's zero is one of the values hardware could hold. What stays hard:
//   - a launch VGPR hardware DOES load but the backend does not supply (native barycentrics);
//   - a word with a write on some path, including one whose old value survives a narrowed write;
//   - every word when the launch profile is unknown (no inventory).
// Each accepted arm has a refused twin that differs only in one of those conditions.
//
// Case (c) of the issue is a proof, not the policy: two complementary arms of a saved mask, the
// second behind a forward s_cbranch_execz, rejoin and are read under the saved mask. Its tests use
// words with a real (written) old value, so the policy cannot be what admits them.
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_launch_vgprs.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;
using Words = std::vector<uint32_t>;

constexpr uint32_t cmpx_a = 0x7da6009fu;     // v_cmpx_le_u32 31, v0: EXEC A = EXEC & (v0 >= 31)
constexpr uint32_t cmpx_b = 0x7da600afu;     // v_cmpx_le_u32 47, v0: B, a proved subset of A
constexpr uint32_t cmpx_sib = 0x7da8009fu;   // v_cmpx_gt_u32 31, v0: EXEC & (v0 < 31)
constexpr uint32_t full_exec = 0xbefe04c1u;  // s_mov_b64 exec, -1
constexpr uint32_t cmp_vcc_47 = 0x7d8600afu;        // v_cmp_le_u32 vcc, 47, v0
constexpr uint32_t and_saveexec_vcc = 0xbe84246au;  // s_and_saveexec_b64 s[4:5], vcc
constexpr uint32_t andn2_exec_saved = 0x8afe7e04u;  // s_andn2_b64 exec, s[4:5], exec
constexpr uint32_t mov6 = 0x7e0c0281u, mov6_3 = 0x7e0c0283u, mov8_3 = 0x7e100283u;
constexpr uint32_t mov6_v7 = 0x7e0c0307u;   // v_mov_b32 v6, v7
constexpr uint32_t compare6 = 0x7d840d06u;  // v_cmp_eq_u32 vcc, v6, v6
constexpr uint32_t compare8 = 0x7d841108u;  // v_cmp_eq_u32 vcc, v8, v8
constexpr uint32_t mov8_v6 = 0x7e100306u;   // v_mov_b32 v8, v6
constexpr uint32_t fmac8_v6 = 0x56100cf2u;  // v_fmac_f32 v8, 1.0, v6 (reads the old v8)
constexpr uint32_t save_a_s8 = 0xbe88047eu, restore_s8 = 0xbefe0408u;   // s[8:9] <-> exec
constexpr uint32_t end = 0xbf810000u;

Words cat(std::initializer_list<Words> pieces) {
    Words code;
    for (const auto& piece : pieces) code.insert(code.end(), piece.begin(), piece.end());
    return code;
}
std::vector<Rdna2Inst> decode(const Words& code) {
    std::vector<Rdna2Inst> instructions;
    EXPECT_EQ(rdna2_walk(code.data(), code.size(), instructions), code.size());
    return instructions;
}

// The producing emitter: emit_alu records the EXEC facts and the marks. `known` passes the compute
// launch inventory for TIDIG_COMP_CNT = 0 (hardware loads v0 only); without it the profile is
// unknown.
struct Emitter {
    SpirvCompute b;
    RegState rs;
    explicit Emitter(bool known) {
        b.begin(4, nullptr, 64, 1, 1, 64, 0, true, true);
        rs.exec = b.btrue();
        rs.vcc = b.bfalse();
        rs.vreg[0] = b.localid_comp[0];
        rs.absent_sgpr_reads_fabricated_zero = true;
        rs.max_vgpr = 20;
        const auto inventory = decode({mov6, mov6_v7, mov8_3, 0x7e260314u});
        const auto launch = compute_launch_vgprs(0);
        seed_vector_word_definedness(rs, inventory, false, 64, known ? &launch : nullptr);
    }
    bool emit(const Words& code) {
        for (const auto& in : decode(code)) {
            const auto before = snapshot_saved_b64_masks(b, rs, in);
            bool ok = true;
            const bool handled = emit_alu(b, rs, in, ok, true);
            EXPECT_TRUE(handled);
            if (!handled || !ok) return false;
            record_scalar_write(rs, in, false, before);
        }
        return true;
    }
    bool defined(uint32_t comparison) {
        return fresh_mask_sources_defined(b, rs, rdna2_decode_one(&comparison, 1));
    }
};

struct Result {
    Words module;
    std::string reason;
};
void accepted(const Result& result) { EXPECT_FALSE(result.module.empty()) << result.reason; }
void refused(const Result& result) {
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("fabricated-fresh-mask-source"), std::string::npos)
        << result.reason;
}

// A compute program with a genuine output store, TIDIG_COMP_CNT = 0 (v0 loaded, nothing else).
Result compile_compute(const Words& body) {
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = 64;
    config.user_sgprs.resize(8);   // s4 is a genuine dynamic branch selector
    prosper::test::complete_compute_launch(config, 2);
    ShaderResourceTable table;
    ShaderResource output;
    output.cls = ResourceClass::ConstantBuffer;
    output.format = DataFormat::Uint32;
    output.num_components = 1;
    output.binding = 3;
    output.stride = 4;
    output.sgpr_base = 12;
    table.resources.push_back(output);
    // v9 = vcc ? 1 : 0 (via v_cndmask); buffer_store_dword v9, v0, s[12:15] offen.
    const Words code = cat({body, {0x7e140281u, 0x02121480u, 0xe0702000u, 0x80030900u, end}});
    static uint64_t next_address = 0x5174000000ull;
    const uint64_t address = ++next_address;
    auto module = recompile_compute(code.data(), code.size(), &table, config,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}

// v_cmp_eq_i32 vcc, 0, v<vgpr>; v8 = vcc ? 0 : 1.0; export (v8, 0, 0, 1.0) to MRT0.
Words fragment_compare(uint32_t vgpr) {
    return {0x7d040080u | (vgpr << 9), 0x7e120280u, 0x021012f2u, 0x7e140280u,
            0x7e1602f2u, 0xf800180fu, 0x0b0a0908u, end};
}
Result compile_fragment(const Words& body, uint32_t vgpr, const PixelSystemInputMapping* inputs) {
    const Words code = cat({body, fragment_compare(vgpr)});
    static uint64_t next_address = 0x5174100000ull;
    const uint64_t address = ++next_address;
    auto module = recompile_fragment(code.data(), code.size(), nullptr, inputs, UINT32_MAX,
                                     nullptr, true, {RecompileDiagnosticStage::Fragment, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}

// Fragment words. POS_X is v0 when it is the only enabled field (FragCoord.x, supplied).
constexpr uint32_t save_exec = 0xbe9e047eu;      // s_mov_b64 s[30:31], exec
constexpr uint32_t restore_exec = 0xbefe041eu;   // s_mov_b64 exec, s[30:31]
constexpr uint32_t cmpx_x4 = 0x7c2200f6u;        // v_cmpx_lt_f32 4.0, v0: P = W & (x > 4)
constexpr uint32_t cmpx_x4_ge = 0x7c2c00f6u;     // v_cmpx_ge_f32 4.0, v0: W & (x <= 4)
constexpr uint32_t andn2_exec_s30 = 0x8afe7e1eu;  // s_andn2_b64 exec, s[30:31], exec
constexpr uint32_t execz_2 = 0xbf880002u;        // s_cbranch_execz over two dwords
// s_add_u32 s5, 0, 1: its SCC write cannot be linearized, so the execz branch stays a real
// structured one-arm region whose skipped edge meets the arm at a merge.
constexpr uint32_t scalar_side_effect = 0x80058180u;
const PixelSystemInputMapping kPosX{0x100u, 0x100u};
}   // namespace

TEST(UnloadedLaunchVgpr, EncodingsNameTheFormsTheyClaim) {
    const auto sib = decode({cmpx_sib});
    ASSERT_EQ(sib.size(), 1u);
    EXPECT_EQ(sib[0].fmt, Rdna2Format::VOPC);
    EXPECT_TRUE(vopc_is_cmpx(sib[0].opcode));
    const auto frag = decode({save_exec, cmpx_x4, cmpx_x4_ge, andn2_exec_s30, execz_2,
                              scalar_side_effect, mov6_3, restore_exec});
    ASSERT_EQ(frag.size(), 8u);
    EXPECT_EQ(frag[0].fmt, Rdna2Format::SOP1);
    EXPECT_EQ(frag[0].dst.value, 30);
    EXPECT_TRUE(vopc_is_cmpx(frag[1].opcode));
    EXPECT_TRUE(vopc_is_cmpx(frag[2].opcode));
    EXPECT_EQ(frag[3].fmt, Rdna2Format::SOP2);
    EXPECT_EQ(frag[3].opcode, 0x15u);
    EXPECT_EQ(frag[4].fmt, Rdna2Format::SOPP);
    EXPECT_EQ(frag[4].opcode, 0x08u);   // s_cbranch_execz
    EXPECT_EQ(frag[5].fmt, Rdna2Format::SOP2);
    EXPECT_EQ(frag[5].opcode, 0x00u);   // s_add_u32
    EXPECT_EQ(frag[7].dst.value, 126);
    const auto launch = compute_launch_vgprs(0);
    ASSERT_EQ(launch.size(), 1u);
    EXPECT_EQ(launch[0].vgpr, 0u);
}

// ---- The policy ---------------------------------------------------------------------------------

TEST(UnloadedLaunchVgpr, AnUnloadedNeverWrittenWordIsSeededAsUnloadedOnlyWithAnInventory) {
    Emitter known(true);
    ASSERT_TRUE(known.rs.vreg_source_marks.at(6).hard);
    EXPECT_TRUE(known.rs.vreg_source_marks.at(6).unloaded);
    EXPECT_TRUE(known.defined(compare6)) << "nothing ever defined v6; hardware never loads it";
    Emitter unknown(false);
    ASSERT_TRUE(unknown.rs.vreg_source_marks.at(6).hard);
    EXPECT_FALSE(unknown.rs.vreg_source_marks.at(6).unloaded);
    EXPECT_FALSE(unknown.defined(compare6)) << "an unknown launch profile proves nothing";
}

TEST(UnloadedLaunchVgpr, AWiderExecReadsOnlyNeverWrittenUnloadedLanes) {
    Emitter known(true);
    ASSERT_TRUE(known.emit({cmpx_a, mov6, full_exec}));
    EXPECT_TRUE(known.rs.vreg_source_marks.at(6).unloaded);
    EXPECT_TRUE(known.defined(compare6));
    Emitter unknown(false);
    ASSERT_TRUE(unknown.emit({cmpx_a, mov6, full_exec}));
    EXPECT_FALSE(unknown.defined(compare6));
}

TEST(UnloadedLaunchVgpr, ASiblingExecReadsOnlyNeverWrittenUnloadedLanes) {
    Emitter known(true);
    ASSERT_TRUE(known.emit({cmpx_a, mov6, full_exec, cmpx_sib}));
    EXPECT_TRUE(known.defined(compare6));
    Emitter unknown(false);
    ASSERT_TRUE(unknown.emit({cmpx_a, mov6, full_exec, cmpx_sib}));
    EXPECT_FALSE(unknown.defined(compare6));
}

TEST(UnloadedLaunchVgpr, AnOldValueThatWasWrittenStaysHardUnderANarrowedWrite) {
    // v6 = v7 under the full EXEC is a write, so A's narrowed write keeps a written old value in
    // the lanes outside A. The decision keeps every written word hard.
    Emitter e(true);
    ASSERT_TRUE(e.emit({mov6_v7, cmpx_a, mov6, full_exec}));
    EXPECT_FALSE(e.rs.vreg_source_marks.at(6).unloaded);
    EXPECT_FALSE(e.defined(compare6));
}

TEST(UnloadedLaunchVgpr, APathJoinWithAWrittenEdgeIsNotNeverWritten) {
    Emitter e(true);
    ASSERT_TRUE(e.emit({cmpx_a, mov6}));
    // The other edge wrote v6 under the full EXEC (no mark there: defined).
    VectorWordMarks written_edge;
    VectorWordMarks merged = e.rs.vreg_source_marks;
    join_vector_word_marks(merged, written_edge, &e.rs);
    ASSERT_TRUE(merged.at(6).hard);
    EXPECT_FALSE(merged.at(6).unloaded);
    // Both edges never wrote the lanes outside A: still unloaded.
    VectorWordMarks same = e.rs.vreg_source_marks;
    join_vector_word_marks(same, e.rs.vreg_source_marks, &e.rs);
    EXPECT_TRUE(same.at(6).unloaded);
}

TEST(UnloadedLaunchVgpr, UnrelatedOwnersKeepUndefinedLanesUnloadedButNotUntouched) {
    // v6 is written under A, then under the unrelated sibling S. Its undefined lanes are still
    // only never-loaded lanes, but lanes of A outside S were written: the word is no longer
    // untouched outside its owner, so a path join can no longer keep it unloaded.
    Emitter e(true);
    ASSERT_TRUE(e.emit({cmpx_a, mov6, full_exec, cmpx_sib, mov6_3, full_exec}));
    const auto marks = e.rs.vreg_source_marks.at(6);
    ASSERT_TRUE(marks.hard);
    EXPECT_TRUE(marks.unloaded);
    EXPECT_FALSE(marks.untouched);
    EXPECT_TRUE(e.defined(compare6));
    VectorWordMarks merged = e.rs.vreg_source_marks;
    join_vector_word_marks(merged, e.rs.vreg_source_marks, &e.rs);
    EXPECT_FALSE(merged.at(6).unloaded);
}

TEST(UnloadedLaunchVgpr, ScalarPlaceholderInputsStayHard) {
    Emitter e(true);
    // v_cmp_eq_u32 vcc, s20, v6: s20 holds a merge placeholder (one path never wrote it).
    // Scalar data is outside the policy even beside an admissible unloaded v6.
    const uint32_t compare_s20 = 0x7d840c14u;
    e.rs.sreg[20] = e.b.uconst(0);
    EXPECT_TRUE(e.defined(compare_s20)) << "a written s20 beside the unloaded v6";
    e.rs.sreg_merge_placeholder.insert(20);
    EXPECT_FALSE(e.defined(compare_s20));
    EXPECT_TRUE(e.defined(compare6));
}

TEST(UnloadedLaunchVgpr, PublicComputeAdmitsUnloadedWordsAndRefusesAWrittenPath) {
    accepted(compile_compute({cmpx_a, mov6, full_exec, compare6}));
    accepted(compile_compute({cmpx_a, mov6, full_exec, cmpx_sib, compare6}));
    accepted(compile_compute({compare6}));   // never written at all
    // s_cmp_eq_u32 s4, 0; s_cbranch_scc1 over one instruction; v6 = 3 on the taken path only.
    // After the join some lanes outside A hold that written 3 on one path: refused.
    refused(compile_compute({0xbf068004u, 0xbf840001u, mov6_3, cmpx_a, mov6, full_exec, compare6}));
}

TEST(UnloadedLaunchVgpr, PublicFragmentRefusesALoadedButUnsuppliedBarycentric) {
    // PERSP_CENTER (v0, v1) and POS_X (v2). Without generated geometry the barycentrics are
    // loaded by hardware but unsupplied: they hold real values the zero would fabricate.
    const PixelSystemInputMapping inputs{0x102u, 0x102u};
    const uint32_t cmpx_pos = 0x7c2204f6u;   // v_cmpx_lt_f32 4.0, v2
    const uint32_t mov1 = 0x7e020281u, mov3 = 0x7e060281u;
    // v1 (persp_center.1) overwritten under P; restored EXEC reads its loaded lanes: refused.
    refused(compile_fragment({save_exec, cmpx_pos, mov1, restore_exec}, 1, &inputs));
    // v3 is past the launch set: hardware never loads it.
    accepted(compile_fragment({save_exec, cmpx_pos, mov3, restore_exec}, 3, &inputs));
    accepted(compile_fragment({}, 3, &inputs));
    // The same never-written word without a launch profile: unknown, so refused.
    refused(compile_fragment({}, 3, nullptr));
    // ADDR reserves POS_X but ENA leaves it clear: v2 is not loaded.
    const PixelSystemInputMapping reserved{0x2u, 0x102u};
    accepted(compile_fragment({}, 2, &reserved));
}

TEST(UnloadedLaunchVgpr, PublicFragmentTreatsEmptyOrPartialProfilesAsUnknown) {
    accepted(compile_fragment({}, 3, &kPosX));
    // ANCILLARY (v1 here) is bound only in part (#5160): a partial profile proves nothing.
    const PixelSystemInputMapping partial{0x2100u, 0x2100u};
    refused(compile_fragment({}, 3, &partial));
    // An empty ENA is no PS launch hardware performs; ENA outside ADDR has no slot.
    const PixelSystemInputMapping empty{0u, 0x100u};
    refused(compile_fragment({}, 3, &empty));
    const PixelSystemInputMapping unslotted{0x300u, 0x100u};
    refused(compile_fragment({}, 3, &unslotted));
}

// ---- Case (c): complementary arms joined, read under the saved mask ------------------------------

TEST(UnloadedLaunchVgpr, TheSkippedSecondArmOfAPartitionIsConfinedToTheSavedMask) {
    Emitter e(false);   // unknown profile: nothing here may ride on the policy
    ASSERT_TRUE(e.emit({cmpx_a}));
    const uint32_t saved = e.rs.exec;
    ASSERT_TRUE(e.emit({cmp_vcc_47, and_saveexec_vcc, mov6, andn2_exec_saved}));
    const uint32_t second = e.rs.exec;
    ASSERT_TRUE(e.rs.exec_complement_in.contains(second));
    VectorWordMarks skipped{{6, e.rs.vreg_source_marks.at(6)}};   // owner P on the skipped edge
    ASSERT_TRUE(e.emit({mov6_3}));   // the taken second arm completes the partition
    ASSERT_EQ(e.rs.vreg_source_marks.at(6).outside_exec, saved);
    VectorWordMarks confined = skipped;
    confine_execz_skip_marks(e.rs, confined, e.rs.vreg_source_marks, second);
    EXPECT_EQ(confined.at(6).outside_exec, saved);
    VectorWordMarks merged = e.rs.vreg_source_marks;
    join_vector_word_marks(merged, confined, &e.rs);
    EXPECT_EQ(merged.at(6).outside_exec, saved);
    // Without the proof the meet keeps the first arm's narrower owner.
    VectorWordMarks plain = e.rs.vreg_source_marks;
    join_vector_word_marks(plain, skipped, &e.rs);
    EXPECT_NE(plain.at(6).outside_exec, saved);
}

TEST(UnloadedLaunchVgpr, AFirstArmNarrowerThanItsMaskDoesNotConfineTheSkippedEdge) {
    Emitter e(false);
    ASSERT_TRUE(e.emit({cmpx_a}));
    const uint32_t saved = e.rs.exec;
    ASSERT_TRUE(e.emit({cmp_vcc_47, and_saveexec_vcc}));
    const uint32_t first = e.rs.exec;
    // v6 is written only under C (v0 >= 55 within P), between saving and restoring P in s[8:9]:
    // lanes of P outside C never receive it.
    ASSERT_TRUE(e.emit({0xbe88047eu, 0x7da600b7u, mov6, 0xbefe0408u}));
    ASSERT_EQ(e.rs.exec, first);
    VectorWordMarks skipped{{6, e.rs.vreg_source_marks.at(6)}};
    ASSERT_TRUE(e.emit({andn2_exec_saved}));
    const uint32_t second = e.rs.exec;
    ASSERT_TRUE(e.rs.exec_complement_in.contains(second));
    VectorWordMarks taken{{6, e.rs.vreg_source_marks.at(6)}};
    taken.at(6).outside_exec = saved;   // even if the taken edge reached the saved mask
    VectorWordMarks confined = skipped;
    confine_execz_skip_marks(e.rs, confined, taken, second);
    EXPECT_NE(confined.at(6).outside_exec, saved);
}

TEST(UnloadedLaunchVgpr, PublicFragmentJoinsTheTwoArmsUnderTheLaunchMask) {
    // v6 first holds a WRITTEN value (v7 relayed), so only the proof can admit the read.
    const Words both_arms{save_exec, mov6_v7, cmpx_x4, mov6, andn2_exec_s30, execz_2,
                          scalar_side_effect, mov6_3, restore_exec};
    accepted(compile_fragment(both_arms, 6, &kPosX));
    // The second arm writes v8 instead: v6's lanes outside P keep the relayed value.
    const Words one_arm{save_exec, mov6_v7, cmpx_x4, mov6, andn2_exec_s30, execz_2,
                        scalar_side_effect, mov8_3, restore_exec};
    refused(compile_fragment(one_arm, 6, &kPosX));
}

TEST(UnloadedLaunchVgpr, PublicFragmentJoinsTheTwoArmsUnderANarrowedMask) {
    // The parent mask is itself narrowed (x <= 4 first), as in Kena's nested shapes.
    const Words both_arms{cmpx_x4_ge, save_exec, mov6_v7, 0x7c2200f4u /* v_cmpx_lt_f32 2.0, v0 */,
                          mov6, andn2_exec_s30, execz_2, scalar_side_effect, mov6_3, restore_exec};
    accepted(compile_fragment(both_arms, 6, &kPosX));
    const Words one_arm{cmpx_x4_ge, save_exec, mov6_v7, 0x7c2200f4u, mov6, andn2_exec_s30,
                        execz_2, scalar_side_effect, mov8_3, restore_exec};
    refused(compile_fragment(one_arm, 6, &kPosX));
}

// ---- A narrowed write over a defined word (found tracing case c) --------------------------------
// Outside EXEC the defined old word survives; inside it every source is defined. The result has
// no undefined lane. Narrowing its owner to EXEC refused the later wider read (Kena ps
// 0x5009050000 pc58, ps 0x5006bd0000 pc365). These run under an unknown launch profile, so the
// unloaded policy cannot be what admits them.

TEST(UnloadedLaunchVgpr, ANarrowedCompleteWriteOverADefinedWordIsDefined) {
    Emitter e(false);
    ASSERT_TRUE(e.emit({mov8_3, cmpx_a, mov6, cmpx_b, mov8_v6, full_exec}));
    EXPECT_FALSE(e.rs.vreg_source_marks.contains(8) &&
                 e.rs.vreg_source_marks.at(8).placeholder);
    EXPECT_TRUE(e.defined(compare8));
    Emitter undefined_old(false);   // v8 was never written before the narrowed write
    ASSERT_TRUE(undefined_old.emit({cmpx_a, mov6, cmpx_b, mov8_v6, full_exec}));
    EXPECT_FALSE(undefined_old.defined(compare8));
}

TEST(UnloadedLaunchVgpr, ANarrowedAccumulationOverADefinedWordIsDefined) {
    Emitter e(false);
    ASSERT_TRUE(e.emit({mov8_3, cmpx_a, mov6, cmpx_b, fmac8_v6, full_exec}));
    EXPECT_TRUE(e.defined(compare8));
    Emitter undefined_old(false);
    ASSERT_TRUE(undefined_old.emit({cmpx_a, mov6, cmpx_b, fmac8_v6, full_exec}));
    EXPECT_FALSE(undefined_old.defined(compare8));
}

TEST(UnloadedLaunchVgpr, ASourceUndefinedInsideTheWriteExecIsNotMadeDefined) {
    // v6 is defined only under B; the write runs under the wider A, so lanes of A outside B take
    // v6's undefined value even though the old v8 was defined.
    for (const uint32_t write : {mov8_v6, fmac8_v6}) {
        Emitter e(false);
        ASSERT_TRUE(e.emit({mov8_3, cmpx_a, save_a_s8, cmpx_b, mov6, restore_s8, write, full_exec}));
        EXPECT_FALSE(e.defined(compare8)) << std::hex << write;
    }
}
