// #5174 / #5185: the fabricated-fresh-mask-source guard (#4949) and ARBITRARY values.
//
// Policy (decisions on #5174 and #5185, 2026-10-15): a VGPR word the stage's explicit launch
// inventory (rdna2_launch_vgprs, #5163) says hardware does not load, and that nothing has written,
// holds an earlier wave's leftovers on hardware: an arbitrary value. So does anything computed
// only from arbitrary and defined values -- a relay, an accumulation, a path join, a cross-lane
// move -- because a deterministic function of values hardware could hold is a value hardware
// could produce. A predicate may read them. What stops admission is a HARD lane, whose hardware
// value is real but unknown to prosper:
//   - a launch VGPR hardware DOES load but the backend does not supply (native barycentrics; in
//     the unit emitter below, v1 under TIDIG_COMP_CNT = 1 with no bound local Y);
//   - scalar placeholders, and every word when the launch profile is unknown (no inventory);
//   - loaded data (#5185 keeps memory results hard, whatever their address).
// Lane-wise forms propagate hardness per lane; a cross-lane form makes every result lane hard
// when any lane it can read is. Each accepted arm has a refused twin that differs only in one hard
// input.
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
#include <set>
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

constexpr uint32_t mov6_v1 = 0x7e0c0301u;   // v_mov_b32 v6, v1
constexpr uint32_t mov8_v1 = 0x7e100301u;   // v_mov_b32 v8, v1
constexpr uint32_t mov10_v8 = 0x7e140308u;   // v_mov_b32 v10, v8
constexpr uint32_t compare10 = 0x7d84150au;   // v_cmp_eq_u32 vcc, v10, v10
constexpr uint32_t fmac8_v1 = 0x561002f2u;   // v_fmac_f32 v8, 1.0, v1
constexpr uint32_t mov6_s20 = 0x7e0c0214u;   // v_mov_b32 v6, s20
// v_mov_b32_dpp v8, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1: lane L reads lane L-1
// of its row (zero for the row's first lane), so every active lane is written.
Words dpp8_v6() {
    return {0x7e1002fau, 0xff091106u};
}
Words dpp10_v8() {   // v_mov_b32_dpp v10, v8 row_shr:1 bound_ctrl:1
    return {0x7e1402fau, 0xff091108u};
}
constexpr uint32_t mov10_v6 = 0x7e140306u;   // v_mov_b32 v10, v6

// The producing emitter: emit_alu records the EXEC facts and the marks. `known` passes the compute
// launch inventory for TIDIG_COMP_CNT = `tidig`; without it the profile is unknown. The emitter
// binds only v0, so with `tidig` = 1 hardware loads v1 but nothing supplies it: a hard word.
struct Emitter {
    SpirvCompute b;
    RegState rs;
    explicit Emitter(bool known, uint32_t tidig = 0) {
        b.begin(4, nullptr, 64, 1, 1, 64, 0, true, true);
        rs.exec = b.btrue();
        rs.vcc = b.bfalse();
        rs.vreg[0] = b.localid_comp[0];
        rs.absent_sgpr_reads_fabricated_zero = true;
        rs.max_vgpr = 20;
        const auto inventory = decode({mov6, mov6_v7, mov8_3, 0x7e260314u});
        const auto launch = compute_launch_vgprs(tidig);
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
void accepted(const Result& result) {
    EXPECT_FALSE(result.module.empty()) << result.reason;
}
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

// A scalar, top-tested loop (s22 from 0 to 3) whose body reads v6 and then runs `rewrite`.
Words loop_over_v6(const Words& rewrite) {
    const uint32_t body_size = static_cast<uint32_t>(rewrite.size() + 3);
    return cat({{0xbe960380u, 0xbe970383u, 0xbf0a1716u, 0xbf840000u | body_size, compare6},
                rewrite,
                {0x80168116u, 0xbf820000u | ((0x10000u - body_size - 2u) & 0xffffu)}});
}

// v_cmp_eq_i32 vcc, 0, v<vgpr>; v8 = vcc ? 0 : 1.0; export (v8, 0, 0, 1.0) to MRT0.
Words fragment_compare(uint32_t vgpr) {
    return {0x7d040080u | (vgpr << 9),
            0x7e120280u,
            0x021012f2u,
            0x7e140280u,
            0x7e1602f2u,
            0xf800180fu,
            0x0b0a0908u,
            end};
}
Result compile_fragment(const Words& body, uint32_t vgpr, const PixelSystemInputMapping* inputs) {
    const Words code = cat({body, fragment_compare(vgpr)});
    static uint64_t next_address = 0x5174100000ull;
    const uint64_t address = ++next_address;
    auto module = recompile_fragment(code.data(), code.size(), nullptr, inputs, UINT32_MAX, nullptr,
                                     true, {RecompileDiagnosticStage::Fragment, address});
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
    EXPECT_TRUE(known.rs.vreg_source_marks.at(6).arbitrary);
    EXPECT_TRUE(known.defined(compare6)) << "nothing ever defined v6; hardware never loads it";
    Emitter unknown(false);
    ASSERT_TRUE(unknown.rs.vreg_source_marks.at(6).hard);
    EXPECT_FALSE(unknown.rs.vreg_source_marks.at(6).arbitrary);
    EXPECT_FALSE(unknown.defined(compare6)) << "an unknown launch profile proves nothing";
}

TEST(UnloadedLaunchVgpr, AWiderExecReadsOnlyNeverWrittenUnloadedLanes) {
    Emitter known(true);
    ASSERT_TRUE(known.emit({cmpx_a, mov6, full_exec}));
    EXPECT_TRUE(known.rs.vreg_source_marks.at(6).arbitrary);
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

TEST(UnloadedLaunchVgpr, AWrittenOldValueIsAsHardAsWhatWroteIt) {
    // v6 = v7 under the full EXEC relays an arbitrary word, so A's narrowed write keeps an
    // arbitrary old value outside A: admitted (#5185; #5174 refused every written word here).
    Emitter e(true);
    ASSERT_TRUE(e.emit({mov6_v7, cmpx_a, mov6, full_exec}));
    EXPECT_TRUE(e.rs.vreg_source_marks.at(6).arbitrary);
    EXPECT_TRUE(e.defined(compare6));
    // The same old value relayed from v1, loaded by hardware but not supplied: refused.
    Emitter hard(true, 1);
    ASSERT_TRUE(hard.emit({mov6_v1, cmpx_a, mov6, full_exec}));
    EXPECT_FALSE(hard.rs.vreg_source_marks.at(6).arbitrary);
    EXPECT_FALSE(hard.defined(compare6));
}

TEST(UnloadedLaunchVgpr, APathJoinIsHardOnlyWhenAnEdgeIsHard) {
    Emitter e(true);
    ASSERT_TRUE(e.emit({cmpx_a, mov6}));
    // The other edge wrote v6 under the full EXEC (no mark there: defined). Each lane is
    // defined on one path and defined or arbitrary on the other: still arbitrary (#5185).
    VectorWordMarks written_edge;
    VectorWordMarks merged = e.rs.vreg_source_marks;
    join_vector_word_marks(merged, written_edge, &e.rs);
    ASSERT_TRUE(merged.at(6).hard);
    EXPECT_TRUE(merged.at(6).arbitrary);
    VectorWordMarks reversed = written_edge;
    join_vector_word_marks(reversed, e.rs.vreg_source_marks, &e.rs);
    EXPECT_TRUE(reversed.at(6).arbitrary);
    // The other edge relayed the hard v1 into v6: hard after the join, from either side.
    Emitter hard(true, 1);
    ASSERT_TRUE(hard.emit({mov6_v1}));
    VectorWordMarks hard_join = e.rs.vreg_source_marks;
    join_vector_word_marks(hard_join, hard.rs.vreg_source_marks, &e.rs);
    EXPECT_FALSE(hard_join.at(6).arbitrary);
    VectorWordMarks hard_first = hard.rs.vreg_source_marks;
    join_vector_word_marks(hard_first, e.rs.vreg_source_marks, &e.rs);
    EXPECT_FALSE(hard_first.at(6).arbitrary);
}

TEST(UnloadedLaunchVgpr, UnrelatedOwnersKeepTheWordArbitraryThroughAJoin) {
    // v6 is written under A, then under the unrelated sibling S. Its undefined lanes are
    // arbitrary, and so are the lanes of A outside S, which hold a value computed from defined
    // inputs: a path join keeps the word arbitrary (#5174 dropped it as "touched").
    Emitter e(true);
    ASSERT_TRUE(e.emit({cmpx_a, mov6, full_exec, cmpx_sib, mov6_3, full_exec}));
    const auto marks = e.rs.vreg_source_marks.at(6);
    ASSERT_TRUE(marks.hard);
    EXPECT_TRUE(marks.arbitrary);
    EXPECT_TRUE(e.defined(compare6));
    VectorWordMarks merged = e.rs.vreg_source_marks;
    join_vector_word_marks(merged, e.rs.vreg_source_marks, &e.rs);
    EXPECT_TRUE(merged.at(6).arbitrary);
}

// ---- #5185: propagation through computed values ----------------------------------------------

TEST(UnloadedLaunchVgpr, ARelayChainIsHardExactlyWhenItEndsAtAHardInput) {
    // v10 = v8 = v6 = <launch word>: a chain of length three.
    Emitter arbitrary(true, 1);
    ASSERT_TRUE(arbitrary.emit({mov6_v7, mov8_v6, mov10_v8}));
    EXPECT_TRUE(arbitrary.defined(compare10)) << "v7 is never loaded and never written";
    Emitter hard(true, 1);
    ASSERT_TRUE(hard.emit({mov6_v1, mov8_v6, mov10_v8}));
    EXPECT_FALSE(hard.defined(compare10)) << "v1 is loaded by hardware and not supplied";
    // The same chain with a narrowed hop: the lanes outside A keep the relayed hard v1.
    Emitter narrowed(true, 1);
    ASSERT_TRUE(narrowed.emit({mov6_v1, cmpx_a, mov8_v6, full_exec, mov10_v8}));
    EXPECT_FALSE(narrowed.defined(compare10));
}

TEST(UnloadedLaunchVgpr, AnAccumulationIsHardWhenItsAccumulatorOrSourceIs) {
    // v_fmac v8, 1.0, v6 reads the old v8 in every lane it writes.
    Emitter arbitrary(true, 1);
    ASSERT_TRUE(arbitrary.emit({fmac8_v6}));
    EXPECT_TRUE(arbitrary.defined(compare8)) << "arbitrary accumulator, arbitrary source";
    Emitter hard_accumulator(true, 1);
    ASSERT_TRUE(hard_accumulator.emit({mov8_v1, fmac8_v6}));
    EXPECT_FALSE(hard_accumulator.defined(compare8)) << "the accumulator relays v1";
    Emitter hard_source(true, 1);
    ASSERT_TRUE(hard_source.emit({fmac8_v1}));
    EXPECT_FALSE(hard_source.defined(compare8)) << "the source is v1";
    // Under a narrowed EXEC the accumulator's hard lanes are read inside it: still hard.
    Emitter narrowed(true, 1);
    ASSERT_TRUE(narrowed.emit({mov8_v1, cmpx_a, fmac8_v6}));
    EXPECT_FALSE(narrowed.defined(compare8));
}

TEST(UnloadedLaunchVgpr, ACrossLaneMovePullsAHardLaneIntoEveryResultLane) {
    // v6 = 3 under A: defined in A. Outside A it holds the relayed hard v1 or, in the arbitrary
    // control, nothing ever written. v8 is then written under A and read under the full EXEC, so
    // its lanes outside A are the never-written old v8 (arbitrary) and its lanes inside A are
    // what the write computed.
    Emitter lane_wise(true, 1);
    ASSERT_TRUE(lane_wise.emit({mov6_v1, cmpx_a, mov6_3, mov8_v6, full_exec}));
    EXPECT_TRUE(lane_wise.defined(compare8)) << "each lane of A reads its own defined v6 lane";
    // row_shr:1 lets a lane of A read a neighbour outside A, which holds the hard v1.
    Emitter cross(true, 1);
    ASSERT_TRUE(cross.emit(cat({{mov6_v1, cmpx_a, mov6_3}, dpp8_v6(), {full_exec}})));
    EXPECT_FALSE(cross.defined(compare8));
    // The same move when v6's lanes outside A were never written: every lane it can read is
    // defined or arbitrary.
    Emitter arbitrary(true, 1);
    ASSERT_TRUE(arbitrary.emit(cat({{cmpx_a, mov6_3}, dpp8_v6(), {full_exec}})));
    EXPECT_TRUE(arbitrary.defined(compare8));
}

TEST(UnloadedLaunchVgpr, ACrossLaneMoveCannotDropTheLoopObligationOfItsDefinedLanes) {
    // v6 is loop-carried: defined on the header's assumption, which the back edge checks.
    Emitter e(true);
    ASSERT_TRUE(e.emit({mov6}));
    LoopBlanketCheck check(e.rs, std::set<int>{}, 0);
    mark_vector_loop_carried(e.rs, decode({mov6}), 0, 1, check.carried, check.confined);
    // Under A, v8 and v10 copy it: defined in A on that assumption, arbitrary outside A.
    ASSERT_TRUE(e.emit({cmpx_a, mov8_v6, mov10_v6}));
    ASSERT_TRUE(e.rs.vreg_source_marks.at(10).arbitrary);
    ASSERT_FALSE(e.rs.vreg_source_marks.at(10).blanket_roots.empty());
    // row_shr:1 moves v8 into v10 across lanes. The owner goes, and with it the obligation the
    // defined lanes owe the loop proof: a read through the result could no longer record it, so
    // it is hard rather than arbitrary.
    ASSERT_TRUE(e.emit(cat({dpp10_v8(), {full_exec}})));
    EXPECT_FALSE(e.rs.vreg_source_marks.at(10).arbitrary);
    EXPECT_FALSE(e.defined(compare10));
}

TEST(UnloadedLaunchVgpr, ALoopBackEdgeCanCarryAHardValueIntoAnArbitraryWord) {
    // v6 is never written before the loop, so it is arbitrary on entry. The body reads it and
    // then rewrites it with LDS data read at the never-written address v7 (hard): the second
    // iteration reads that. The header cannot know what the back edge brings, so it does not keep
    // the entry's arbitrary property for a word the loop writes. (Conservative: the same refusal
    // applies when the body writes a defined value; proving the back edge is a follow-up.)
    refused(compile_compute(loop_over_v6({0xd8d80000u, 0x06000007u})));
}

TEST(UnloadedLaunchVgpr, AnM0RelativeWriteIsHardWhenM0IsAPlaceholder) {
    // v_movreld_b32 v6, v7 with M0 = 0: every word from v6 up takes v7 or keeps its old value.
    for (const bool placeholder : {false, true}) {
        Emitter e(true);
        e.rs.sreg[124] = e.b.uconst(0);
        if (placeholder) e.rs.sreg_merge_placeholder.insert(124);   // one path never wrote M0
        ASSERT_TRUE(e.emit({0x7e0c8507u}));
        // A genuine M0 replaces the word hardware replaced: arbitrary over arbitrary. A
        // placeholder M0 may replace another one, keeping v7 where hardware kept a real value.
        EXPECT_EQ(e.defined(compare6), !placeholder) << placeholder;
    }
}

TEST(UnloadedLaunchVgpr, AScalarPlaceholderRelayedIntoAVectorWordStaysHard) {
    Emitter e(true);
    e.rs.sreg[20] = e.b.uconst(0);
    ASSERT_TRUE(e.emit({mov6_s20, mov8_v6}));
    EXPECT_TRUE(e.defined(compare8)) << "a written s20";
    Emitter placeholder(true);
    placeholder.rs.sreg[20] = placeholder.b.uconst(0);
    placeholder.rs.sreg_merge_placeholder.insert(20);   // one path never wrote s20
    ASSERT_TRUE(placeholder.emit({mov6_s20, mov8_v6}));
    EXPECT_FALSE(placeholder.defined(compare8));
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

TEST(UnloadedLaunchVgpr, PublicComputeAdmitsArbitraryWordsAndRefusesAHardPath) {
    accepted(compile_compute({cmpx_a, mov6, full_exec, compare6}));
    accepted(compile_compute({cmpx_a, mov6, full_exec, cmpx_sib, compare6}));
    accepted(compile_compute({compare6}));   // never written at all
    // s_cmp_eq_u32 s4, 0; s_cbranch_scc1 over the write; v6 written on the taken path only.
    // Writing 3 leaves each lane outside A defined or arbitrary after the join (#5185; #5174
    // refused it as a word written on some path).
    accepted(
        compile_compute({0xbf068004u, 0xbf840001u, mov6_3, cmpx_a, mov6, full_exec, compare6}));
    // Writing LDS data read at the arbitrary address v7 leaves a hard lane on that path.
    refused(compile_compute(
        {0xbf068004u, 0xbf840002u, 0xd8d80000u, 0x06000007u, cmpx_a, mov6, full_exec, compare6}));
}

TEST(UnloadedLaunchVgpr, PublicComputeKeepsLoadedDataHardWhateverItsAddress) {
    // ds_read_b32 v6 at the defined address v0, then at the arbitrary address v7. The second is
    // real LDS content at an address hardware never formed: #5185 keeps loaded data hard.
    accepted(compile_compute({0xd8d80000u, 0x06000000u, compare6}));
    refused(compile_compute({0xd8d80000u, 0x06000007u, compare6}));
}

TEST(UnloadedLaunchVgpr, PublicFragmentRelayChainsAreAsHardAsTheirLaunchWord) {
    // PERSP_CENTER (v0, v1) and POS_X (v2): v1 is loaded by hardware but not supplied, v3 is
    // past the launch set. v8 = v6 = <word> is a relay chain of length two.
    const PixelSystemInputMapping inputs{0x102u, 0x102u};
    accepted(compile_fragment({0x7e0c0303u /* v_mov_b32 v6, v3 */, mov8_v6}, 8, &inputs));
    refused(compile_fragment({mov6_v1, mov8_v6}, 8, &inputs));
    // ...and through an accumulation whose accumulator relays the unsupplied word.
    accepted(compile_fragment({0x7e100303u /* v_mov_b32 v8, v3 */, fmac8_v6}, 8, &inputs));
    refused(compile_fragment({mov8_v1, fmac8_v6}, 8, &inputs));
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

// PERSP_CENTER (v0, v1) and POS_X (v2): v6 first holds the relayed v1, which hardware loads and
// prosper does not supply, so only the proof can admit the read. (#5174 used a relayed
// never-loaded v7, which #5185 makes arbitrary: that one-arm twin is now admitted.)
const PixelSystemInputMapping kBaryPosX{0x102u, 0x102u};
constexpr uint32_t cmpx_x4_v2 = 0x7c2204f6u;   // v_cmpx_lt_f32 4.0, v2
constexpr uint32_t cmpx_x4_ge_v2 = 0x7c2c04f6u;   // v_cmpx_ge_f32 4.0, v2
constexpr uint32_t cmpx_x2_v2 = 0x7c2204f4u;   // v_cmpx_lt_f32 2.0, v2

TEST(UnloadedLaunchVgpr, PublicFragmentJoinsTheTwoArmsUnderTheLaunchMask) {
    const Words both_arms{save_exec,          mov6_v1, cmpx_x4_v2,  mov6, andn2_exec_s30, execz_2,
                          scalar_side_effect, mov6_3,  restore_exec};
    accepted(compile_fragment(both_arms, 6, &kBaryPosX));
    // The second arm writes v8 instead: v6's lanes outside P keep the relayed hard value.
    const Words one_arm{save_exec,          mov6_v1, cmpx_x4_v2,  mov6, andn2_exec_s30, execz_2,
                        scalar_side_effect, mov8_3,  restore_exec};
    refused(compile_fragment(one_arm, 6, &kBaryPosX));
    // With the never-loaded v7 relayed instead, those lanes are arbitrary: admitted (#5185).
    const Words arbitrary_one_arm{
        save_exec,          mov6_v7, cmpx_x4,     mov6, andn2_exec_s30, execz_2,
        scalar_side_effect, mov8_3,  restore_exec};
    accepted(compile_fragment(arbitrary_one_arm, 6, &kPosX));
}

TEST(UnloadedLaunchVgpr, PublicFragmentJoinsTheTwoArmsUnderANarrowedMask) {
    // The parent mask is itself narrowed (x <= 4 first), as in Kena's nested shapes.
    const Words both_arms{cmpx_x4_ge_v2, save_exec,      mov6_v1, cmpx_x2_v2,
                          mov6,          andn2_exec_s30, execz_2, scalar_side_effect,
                          mov6_3,        restore_exec};
    accepted(compile_fragment(both_arms, 6, &kBaryPosX));
    const Words one_arm{cmpx_x4_ge_v2, save_exec,      mov6_v1, cmpx_x2_v2,
                        mov6,          andn2_exec_s30, execz_2, scalar_side_effect,
                        mov8_3,        restore_exec};
    refused(compile_fragment(one_arm, 6, &kBaryPosX));
}

// ---- A narrowed write over a defined word (found tracing case c) --------------------------------
// Outside EXEC the defined old word survives; inside it every source is defined. The result has
// no undefined lane. Narrowing its owner to EXEC refused the later wider read (Kena ps
// 0x5009050000 pc58, ps 0x5006bd0000 pc365). These run under an unknown launch profile, so the
// unloaded policy cannot be what admits them.

TEST(UnloadedLaunchVgpr, ANarrowedCompleteWriteOverADefinedWordIsDefined) {
    Emitter e(false);
    ASSERT_TRUE(e.emit({mov8_3, cmpx_a, mov6, cmpx_b, mov8_v6, full_exec}));
    EXPECT_FALSE(e.rs.vreg_source_marks.contains(8) && e.rs.vreg_source_marks.at(8).placeholder);
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
        ASSERT_TRUE(
            e.emit({mov8_3, cmpx_a, save_a_s8, cmpx_b, mov6, restore_s8, write, full_exec}));
        EXPECT_FALSE(e.defined(compare8)) << std::hex << write;
    }
}

// ---- #5183: a graphics ADDTID reload carries the spilled word's provenance ---------------------
// In a graphics stage ds_write_addtid_b32 / ds_read_addtid_b32 are a per-lane VGPR spill: the
// reload returns the word the matching write stored under the same M0. Now that the reload is a
// complete active-lane read, it must not look genuine when the spilled word was not.

TEST(UnloadedLaunchVgpr, PublicFragmentAddtidReloadCarriesTheSpilledWord) {
    constexpr uint32_t m0_zero = 0xb07c0000u;   // s_movk_i32 m0, 0
    const Words spill_v6{0xdac00000u, 0x00000600u};   // ds_write_addtid_b32 v6
    const Words reload_v8{0xdac40000u, 0x08000000u};   // ds_read_addtid_b32 v8
    // v8 first relays the unsupplied v1, so only the reload can make the compare genuine.
    accepted(compile_fragment(cat({{mov8_v1, m0_zero, mov6_3}, spill_v6, {cmpx_x4_v2}, reload_v8}),
                              8, &kBaryPosX));
    // The spilled word relays v1 itself: the reload is as hard as it.
    refused(compile_fragment(cat({{mov8_v1, m0_zero, mov6_v1}, spill_v6, {cmpx_x4_v2}, reload_v8}),
                             8, &kBaryPosX));
    // Spilled under P, reloaded under the restored mask: lanes outside P never wrote their slot.
    refused(compile_fragment(cat({{mov8_v1, m0_zero, mov6_3, save_exec, cmpx_x4_v2},
                                  spill_v6,
                                  {restore_exec},
                                  reload_v8}),
                             8, &kBaryPosX));
}
