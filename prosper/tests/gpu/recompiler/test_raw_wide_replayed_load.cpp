// #4519: a V# descriptor load inside a loop is a fresh observation each time round, not a numeric
// load. Space Adventure Cobra's lit fragment programs reload a V# from the resource table on every
// iteration of their per-light loop and recycle its register pair for a compare mask:
//
//     1197  s_load_dwordx4 s[16:19], s[28:29], 0xf0
//     1201  s_buffer_load_dwordx4 s[8:11], s[16:19], vcc_lo     the V#'s only use: SBASE
//     1238  v_cmp_*_sdwa s[16:17], 0, s10                       the pair becomes a mask
//     1262  s_mov_b64 vcc, s[16:17]
//     1818  s_branch 1191
//
// The classifier called that load numeric raw data. With a v_readfirstlane in the program, the
// draw was then routed to the owned-wave path, which refuses any draw with a depth buffer, and
// the ground, rocks and deck were dropped.
//
// The load, the compare, the mask move and the v_readfirstlane are the live instruction words.
// The fetch is the live opcode with an immediate offset where the shader uses vcc_lo. The
// classifier does read a fetch's offset register, as a possible numeric use of a loaded word;
// vcc_lo is not a loaded word here, so the substitution changes nothing it decides. Each refusal
// arm changes exactly one property the admission depends on, and names the blocker it expects,
// so an arm cannot pass for another arm's reason.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t kLoadPc = 1;

struct Shape {
    uint32_t first_word = 0x7e020280u;   // v_mov_b32 v1, 0: above the loop, never walked
    uint32_t load_offset_word = 0xfa0000f0u;   // SOFFSET null, immediate 0xf0
    std::vector<uint32_t> in_loop;   // extra instructions between the fetch and the mask
    bool compare = true;   // v_cmp_*_sdwa s[16:17], 0, s10: recycles the pair as a mask
    bool move_mask_to_vcc = true;   // s_mov_b64 vcc, s[16:17] after the compare
    std::vector<uint32_t> after_mask;   // extra instructions after the compare (and move)
    std::vector<uint32_t> after_exit;   // extra instructions past the loop, before the end
};

//  0  v_mov_b32 v1, 0
//  1  s_load_dwordx4 s[16:19], s[28:29], 0xf0            <- the load, and the loop header
//  3  s_buffer_load_dwordx4 s[8:11], s[16:19], 0xc0      descriptor use
//     (in_loop)
//  .  v_cmp_*_sdwa s[16:17], 0, s10                      recycles the pair as a mask
//  .  s_mov_b64 vcc, s[16:17]
//  .  s_cbranch_vccz +1                                  leave the loop
//  .  s_branch 1                                         back to the load
//  .  v_readfirstlane vcc_lo, v1                         what makes a numeric load "wave wide"
//  .  s_endpgm
std::vector<Rdna2Inst> program(const Shape& shape) {
    std::vector<uint32_t> code{
        shape.first_word, 0xf408040eu, shape.load_offset_word, 0xf4280208u, 0xfa0000c0u,
    };
    code.insert(code.end(), shape.in_loop.begin(), shape.in_loop.end());
    if (shape.compare) code.insert(code.end(), {0x7c1a14f9u, 0x86869080u});
    if (shape.move_mask_to_vcc) code.push_back(0xbeea0410u);
    code.insert(code.end(), shape.after_mask.begin(), shape.after_mask.end());
    code.push_back(0xbf860001u);
    const uint32_t branch_pc = static_cast<uint32_t>(code.size());
    const uint32_t back = (kLoadPc - (branch_pc + 1u)) & 0xffffu;
    code.push_back(0xbf820000u | back);
    code.insert(code.end(), shape.after_exit.begin(), shape.after_exit.end());
    code.insert(code.end(), {0x7ed40501u, 0xbf810000u});
    std::vector<Rdna2Inst> instructions;
    EXPECT_EQ(rdna2_walk(code.data(), code.size(), instructions), code.size());
    return instructions;
}

const Rdna2Inst& at(const std::vector<Rdna2Inst>& instructions, uint32_t pc) {
    for (const Rdna2Inst& in : instructions)
        if (in.pc == pc) return in;
    ADD_FAILURE() << "no instruction at pc " << pc;
    return instructions.front();
}

// The single diagnosis row for the load, or an empty kind when the classifier cleared it.
std::string numeric_blocker(const std::vector<Rdna2Inst>& instructions, uint32_t* pc = nullptr,
                            bool wave64 = false) {
    for (const RawWideLoadDiagnosis& row : rdna2_raw_wide_data_load_diagnoses(instructions, wave64))
        if (row.load_pc == kLoadPc) {
            if (pc) *pc = row.numeric_pc;
            return row.numeric_kind;
        }
    return {};
}

// The load's diagnosis row; `found` is false when the classifier cleared the load.
RawWideLoadDiagnosis diagnosis(const std::vector<Rdna2Inst>& instructions, bool* found = nullptr) {
    for (const RawWideLoadDiagnosis& row : rdna2_raw_wide_data_load_diagnoses(instructions))
        if (row.load_pc == kLoadPc) {
            if (found) *found = true;
            return row;
        }
    if (found) *found = false;
    return {};
}

bool flagged(const std::vector<Rdna2Inst>& instructions, bool wave64 = false) {
    const auto loads = rdna2_raw_wide_data_loads(instructions, wave64);
    return std::find(loads.begin(), loads.end(), kLoadPc) != loads.end();
}

}   // namespace

TEST(RawWideReplayedLoad, FixtureIsTheLiveShape) {
    const auto instructions = program({});
    const Rdna2Inst& load = at(instructions, kLoadPc);
    ASSERT_EQ(load.fmt, Rdna2Format::SMEM);
    ASSERT_EQ(load.opcode, 0x2u) << "s_load_dwordx4";
    EXPECT_EQ(load.dst.value, 16);
    EXPECT_EQ(load.src[0].value, 28);
    EXPECT_EQ(load.src[1].kind, OperandKind::Special);
    EXPECT_EQ(load.src[1].value, 125) << "null SOFFSET: an immediate load";
    const Rdna2Inst& fetch = at(instructions, 3);
    ASSERT_EQ(fetch.fmt, Rdna2Format::SMEM);
    EXPECT_GE(fetch.opcode, 0x8u) << "s_buffer_load: SBASE is a descriptor";
    EXPECT_EQ(fetch.src[0].value, 16);
    const Rdna2Inst& compare = at(instructions, 5);
    ASSERT_EQ(compare.fmt, Rdna2Format::VOPC);
    EXPECT_EQ(compare.dst.kind, OperandKind::SGPR);
    EXPECT_EQ(compare.dst.value, 16) << "the compare recycles the V#'s first pair";
    const Rdna2Inst& move = at(instructions, 7);
    ASSERT_EQ(move.fmt, Rdna2Format::SOP1);
    EXPECT_EQ(move.src[0].value, 16);
    const Rdna2Inst& branch = at(instructions, 9);
    ASSERT_EQ(branch.fmt, Rdna2Format::SOPP);
    EXPECT_EQ(static_cast<int64_t>(branch.pc) + branch.len_dwords + branch.simm16, kLoadPc)
        << "the back-edge returns to the load itself";
    const Rdna2Inst& first_lane = at(instructions, 10);
    ASSERT_EQ(first_lane.fmt, Rdna2Format::VOP1);
    EXPECT_EQ(first_lane.opcode, 0x2u) << "v_readfirstlane";
}

TEST(RawWideReplayedLoad, DescriptorLoadInALoopIsNotNumericData) {
    const auto instructions = program({});
    EXPECT_FALSE(flagged(instructions)) << numeric_blocker(instructions);
    // The consequence that dropped the draws: no numeric load, so no owned-wave requirement.
    EXPECT_TRUE(rdna2_raw_wave_wide_data_loads(instructions).empty());
}

TEST(RawWideReplayedLoad, RealNumericReaderInTheLoopStillNeedsBacking) {
    // v_mov_b32 v0, s18: reads a loaded word the compare never touches.
    const auto instructions = program({.in_loop = {0x7e000212u}});
    ASSERT_EQ(at(instructions, 5).src[0].value, 18);
    EXPECT_TRUE(flagged(instructions));
    uint32_t pc = 0;
    EXPECT_EQ(numeric_blocker(instructions, &pc), "numeric-reader");
    EXPECT_EQ(pc, 5u);
    EXPECT_FALSE(rdna2_raw_wave_wide_data_loads(instructions).empty());
}

TEST(RawWideReplayedLoad, RegisterOffsetReplayStaysUncertain) {
    // s_load_dwordx4 s[16:19], s[28:29], s20: the offset is exactly what a loop changes.
    const auto instructions = program({.load_offset_word = 0x28000000u});
    ASSERT_EQ(at(instructions, kLoadPc).src[1].kind, OperandKind::SGPR);
    ASSERT_EQ(at(instructions, kLoadPc).src[1].value, 20);
    EXPECT_TRUE(flagged(instructions));
    EXPECT_EQ(numeric_blocker(instructions), "load-re-executed");
}

TEST(RawWideReplayedLoad, RewrittenBaseReplayStaysUncertain) {
    // s_add_i32 s28, s28, 1: the next iteration reads a different address.
    const auto instructions = program({.in_loop = {0x811c811cu}});
    ASSERT_EQ(at(instructions, 5).fmt, Rdna2Format::SOP2);
    ASSERT_EQ(at(instructions, 5).dst.value, 28);
    EXPECT_TRUE(flagged(instructions));
    EXPECT_EQ(numeric_blocker(instructions), "load-re-executed");
}

TEST(RawWideReplayedLoad, GuestMemoryWriteReplayStaysUncertain) {
    // image_store: the bytes behind the same address may differ on the next execution.
    const auto instructions = program({.in_loop = {0xf0200308u, 0x00c20007u}});
    ASSERT_EQ(at(instructions, 5).fmt, Rdna2Format::MIMG);
    ASSERT_TRUE(rdna2_may_write_guest_memory(at(instructions, 5)));
    EXPECT_TRUE(flagged(instructions));
    EXPECT_EQ(numeric_blocker(instructions), "load-re-executed");
}

namespace {
struct Escape {
    const char* name;
    uint32_t word;
    Rdna2Format fmt;
    uint32_t opcode;
    // False for s_movrels_b32 alone: it writes the register it names, and what it reads through
    // M0 is inventoried as a range (#4538). Its pair form is not lowered and stays refused.
    bool writes_unnamed_or_leaves_cfg = true;
};
// One encoding of every instruction rdna2_escapes_decoded_effects names.
constexpr Escape kEscapes[] = {
    {"s_setpc_b64 s[60:61]", 0xbe80203cu, Rdna2Format::SOP1, kSop1OpcodeSetpcB64},
    {"s_swappc_b64 s[60:61], s[40:41]", 0xbebc2128u, Rdna2Format::SOP1, kSop1OpcodeSwappcB64},
    {"s_rfe_b64 s[60:61]", 0xbe80223cu, Rdna2Format::SOP1, kSop1OpcodeRfeB64},
    {"s_movrels_b32 s60, s40", 0xbebc2e28u, Rdna2Format::SOP1, kSop1OpcodeMovrelsB32, false},
    {"s_movrels_b64 s[60:61], s[40:41]", 0xbebc2f28u, Rdna2Format::SOP1, kSop1OpcodeMovrelsB64},
    {"s_movreld_b32 s60, s40", 0xbebc3028u, Rdna2Format::SOP1, kSop1OpcodeMovreldB32},
    {"s_movreld_b64 s[60:61], s[40:41]", 0xbebc3128u, Rdna2Format::SOP1, kSop1OpcodeMovreldB64},
    {"s_movrelsd_2_b32 s60, s40", 0xbebc4928u, Rdna2Format::SOP1, kSop1OpcodeMovrelsd2B32},
    {"s_call_b64 s[60:61], +0", 0xbb3c0000u, Rdna2Format::SOPK, kSopkOpcodeCallB64},
    {"s_subvector_loop_begin s60, +0", 0xbdbc0000u, Rdna2Format::SOPK,
     kSopkOpcodeSubvectorLoopBegin},
    {"s_subvector_loop_end s60, +0", 0xbe3c0000u, Rdna2Format::SOPK, kSopkOpcodeSubvectorLoopEnd},
};
}   // namespace

TEST(RawWideReplayedLoad, EscapeListIsTheOneTheDecoderNames) {
    for (const Escape& escape : kEscapes) {
        const Rdna2Inst in = rdna2_decode_one(&escape.word, 1);
        ASSERT_EQ(in.fmt, escape.fmt) << escape.name;
        ASSERT_EQ(in.opcode, escape.opcode) << escape.name;
        EXPECT_TRUE(rdna2_escapes_decoded_effects(in)) << escape.name;
        EXPECT_EQ(rdna2_may_write_unnamed_register_or_leave_cfg(in),
                  escape.writes_unnamed_or_leaves_cfg)
            << escape.name;
    }
    // The range three guards used to call "relative SGPR write" is B64 saveexec, and stays out.
    for (uint32_t opcode = kSop1OpcodeAndSaveexecB64; opcode <= kSop1OpcodeXnorSaveexecB64;
         ++opcode) {
        const uint32_t word = 0xbea800c1u | (opcode << 8u);   // s_*_saveexec_b64 s[40:41], -1
        const Rdna2Inst in = rdna2_decode_one(&word, 1);
        ASSERT_EQ(in.fmt, Rdna2Format::SOP1);
        ASSERT_EQ(in.opcode, opcode);
        EXPECT_FALSE(rdna2_escapes_decoded_effects(in)) << "opcode 0x" << std::hex << opcode;
        EXPECT_FALSE(rdna2_may_write_unnamed_register_or_leave_cfg(in))
            << "opcode 0x" << std::hex << opcode;
    }
}

TEST(RawWideReplayedLoad, AWriterOrTransferAnywhereInTheProgramKeepsReplayUncertain) {
    // Each sits ABOVE the loop, where neither walk ever goes, so only the whole-program condition
    // can refuse it. A transfer may run code that was never decoded; an M0-relative DESTINATION
    // move writes a register its encoding does not name. Either could change the base pair
    // between executions. s_movrels_b32 cannot, and must not cost the load its proof: it is an
    // instruction the emitter lowers.
    for (const Escape& escape : kEscapes) {
        const auto instructions = program({.first_word = escape.word});
        ASSERT_EQ(at(instructions, 0).opcode, escape.opcode) << escape.name;
        if (escape.writes_unnamed_or_leaves_cfg) {
            EXPECT_TRUE(flagged(instructions)) << escape.name;
            EXPECT_EQ(numeric_blocker(instructions), "load-re-executed") << escape.name;
        } else {
            EXPECT_FALSE(flagged(instructions))
                << escape.name << ": " << numeric_blocker(instructions);
        }
    }
}

TEST(RawWideReplayedLoad, AnEscapeOnTheWalkedPathStopsBothWalks) {
    // In the loop body, with the loaded words live. Before #4529 only the transfers and the call
    // stopped the numeric walk, and only the transfers stopped the cheap one. s_movrels_b32 is
    // the one escape a walk now steps through; ARelativeReadIsChargedItsWholeRange has its arms.
    for (const Escape& escape : kEscapes) {
        if (!escape.writes_unnamed_or_leaves_cfg) continue;
        const auto instructions = program({.in_loop = {escape.word}});
        ASSERT_EQ(at(instructions, 5).opcode, escape.opcode) << escape.name;
        bool found = false;
        const RawWideLoadDiagnosis row = diagnosis(instructions, &found);
        ASSERT_TRUE(found) << escape.name;
        // Each walk is asserted on its own: the cheap one would otherwise go unnoticed, because
        // without its guard it steps over the escape and stops at the s_mov_b64 at pc 8 instead,
        // and the load is flagged either way.
        EXPECT_STREQ(row.backing_kind, "unknown-or-indirect-control") << escape.name;
        EXPECT_EQ(row.backing_pc, 5u) << escape.name;
        EXPECT_STREQ(row.numeric_kind, "unmodelled-control-or-relative-sgpr") << escape.name;
        EXPECT_EQ(row.numeric_pc, 5u) << escape.name;
    }
}

TEST(RawWideReplayedLoad, ARelativeReadIsChargedItsWholeRange) {
    // s_movrels_b32 s60, sS; v_mov_b32 v0, s60 in the loop body. The loaded words are s16..s19,
    // and the relative read may return any register from sS up to s105.
    const uint32_t reader = 0x7e00023cu;   // v_mov_b32 v0, s60
    {
        // Base above the loaded words: none of them is a candidate, so s60 carries nothing the
        // load produced and the loop is still cleared. Stopping at the instruction, as both
        // walks did, flagged this.
        const auto above = program({.in_loop = {0xbebc2e28u, reader}});   // s_movrels_b32 s60, s40
        ASSERT_EQ(at(above, 5).opcode, kSop1OpcodeMovrelsB32);
        ASSERT_EQ(at(above, 5).src[0].value, 40);
        ASSERT_EQ(at(above, 6).src[0].value, 60);
        EXPECT_FALSE(flagged(above)) << numeric_blocker(above);
    }
    {
        // Base below them: s16..s19 are candidates. The relative read is a scalar derivation, and
        // the v_mov that consumes its result is the numeric reader -- at its own pc, which a walk
        // that charged the relative read s12 alone never reports.
        const auto below = program({.in_loop = {0xbebc2e0cu, reader}});   // s_movrels_b32 s60, s12
        ASSERT_EQ(at(below, 5).opcode, kSop1OpcodeMovrelsB32);
        ASSERT_EQ(at(below, 5).src[0].value, 12);
        bool found = false;
        const RawWideLoadDiagnosis row = diagnosis(below, &found);
        ASSERT_TRUE(found);
        EXPECT_STREQ(row.backing_kind, "data-read");
        EXPECT_EQ(row.backing_pc, 5u);
        EXPECT_STREQ(row.numeric_kind, "numeric-reader");
        EXPECT_EQ(row.numeric_pc, 6u);
    }
    {
        // The same relative read with nobody consuming its result is a derivation and nothing
        // more, exactly as s_mov_b32 s60, s16 would be.
        const auto unread = program({.in_loop = {0xbebc2e0cu}});
        ASSERT_EQ(at(unread, 5).opcode, kSop1OpcodeMovrelsB32);
        EXPECT_FALSE(flagged(unread)) << numeric_blocker(unread);
    }
}

TEST(RawWideReplayedLoad, ALoadedWordThatReachesM0IsANumericUse) {
    // s_mov_b32 m0, s18; s_movrels_b32 s60, s40; v_mov_b32 v0, s60. No loaded word is in the
    // relative read's range, but one is its INDEX. Nothing names M0 as an operand, here or in
    // v_movrels_b32 or the LDS forms, so the walk counts the use where the word enters M0.
    const uint32_t relative = 0xbebc2e28u, reader = 0x7e00023cu;
    const auto indexed = program({.in_loop = {0xbefc0312u, relative, reader}});
    ASSERT_EQ(at(indexed, 5).opcode, kSop1OpcodeMovB32);
    ASSERT_EQ(at(indexed, 5).dst.value, 124);
    ASSERT_EQ(at(indexed, 5).src[0].value, 18);
    ASSERT_EQ(at(indexed, 6).opcode, kSop1OpcodeMovrelsB32);
    EXPECT_TRUE(flagged(indexed));
    uint32_t pc = 0;
    EXPECT_EQ(numeric_blocker(indexed, &pc), "derived-value-enters-m0");
    EXPECT_EQ(pc, 5u);
    // Through a copy, and through arithmetic: s_mov_b32 s20, s18; s_add_i32 m0, s20, 1.
    const auto computed = program({.in_loop = {0xbe940312u, 0x817c8114u}});
    ASSERT_EQ(at(computed, 6).fmt, Rdna2Format::SOP2);
    ASSERT_EQ(at(computed, 6).dst.value, 124);
    ASSERT_EQ(at(computed, 6).src[0].value, 20);
    EXPECT_EQ(numeric_blocker(computed, &pc), "derived-value-enters-m0");
    EXPECT_EQ(pc, 6u);
    // Control: writing M0 is not the objection. s_mov_b32 m0, s12 moves a register the load
    // never produced, and the same relative read and reader are then cleared.
    const auto unrelated = program({.in_loop = {0xbefc030cu, relative, reader}});
    ASSERT_EQ(at(unrelated, 5).dst.value, 124);
    ASSERT_EQ(at(unrelated, 5).src[0].value, 12);
    EXPECT_FALSE(flagged(unrelated)) << numeric_blocker(unrelated);
}

TEST(RawWideReplayedLoad, OnlyARealSccWriterEndsACompareOnALoadedWord) {
    // s_cmp_eq_u32 s18, 0; (one instruction); s_cbranch_scc1. The compare puts a loaded word's
    // truth value in SCC and the branch consumes it, so the load is numeric data unless the
    // instruction in between REPLACES SCC. The walk used to treat every SOP1 that is not on the
    // short leaves-SCC-unmodified list, and two SOP2 packs, as a replacement. These three leave
    // SCC alone and all three are lowered (rdna2_movrels.cpp; rdna2_emit_alu.cpp for S_FF1 and
    // the pack family).
    const uint32_t compare = 0xbf068012u, branch = 0xbf850000u;
    struct Between {
        const char* name;
        uint32_t word;
        Rdna2Format fmt;
        uint32_t opcode;
        bool replaces_scc;
    };
    const Between cases[] = {
        {"s_movrels_b32 s60, s40", 0xbebc2e28u, Rdna2Format::SOP1, kSop1OpcodeMovrelsB32, false},
        {"s_ff1_i32_b32 s60, s40", 0xbebc1328u, Rdna2Format::SOP1, 0x13u, false},
        {"s_pack_lh_b32_b16 s60, s40, s41", 0x99bc2928u, Rdna2Format::SOP2, 0x33u, false},
        // Controls: these do write SCC, from registers the load never produced, so the branch no
        // longer depends on the loaded word and the loop is cleared.
        {"s_not_b32 s60, s40", 0xbebc0728u, Rdna2Format::SOP1, 0x07u, true},
        {"s_cmp_eq_i32 s5, 0", 0xbf008005u, Rdna2Format::SOPC, 0x00u, true},
    };
    {
        const auto direct = program({.in_loop = {compare, branch}});
        ASSERT_EQ(at(direct, 5).fmt, Rdna2Format::SOPC);
        ASSERT_EQ(at(direct, 5).src[0].value, 18);
        ASSERT_EQ(at(direct, 6).fmt, Rdna2Format::SOPP);
        ASSERT_EQ(at(direct, 6).opcode, 0x05u);
        uint32_t pc = 0;
        EXPECT_EQ(numeric_blocker(direct, &pc), "scc-branch-on-derived-value");
        EXPECT_EQ(pc, 6u);
    }
    for (const Between& between : cases) {
        const auto instructions = program({.in_loop = {compare, between.word, branch}});
        ASSERT_EQ(at(instructions, 6).fmt, between.fmt) << between.name;
        ASSERT_EQ(at(instructions, 6).opcode, between.opcode) << between.name;
        ASSERT_EQ(at(instructions, 7).opcode, 0x05u) << between.name;
        if (between.replaces_scc) {
            EXPECT_FALSE(flagged(instructions))
                << between.name << ": " << numeric_blocker(instructions);
            continue;
        }
        uint32_t pc = 0;
        EXPECT_EQ(numeric_blocker(instructions, &pc), "scc-branch-on-derived-value")
            << between.name;
        EXPECT_EQ(pc, 7u) << between.name;
    }
}

TEST(RawWideReplayedLoad, AUnaryMaskTransferKeepsExecIndependent) {
    // #4555. `s_wqm_b64 exec, exec` is how nearly every pixel shader begins. It derives EXEC
    // from EXEC, so an independent EXEC stays independent, and the compare that later recycles
    // the V#'s pair is still a fresh mask. While only s_mov_b64 was a recognised unary transfer,
    // this instruction made the walk distrust every compare after it, for any load fetched
    // above it. Here it sits in the loop, below the load; in GTA V the V# loads are at pc 4
    // and the prologue at pc 9.
    const auto wqm = program({.in_loop = {0xbefe0a7eu}});   // s_wqm_b64 exec, exec
    ASSERT_EQ(at(wqm, 5).fmt, Rdna2Format::SOP1);
    ASSERT_EQ(at(wqm, 5).opcode, kSop1OpcodeWqmB64);
    ASSERT_EQ(at(wqm, 5).dst.value, 126);
    ASSERT_EQ(at(wqm, 5).src[0].value, 126);
    EXPECT_FALSE(flagged(wqm)) << numeric_blocker(wqm);
    EXPECT_TRUE(rdna2_raw_wave_wide_data_loads(wqm).empty());

    const auto inverted = program({.in_loop = {0xbefe087eu}});   // s_not_b64 exec, exec
    ASSERT_EQ(at(inverted, 5).opcode, kSop1OpcodeNotB64);
    EXPECT_FALSE(flagged(inverted)) << numeric_blocker(inverted);

    // From the recycled pair itself, whose high word still MAY be the load's: the transfer is
    // from an independent root, so EXEC stays independent and the loop is cleared.
    const auto from_fresh = program({.after_mask = {0xbefe0a10u}});   // s_wqm_b64 exec, s[16:17]
    ASSERT_EQ(at(from_fresh, 8).opcode, kSop1OpcodeWqmB64);
    ASSERT_EQ(at(from_fresh, 8).src[0].value, 16);
    EXPECT_FALSE(flagged(from_fresh)) << numeric_blocker(from_fresh);

    // Control: the transfer is only as independent as its source. s_wqm_b64 exec, s[18:19] makes
    // EXEC out of two loaded words, and that is a use of them.
    const auto from_load = program({.in_loop = {0xbefe0a12u}});
    ASSERT_EQ(at(from_load, 5).opcode, kSop1OpcodeWqmB64);
    ASSERT_EQ(at(from_load, 5).src[0].value, 18);
    EXPECT_TRUE(flagged(from_load));
    uint32_t pc = 0;
    EXPECT_EQ(numeric_blocker(from_load, &pc), "derived-value-leaves-scalar-data");
    EXPECT_EQ(pc, 5u);
}

TEST(RawWideReplayedLoad, AUnaryMaskTransferIntoVccReplacesWhatVccHeld) {
    // s_mov_b64 vcc, s[18:19] ; (compare into s[16:17]) ; s_wqm_b64 vcc, s[16:17] ;
    // s_cbranch_vccz. VCC first holds two loaded words, then the fresh compare's mask, and the
    // branch reads the second. This is only true because the emitter makes S_WQM into VCC a VCC
    // write (ComputeWqmIntoVcc pins that on the GPU); it used to leave the branch on the stale
    // VCC, and with that the classification below would have been wrong.
    const auto replaced =
        program({.in_loop = {0xbeea0412u}, .move_mask_to_vcc = false, .after_mask = {0xbeea0a10u}});
    ASSERT_EQ(at(replaced, 5).opcode, kSop1OpcodeMovB64);
    ASSERT_EQ(at(replaced, 5).dst.value, 106);
    ASSERT_EQ(at(replaced, 5).src[0].value, 18);
    ASSERT_EQ(at(replaced, 8).opcode, kSop1OpcodeWqmB64);
    ASSERT_EQ(at(replaced, 8).dst.value, 106);
    ASSERT_EQ(at(replaced, 8).src[0].value, 16);
    ASSERT_EQ(at(replaced, 9).fmt, Rdna2Format::SOPP);
    EXPECT_FALSE(flagged(replaced)) << numeric_blocker(replaced);
    // Control: without the S_WQM the branch reads the loaded words the move put in VCC.
    const auto stale = program({.in_loop = {0xbeea0412u}, .move_mask_to_vcc = false});
    uint32_t pc = 0;
    EXPECT_EQ(numeric_blocker(stale, &pc), "implicit-vcc-reader");
    EXPECT_EQ(pc, 8u);
}

TEST(RawWideReplayedLoad, ACarryInFromAFreshCompareIsNotANumericRead) {
    // #4555, the MOUSE: P.I. For Hire program. v_addc_co_u32 v3, s[44:45], 0, v3, s[16:17]
    // (VOP3B 0x128) takes its carry-in from the pair the compare just recycled. The emitter
    // consumes that operand as a Bool and refuses an untracked one. v_cndmask's condition was
    // already exempt here when it is an independent root; the carry-in forms were not.
    const std::vector<uint32_t> carry_in{0xd5282c03u, 0x00420680u};
    const auto fresh = program({.after_mask = carry_in});
    ASSERT_EQ(at(fresh, 8).fmt, Rdna2Format::VOP3);
    ASSERT_EQ(at(fresh, 8).opcode, 0x128u);
    ASSERT_EQ(at(fresh, 8).src[2].kind, OperandKind::SGPR);
    ASSERT_EQ(at(fresh, 8).src[2].value, 16);
    ASSERT_EQ(at(fresh, 8).sdst.value, 44);
    EXPECT_FALSE(flagged(fresh)) << numeric_blocker(fresh);

    // Control: with no compare the pair still holds the load's own words, and a carry taken
    // from them is a read of the load.
    const auto loaded =
        program({.compare = false, .move_mask_to_vcc = false, .after_mask = carry_in});
    ASSERT_EQ(at(loaded, 5).opcode, 0x128u);
    EXPECT_TRUE(flagged(loaded));
    uint32_t pc = 0;
    EXPECT_EQ(numeric_blocker(loaded, &pc), "numeric-reader");
    EXPECT_EQ(pc, 5u);
}

TEST(RawWideReplayedLoad, ADefiniteLaneReadEndsTheMarkOnItsDestination) {
    // s_mov_b32 s20, s19 ; v_readfirstlane s20, v1 ; v_mov_b32 v0, s20. The copy carries a
    // loaded word into s20, and v_readfirstlane then replaces s20 whatever EXEC is. Only
    // v_readlane was treated as a definite write, so s20 kept the load's mark and the v_mov
    // was a numeric reader (#4555, the MOUSE program: vcc_hi rewritten at its pc 390).
    const uint32_t copy = 0xbe940313u, lane_read = 0x7e280501u, reader = 0x7e000214u;
    const auto replaced = program({.in_loop = {copy, lane_read, reader}});
    ASSERT_EQ(at(replaced, 5).dst.value, 20);
    ASSERT_EQ(at(replaced, 5).src[0].value, 19);
    ASSERT_EQ(at(replaced, 6).fmt, Rdna2Format::VOP1);
    ASSERT_EQ(at(replaced, 6).opcode, 0x02u);
    ASSERT_EQ(at(replaced, 6).dst.value, 20);
    ASSERT_EQ(at(replaced, 7).src[0].value, 20);
    EXPECT_FALSE(flagged(replaced)) << numeric_blocker(replaced);
    // Control: without the lane read the v_mov reads the copied word.
    const auto copied = program({.in_loop = {copy, reader}});
    uint32_t pc = 0;
    EXPECT_EQ(numeric_blocker(copied, &pc), "numeric-reader");
    EXPECT_EQ(pc, 6u);
}

namespace {
// The same loop with instructions in front of it, so the load is no longer at pc 1:
//      (prefix)
//   L  s_load_dwordx4 s[16:19], s[28:29], 0xf0
//      s_buffer_load_dwordx4 s[8:11], s[16:19], 0xc0
//      (in_loop)
//      v_cmp_*_sdwa s[16:17], 0, s10 ; s_mov_b64 vcc, s[16:17]
//      s_cbranch_vccz +1 ; s_branch L
//      v_readfirstlane vcc_lo, v1 ; s_endpgm
// Returns whether the load is flagged, and its numeric blocker through `kind`, with how far past
// the load that blocker is through `offset`.
bool prefixed_load_is_flagged(const std::vector<uint32_t>& prefix,
                              const std::vector<uint32_t>& in_loop, std::string* kind = nullptr,
                              uint32_t* offset = nullptr) {
    std::vector<uint32_t> code = prefix;
    const uint32_t load_pc = static_cast<uint32_t>(code.size());
    code.insert(code.end(), {0xf408040eu, 0xfa0000f0u, 0xf4280208u, 0xfa0000c0u});
    code.insert(code.end(), in_loop.begin(), in_loop.end());
    code.insert(code.end(), {0x7c1a14f9u, 0x86869080u, 0xbeea0410u, 0xbf860001u});
    const uint32_t branch_pc = static_cast<uint32_t>(code.size());
    code.push_back(0xbf820000u | ((load_pc - (branch_pc + 1u)) & 0xffffu));
    code.insert(code.end(), {0x7ed40501u, 0xbf810000u});
    std::vector<Rdna2Inst> instructions;
    EXPECT_EQ(rdna2_walk(code.data(), code.size(), instructions), code.size());
    if (kind) kind->clear();
    for (const RawWideLoadDiagnosis& row : rdna2_raw_wide_data_load_diagnoses(instructions))
        if (row.load_pc == load_pc) {
            if (kind) *kind = row.numeric_kind;
            if (offset) *offset = row.numeric_pc - load_pc;
        }
    const auto loads = rdna2_raw_wide_data_loads(instructions);
    return std::find(loads.begin(), loads.end(), load_pc) != loads.end();
}
constexpr uint32_t kSaveExecInVcc = 0xbeea047eu;   // s_mov_b64 vcc, exec
constexpr uint32_t kRestoreExecFromVcc = 0xbefe046au;   // s_mov_b64 exec, vcc
constexpr uint32_t kSaveExecInS96 = 0xbee0047eu;   // s_mov_b64 s[96:97], exec
constexpr uint32_t kRestoreExecFromS96 = 0xbefe0460u;   // s_mov_b64 exec, s[96:97]
}   // namespace

TEST(RawWideReplayedLoad, ExecSavedInVccBeforeTheLoadIsAnIndependentRoot) {
    // #4555, the second GTA V program: `s_mov_b64 vcc, exec` above a guarded sample and
    // `s_mov_b64 exec, vcc` after it. The seeds only recognised a save into s0..s104, so the
    // restore made EXEC "dependent" and every compare after it stopped counting as fresh.
    std::string kind;
    EXPECT_FALSE(prefixed_load_is_flagged({kSaveExecInVcc}, {kRestoreExecFromVcc}, &kind)) << kind;
    // Control: with nothing saved, the same restore takes EXEC from an unknown VCC.
    EXPECT_TRUE(prefixed_load_is_flagged({0x7e020280u}, {kRestoreExecFromVcc}));
    // And VCC stops being the save when something writes it. A compare with the implicit
    // destination is not in the explicit writer inventory, so the seed pass names it itself.
    EXPECT_TRUE(prefixed_load_is_flagged({kSaveExecInVcc, 0x7d820204u /* v_cmp_lt_u32 vcc */},
                                         {kRestoreExecFromVcc}));
    // So is the carry-out of an e32 add-with-carry: v_addc_co_u32 v4, vcc, 0, v3, vcc.
    constexpr uint32_t kAddWithCarry = 0x50080680u;
    std::vector<Rdna2Inst> carry;
    ASSERT_EQ(rdna2_walk(&kAddWithCarry, 1, carry), 1u);
    ASSERT_EQ(carry.front().fmt, Rdna2Format::VOP2);
    ASSERT_EQ(carry.front().opcode, 0x28u);
    EXPECT_TRUE(prefixed_load_is_flagged({kSaveExecInVcc, kAddWithCarry}, {kRestoreExecFromVcc}));
}

TEST(RawWideReplayedLoad, ALoopAboveTheLoadDoesNotCostItsSeeds) {
    // The seed pass was one forward sweep that gave up at the first backward branch, so any
    // loop earlier in the program (GTA V has one at its pc 149) left every later load with no
    // saved-EXEC fact at all. It is a fixed point now.
    //   0  s_mov_b64 s[96:97], exec
    //   1  s_nop
    //   2  s_cbranch_scc1 -2        back to pc 1
    std::string kind;
    EXPECT_FALSE(prefixed_load_is_flagged({kSaveExecInS96, 0xbf800000u, 0xbf85fffeu},
                                          {kRestoreExecFromS96}, &kind))
        << kind;
    // Control: a loop body that overwrites the saved pair leaves nothing to seed.
    //   1  s_mov_b32 s96, 0
    EXPECT_TRUE(prefixed_load_is_flagged({kSaveExecInS96, 0xbee00380u, 0xbf85fffeu},
                                         {kRestoreExecFromS96}));
    // Control: a save that only one path makes is not a fact at the join.
    //   0  s_cbranch_scc1 +1 ; 1  s_mov_b64 s[96:97], exec
    EXPECT_TRUE(prefixed_load_is_flagged({0xbf850001u, kSaveExecInS96}, {kRestoreExecFromS96}));
    // Control: the overwrite is only on the way BACK round the loop, after the exit branch, so
    // it reaches the load through the back-edge alone. A pass that skipped backward branches
    // instead of following them would still seed here.
    //   0  save ; 1  s_nop ; 2  s_cbranch_scc1 +2 (out, to the load) ; 3  s_mov_b32 s96, 0 ;
    //   4  s_branch -4 (to pc 1)
    EXPECT_TRUE(prefixed_load_is_flagged(
        {kSaveExecInS96, 0xbf800000u, 0xbf850002u, 0xbee00380u, 0xbf82fffcu},
        {kRestoreExecFromS96}));
}

TEST(RawWideReplayedLoad, OnlyAWriterOrTransferVoidsAProvenImmediateLoad) {
    // prefix; s_load_dwordx4 s[16:19], s[28:29], 0xf0; v_mov_b32 v0, s18; s_endpgm.
    // The load is numeric (the v_mov reads a loaded word) and its entry pointer is stable, so it
    // is a proven immediate load: it gets real bytes. The prefix sits before the load, where no
    // walk goes.
    const auto proven = [](uint32_t prefix_word) {
        const std::vector<uint32_t> code{prefix_word, 0xf408040eu, 0xfa0000f0u, 0x7e000212u,
                                         0xbf810000u};
        std::vector<Rdna2Inst> instructions;
        EXPECT_EQ(rdna2_walk(code.data(), code.size(), instructions), code.size());
        return rdna2_proven_raw_immediate_wide_data_loads(instructions);
    };
    const std::vector<uint32_t> kept{kLoadPc};
    EXPECT_EQ(proven(0x7e020280u), kept) << "control: v_mov_b32 v1, 0";
    for (const Escape& escape : kEscapes) {
        if (escape.writes_unnamed_or_leaves_cfg)
            EXPECT_TRUE(proven(escape.word).empty()) << escape.name;
        else
            EXPECT_EQ(proven(escape.word), kept)
                << escape.name << " writes s60 alone; it cannot move the entry pointer";
    }
}

TEST(RawWideReplayedLoad, SaveexecFromAnIndependentMaskDoesNotStopTheWalk) {
    // s_orn2/s_nand/s_nor_saveexec_b64 s[40:41], -1 (SOP1 0x28..0x2a). They were refused as
    // "relative SGPR write". They are ordinary mask transfers, and from an independent source
    // they leave EXEC independent, so the loop is still cleared.
    for (uint32_t opcode : {0x28u, 0x29u, 0x2au}) {
        const auto instructions = program({.in_loop = {0xbea800c1u | (opcode << 8u)}});
        ASSERT_EQ(at(instructions, 5).fmt, Rdna2Format::SOP1);
        ASSERT_EQ(at(instructions, 5).opcode, opcode);
        EXPECT_FALSE(flagged(instructions))
            << "opcode 0x" << std::hex << opcode << ": " << numeric_blocker(instructions);
    }
}

TEST(RawWideReplayedLoad, AnSmemThatIsNotAPlainLoadKeepsReplayUncertain) {
    // SMEM opcode 0x05 is not one of the ten loads, so the writer inventory says nothing about
    // it. Placed past the loop with operands clear of the loaded words, neither walk objects to
    // it; only the whole-program condition can.
    const auto instructions = program({.after_exit = {0xf414000eu, 0xfa000000u}});
    const Rdna2Inst& odd = at(instructions, 10);
    ASSERT_EQ(odd.fmt, Rdna2Format::SMEM);
    ASSERT_EQ(odd.opcode, 0x5u);
    ASSERT_FALSE(rdna2_may_write_guest_memory(odd)) << "else the guest-write arm covers it";
    EXPECT_TRUE(flagged(instructions));
    EXPECT_EQ(numeric_blocker(instructions), "load-re-executed");
}

TEST(RawWideReplayedLoad, LoadedWordsCopiedIntoVccAreReadByItsImplicitReaders) {
    // No compare: s_mov_b64 vcc, s[16:17] moves the V#'s own words into VCC, and the
    // s_cbranch_vccz that follows branches on them. Nothing names VCC as an operand (#4527).
    const auto branch = program({.compare = false});
    ASSERT_EQ(at(branch, 5).fmt, Rdna2Format::SOP1);   // the move
    ASSERT_EQ(at(branch, 6).fmt, Rdna2Format::SOPP);   // s_cbranch_vccz
    EXPECT_TRUE(flagged(branch));
    uint32_t pc = 0;
    EXPECT_EQ(numeric_blocker(branch, &pc), "implicit-vcc-reader");
    EXPECT_EQ(pc, 6u);

    // v_cndmask_b32 v2, v0, v3 (e32: the mask is VCC, implicitly) ahead of the branch.
    const auto select = program({.compare = false, .after_mask = {0x02040700u}});
    ASSERT_EQ(at(select, 6).fmt, Rdna2Format::VOP2);
    ASSERT_EQ(at(select, 6).opcode, 0x1u);
    EXPECT_TRUE(flagged(select));
    EXPECT_EQ(numeric_blocker(select, &pc), "implicit-vcc-reader");
    EXPECT_EQ(pc, 6u);
}

TEST(RawWideReplayedLoad, AFreshMaskInVccIsNotAnImplicitRead) {
    // The live shape, plus the same e32 select: VCC was just written from the compare's pair,
    // so the select and the branch consume a mask, not the load.
    const auto instructions = program({.after_mask = {0x02040700u}});
    ASSERT_EQ(at(instructions, 8).fmt, Rdna2Format::VOP2);
    EXPECT_FALSE(flagged(instructions)) << numeric_blocker(instructions);
}

TEST(RawWideReplayedLoad, CopyCarriedRoundTheBackEdgeReachesItsReader) {
    // v_mov_b32 v0, s40 then s_mov_b32 s40, s18. On the first iteration s40 is unrelated to the
    // load when it is read; the copy only reaches the reader on the NEXT one. The old rule gave
    // up at the back-edge; the walk now goes round and names the instruction that reads it.
    const auto instructions = program({.in_loop = {0x7e000228u, 0xbea80312u}});
    ASSERT_EQ(at(instructions, 5).fmt, Rdna2Format::VOP1);
    ASSERT_EQ(at(instructions, 5).src[0].value, 40);
    ASSERT_EQ(at(instructions, 6).fmt, Rdna2Format::SOP1);
    ASSERT_EQ(at(instructions, 6).dst.value, 40);
    ASSERT_EQ(at(instructions, 6).src[0].value, 18);
    EXPECT_TRUE(flagged(instructions));
    uint32_t pc = 0;
    EXPECT_EQ(numeric_blocker(instructions, &pc), "numeric-reader");
    EXPECT_EQ(pc, 5u);
}

TEST(RawWideReplayedLoad, MaskFactsNarrowedByTheLoopReachTheNextIteration) {
    // v_cndmask_b32_e64 v0, 0, 1.0, s[16:17]: a Bool consumer of the freshly compared pair. On
    // the first iteration the compare is proven independent of the load, so this reads a mask.
    const std::vector<uint32_t> select{0xd5010000u, 128u | (242u << 9u) | (16u << 18u)};
    const auto steady = program({.move_mask_to_vcc = false, .after_mask = select});
    ASSERT_EQ(at(steady, 7).fmt, Rdna2Format::VOP3);
    ASSERT_EQ(at(steady, 7).opcode, 0x101u) << "v_cndmask_b32";
    EXPECT_FALSE(flagged(steady)) << numeric_blocker(steady);

    // s_mov_b64 exec, s[40:41] at the end of the body: EXEC now comes from an uncertified pair,
    // so on the NEXT iteration the same compare proves nothing and the select may observe the
    // reloaded V#'s second word. That is only visible if the replayed load's words are derived
    // again and the walk re-enters the body with the narrowed mask facts.
    auto narrowed_body = select;
    narrowed_body.push_back(0xbefe0428u);
    const auto narrowed = program({.move_mask_to_vcc = false, .after_mask = narrowed_body});
    ASSERT_EQ(at(narrowed, 9).fmt, Rdna2Format::SOP1);
    ASSERT_EQ(at(narrowed, 9).src[0].value, 40);
    EXPECT_TRUE(flagged(narrowed));
    uint32_t pc = 0;
    EXPECT_EQ(numeric_blocker(narrowed, &pc), "numeric-reader");
    EXPECT_EQ(pc, 7u) << "the select, on its second visit";
}

TEST(RawWideReplayedLoad, DiagnosisNamesBothBlockers) {
    // The row a title investigation reads. On a load that is still refused it names the
    // instruction that made the cheaper walk unsure -- here the B64 move of the recycled pair,
    // whose high word that walk cannot expire without knowing the wave width -- and what stopped
    // the numeric walk.
    const auto rows =
        rdna2_raw_wide_data_load_diagnoses(program({.load_offset_word = 0x28000000u}));
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].load_pc, kLoadPc);
    EXPECT_STREQ(rows[0].backing_kind, "data-read");
    EXPECT_EQ(rows[0].backing_pc, 7u) << "s_mov_b64 vcc, s[16:17]";
    EXPECT_STREQ(rows[0].numeric_kind, "load-re-executed");
    EXPECT_EQ(rows[0].numeric_pc, kLoadPc);
    // A cleared load has no row at all.
    EXPECT_TRUE(rdna2_raw_wide_data_load_diagnoses(program({})).empty());
}

TEST(RawWideReplayedLoad, APlainCopyOfLoadedWordsIntoExecIsANumericUse) {
    // #4574. s_mov_b64 exec, s[18:19] ; v_mov_b32 v0, v1 ; s_mov_b64 exec, -1. Which lanes the
    // v_mov writes is decided by two words of the load, and nothing names EXEC as an operand, so
    // the walk used to mark EXEC, find no reader, and clear the load when EXEC was put back.
    const uint32_t lane_write = 0x7e000301u, all_lanes = 0xbefe04c1u;
    const auto copied = program({.in_loop = {0xbefe0412u, lane_write, all_lanes}});
    ASSERT_EQ(at(copied, 5).opcode, kSop1OpcodeMovB64);
    ASSERT_EQ(at(copied, 5).dst.value, 126);
    ASSERT_EQ(at(copied, 5).src[0].value, 18);
    ASSERT_EQ(at(copied, 6).fmt, Rdna2Format::VOP1);
    ASSERT_EQ(at(copied, 7).dst.value, 126);
    EXPECT_TRUE(flagged(copied));
    uint32_t pc = 0;
    EXPECT_EQ(numeric_blocker(copied, &pc), "derived-value-enters-exec");
    EXPECT_EQ(pc, 5u);
    // One half is enough, either half: s_mov_b32 exec_lo, s18 and s_mov_b32 exec_hi, s19.
    for (const uint32_t half : {0xbefe0312u, 0xbeff0313u}) {
        const auto one_word = program({.in_loop = {half, lane_write, all_lanes}});
        ASSERT_EQ(at(one_word, 5).opcode, kSop1OpcodeMovB32);
        EXPECT_TRUE(flagged(one_word)) << std::hex << half;
        EXPECT_EQ(numeric_blocker(one_word, &pc), "derived-value-enters-exec") << std::hex << half;
        EXPECT_EQ(pc, 5u);
    }
    // Control: moving a pair into EXEC is not the objection. s[40:41] is nothing the load
    // produced, and the same body is cleared.
    const auto unrelated = program({.in_loop = {0xbefe0428u, lane_write, all_lanes}});
    ASSERT_EQ(at(unrelated, 5).src[0].value, 40);
    EXPECT_FALSE(flagged(unrelated)) << numeric_blocker(unrelated);
    // Control: nor is restoring a saved EXEC, the move every guarded sample ends with.
    const auto restored = program(
        {.first_word = kSaveExecInS96, .in_loop = {kRestoreExecFromS96, lane_write, all_lanes}});
    EXPECT_FALSE(flagged(restored)) << numeric_blocker(restored);
    // Control: nor a move from the recycled pair once a fresh compare has made it a mask. Its
    // high word still MAY be the load's, and the move marks EXEC's accordingly, but the source
    // is an independent root and what the emitter installs is that root's Bool.
    const auto from_fresh = program({.after_mask = {0xbefe0410u}});   // s_mov_b64 exec, s[16:17]
    ASSERT_EQ(at(from_fresh, 8).opcode, kSop1OpcodeMovB64);
    ASSERT_EQ(at(from_fresh, 8).dst.value, 126);
    ASSERT_EQ(at(from_fresh, 8).src[0].value, 16);
    EXPECT_FALSE(flagged(from_fresh)) << numeric_blocker(from_fresh);
    // The same move when the compare was NOT fresh, because EXEC had been taken from a pair
    // nothing vouches for: s_mov_b64 exec, s[40:41] ; (compare) ; s_mov_b64 exec, s[16:17] ;
    // then EXEC and the loaded words are put back. The pair is no root, and its high word is
    // still marked, since a 32-lane compare writes only the low one. So this stops, which is
    // right at 32 lanes and conservative at 64: the walk does not know the width (#4555).
    const auto not_fresh =
        program({.in_loop = {0xbefe0428u},
                 .move_mask_to_vcc = false,
                 .after_mask = {0xbefe0410u, all_lanes, 0xbe900480u, 0xbe920480u}});
    ASSERT_EQ(at(not_fresh, 8).dst.value, 126);
    ASSERT_EQ(at(not_fresh, 8).src[0].value, 16);
    EXPECT_TRUE(flagged(not_fresh));
    EXPECT_EQ(numeric_blocker(not_fresh, &pc), "derived-value-enters-exec");
    EXPECT_EQ(pc, 8u);
    // And through a register the decoder does not call an SGPR: s_mov_b64 vcc, s[18:19] ;
    // s_mov_b64 exec, vcc.
    const auto through_vcc = program({.in_loop = {0xbeea0412u, kRestoreExecFromVcc}});
    ASSERT_EQ(at(through_vcc, 5).dst.value, 106);
    ASSERT_EQ(at(through_vcc, 6).src[0].kind, OperandKind::Special);
    ASSERT_EQ(at(through_vcc, 6).src[0].value, 106);
    EXPECT_EQ(numeric_blocker(through_vcc, &pc), "derived-value-enters-exec");
    EXPECT_EQ(pc, 6u);
}

TEST(RawWideReplayedLoad, ALoadIntoTheSpecialRegistersIsNumericByItself) {
    // s_load_dwordx4 s[124:127], s[28:29], 0xf0 loads M0 and EXEC themselves. The walk starts
    // every load from an EXEC that does not depend on it, which is false here, and the only
    // thing that used to stop it was the word in M0. Overwrite M0 first and the lanes written
    // under the loaded EXEC went unnoticed:
    //   0  s_load_dwordx4 s[124:127], s[28:29], 0xf0
    //   2  s_mov_b32 m0, 0
    //   3  v_mov_b32 v0, v1
    //   4  s_mov_b64 exec, -1
    //   5  v_readfirstlane vcc_lo, v1 ; s_endpgm
    const std::vector<uint32_t> code{0xf4081f0eu, 0xfa0000f0u, 0xbefc0380u, 0x7e000301u,
                                     0xbefe04c1u, 0x7ed40501u, 0xbf810000u};
    std::vector<Rdna2Inst> instructions;
    ASSERT_EQ(rdna2_walk(code.data(), code.size(), instructions), code.size());
    ASSERT_EQ(at(instructions, 0).fmt, Rdna2Format::SMEM);
    ASSERT_EQ(at(instructions, 0).dst.value, 124);
    ASSERT_EQ(at(instructions, 2).dst.value, 124);
    ASSERT_EQ(at(instructions, 4).dst.value, 126);
    const auto rows = rdna2_raw_wide_data_load_diagnoses(instructions);
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_STREQ(rows[0].backing_kind, "destination-above-s105");
    EXPECT_STREQ(rows[0].numeric_kind, "destination-above-s105");
    EXPECT_EQ(rows[0].numeric_pc, 0u);
    const auto loads = rdna2_raw_wide_data_loads(instructions);
    EXPECT_EQ(loads, std::vector<uint32_t>{0u});
}

TEST(RawWideReplayedLoad, AMaskFormedUnderCopiedExecCannotReachASavedExecSeed) {
    // The second case on #4574, from the re-review of #4569. The copy does more than predicate
    // lanes: a compare run under that EXEC produces a mask that depends on the load and carries
    // no mark. Here it is moved into the pair the load's seed names, after the walk has run out
    // of derived words, and the loop brings it back to the load as an "independent" EXEC.
    //    0  s_mov_b64 s[96:97], exec            the seed
    //    1  s_load_dwordx4 s[16:19], ...
    //    3  s_buffer_load_dwordx4 s[8:11], s[16:19], 0xc0
    //    5  s_mov_b64 exec, s[96:97]            restore from the seeded pair
    //    6  v_cmp_*_sdwa s[16:17], 0, s10       counted as fresh
    //    8  v_cndmask_b32 v0, 0, 1.0, s[16:17]  an exempt Bool consumer
    //   10  s_mov_b64 exec, s[18:19]            EXEC := two loaded words
    //   11  v_cmp_*_sdwa s[40:41], 0, s10       a mask formed under them
    //   13  s_mov_b64 exec, -1
    //   14  s_mov_b64 s[16:17], 0
    //   15  s_mov_b64 s[18:19], 0               nothing derived is left
    //   16  s_mov_b64 s[96:97], s[40:41]        the seeded pair now depends on the load
    //   17  s_cbranch_vccz +1 ; s_branch 1
    const auto instructions =
        program({.first_word = kSaveExecInS96,
                 .in_loop = {kRestoreExecFromS96, 0x7c1a14f9u, 0x86869080u, 0xd5010000u,
                             0x0041e480u, 0xbefe0412u, 0x7c1a14f9u, 0x8686a880u, 0xbefe04c1u,
                             0xbe900480u, 0xbe920480u, 0xbee00428u},
                 .compare = false,
                 .move_mask_to_vcc = false});
    ASSERT_EQ(at(instructions, 6).fmt, Rdna2Format::VOPC);
    ASSERT_EQ(at(instructions, 6).dst.value, 16);
    ASSERT_EQ(at(instructions, 8).fmt, Rdna2Format::VOP3);
    ASSERT_EQ(at(instructions, 8).opcode, 0x101u);
    ASSERT_EQ(at(instructions, 10).dst.value, 126);
    ASSERT_EQ(at(instructions, 10).src[0].value, 18);
    ASSERT_EQ(at(instructions, 11).dst.value, 40);
    ASSERT_EQ(at(instructions, 14).dst.value, 16);
    ASSERT_EQ(at(instructions, 15).dst.value, 18);
    ASSERT_EQ(at(instructions, 16).dst.value, 96);
    ASSERT_EQ(at(instructions, 16).src[0].value, 40);
    EXPECT_TRUE(flagged(instructions));
    uint32_t pc = 0;
    EXPECT_EQ(numeric_blocker(instructions, &pc), "derived-value-enters-exec");
    EXPECT_EQ(pc, 10u);
}

TEST(RawWideReplayedLoad, ACarryOutEndsAVccRootUnlessEverythingItIsMadeOfIsOne) {
    // v_addc_co_u32 v4, vcc, 0, v3, vcc writes its carry-out to VCC and names VCC nowhere a
    // writer inventory looks. The pass that seeds a load knew that; the walk did not, so a VCC
    // holding a saved EXEC went on counting as one after a carry-out had replaced it. In the
    // loop the instructions sit at these offsets from the load:
    //   +4 (in_loop) ... v_cmp s[16:17] ; s_mov_b64 vcc, s[16:17] ; s_cbranch_vccz ; s_branch
    constexpr uint32_t kAddWithCarry = 0x50080680u, kExecFromUnknownPair = 0xbefe0428u;
    std::string kind;
    uint32_t offset = 0;
    // Formed under an EXEC nothing vouches for, the carry-out is not an independent root. The
    // EXEC then taken from it makes the loop's compare prove nothing, and the branch on the
    // recycled pair is the reader.
    EXPECT_TRUE(prefixed_load_is_flagged({kSaveExecInVcc},
                                         {kExecFromUnknownPair, kAddWithCarry, kRestoreExecFromVcc},
                                         &kind, &offset));
    EXPECT_EQ(kind, "implicit-vcc-reader");
    EXPECT_EQ(offset, 10u) << "s_cbranch_vccz";
    // Control: without the carry-out VCC is still the save, and the restore is a restore.
    EXPECT_FALSE(prefixed_load_is_flagged({kSaveExecInVcc},
                                          {kExecFromUnknownPair, kRestoreExecFromVcc}, &kind))
        << kind;
    // Under an independent EXEC, with an independent root as its carry-in, the carry-out is one
    // too: ending the root at every carry-out would refuse this.
    EXPECT_FALSE(
        prefixed_load_is_flagged({kSaveExecInVcc}, {kAddWithCarry, kRestoreExecFromVcc}, &kind))
        << kind;
    // But it does not make a root out of a VCC that was not one: the carry-in is part of it.
    // Stopped in the first pass round the loop, at the branch; if the carry-out counted as a
    // root here, the walk would get round once and stop at the carry-in instead.
    EXPECT_TRUE(prefixed_load_is_flagged({0x7e020280u}, {kAddWithCarry, kRestoreExecFromVcc}, &kind,
                                         &offset));
    EXPECT_EQ(kind, "implicit-vcc-reader");
    EXPECT_EQ(offset, 9u) << "s_cbranch_vccz";
}

TEST(RawWideReplayedLoad, AtSixtyFourLanesACompareReplacesBothWordsOfThePair) {
    // #4555, MOUSE: P.I. For Hire. The walks assume a compare into a register pair may write only
    // the low word, which is what happens at 32 lanes, so the high word of a pair the shader
    // recycles as a mask is taken to still hold what was loaded. A 64-bit mask operation on that
    // pair then "reads" the loaded word, and the branch on its SCC stops the walk:
    //   .  v_cmp_*_sdwa s[16:17], 0, s10 ; s_mov_b64 vcc, s[16:17]
    //   8  s_and_b64 s[40:41], s[16:17], exec      SCC = (result != 0), over both words
    //   9  s_cbranch_scc1 +0
    // A caller that knows the program runs 64 lanes wide says so, and the compare is then a write
    // of both words.
    const auto mouse = program({.after_mask = {0x87a87e10u, 0xbf850000u}});
    ASSERT_EQ(at(mouse, 8).fmt, Rdna2Format::SOP2);
    ASSERT_EQ(at(mouse, 8).opcode, 0x0fu);
    ASSERT_EQ(at(mouse, 8).dst.value, 40);
    ASSERT_EQ(at(mouse, 8).src[0].value, 16);
    ASSERT_EQ(at(mouse, 9).fmt, Rdna2Format::SOPP);
    ASSERT_EQ(at(mouse, 9).opcode, 0x05u);
    uint32_t pc = 0;
    EXPECT_TRUE(flagged(mouse));
    EXPECT_EQ(numeric_blocker(mouse, &pc), "scc-branch-on-derived-value");
    EXPECT_EQ(pc, 9u);
    EXPECT_FALSE(flagged(mouse, /*wave64*/ true)) << numeric_blocker(mouse, nullptr, true);
    // The obligation that sends a draw to the owned-wave path follows: this program has a
    // v_readfirstlane, so by default the load needs logical-wave admission, and at 64 lanes not.
    EXPECT_EQ(rdna2_raw_wave_wide_data_loads(mouse), std::vector<uint32_t>{kLoadPc});
    EXPECT_TRUE(rdna2_raw_wave_wide_data_loads(mouse, /*wave64*/ true).empty());

    // The same for a compare into VCC itself, whose destination no writer inventory names:
    //   5  s_mov_b64 vcc, s[18:19]            VCC holds two loaded words
    //   6  v_cmp_lt_u32 vcc, s4, v1           at 32 lanes vcc_hi still does
    //   7  s_and_b64 s[40:41], vcc, exec ; s_cbranch_scc1 +0
    const auto through_vcc =
        program({.in_loop = {0xbeea0412u, 0x7d820204u, 0x87a87e6au, 0xbf850000u}});
    ASSERT_EQ(at(through_vcc, 6).fmt, Rdna2Format::VOPC);
    ASSERT_EQ(at(through_vcc, 6).dst.value, 106);
    ASSERT_EQ(at(through_vcc, 7).src[0].value, 106);
    EXPECT_EQ(numeric_blocker(through_vcc, &pc), "scc-branch-on-derived-value");
    EXPECT_EQ(pc, 8u);
    EXPECT_FALSE(flagged(through_vcc, /*wave64*/ true))
        << numeric_blocker(through_vcc, nullptr, true);

    // In the first program above the cheaper walk clears the load at 64 lanes by itself, since
    // nothing else reads a loaded word. (In the second it stops at the copy into VCC at any
    // width, and the numeric walk is what clears it.) Give the first something to hold on to
    // (s_mov_b32 s50, s18, a copy nobody reads, which the cheaper walk counts and the numeric
    // walk does not) and the numeric walk is the one that has to know the width.
    const auto held = program({.in_loop = {0xbeb20312u}, .after_mask = {0x87a87e10u, 0xbf850000u}});
    ASSERT_EQ(at(held, 5).opcode, kSop1OpcodeMovB32);
    ASSERT_EQ(at(held, 5).dst.value, 50);
    ASSERT_EQ(at(held, 5).src[0].value, 18);
    EXPECT_EQ(numeric_blocker(held, &pc), "scc-branch-on-derived-value");
    EXPECT_EQ(pc, 10u);
    EXPECT_FALSE(flagged(held, /*wave64*/ true)) << numeric_blocker(held, nullptr, true);

    // The cheaper walk makes the same assumption, and it is the binding one here: this load has
    // a register offset inside a loop, so the numeric walk answers "re-executed" at any width,
    // and what keeps the load flagged by default is the B64 move of the recycled pair reading a
    // high word the compare is not known to have replaced (DiagnosisNamesBothBlockers).
    const auto replayed = program({.load_offset_word = 0x28000000u});
    EXPECT_TRUE(flagged(replayed));
    EXPECT_EQ(numeric_blocker(replayed), "load-re-executed");
    EXPECT_FALSE(flagged(replayed, /*wave64*/ true));
    // A carry-out is a mask write too, and the cheaper walk treats it the same way: the pair is
    // recycled by v_addc_co_u32 v3, s[16:17], 0, v3, vcc instead of by a compare.
    const auto carried = program(
        {.load_offset_word = 0x28000000u, .in_loop = {0xd5281003u, 0x01aa0680u}, .compare = false});
    ASSERT_EQ(at(carried, 5).fmt, Rdna2Format::VOP3);
    ASSERT_EQ(at(carried, 5).opcode, 0x128u);
    ASSERT_EQ(at(carried, 5).sdst.value, 16);
    ASSERT_EQ(at(carried, 5).src[2].value, 106);
    EXPECT_TRUE(flagged(carried));
    EXPECT_FALSE(flagged(carried, /*wave64*/ true));

    // The move into EXEC of a pair a not-fresh compare recycled (the case above that stops at
    // pc 8 "right at 32 lanes and conservative at 64") is the same assumption, and clears too.
    const auto not_fresh =
        program({.in_loop = {0xbefe0428u},
                 .move_mask_to_vcc = false,
                 .after_mask = {0xbefe0410u, 0xbefe04c1u, 0xbe900480u, 0xbe920480u}});
    EXPECT_TRUE(flagged(not_fresh));
    EXPECT_FALSE(flagged(not_fresh, /*wave64*/ true)) << numeric_blocker(not_fresh, nullptr, true);
}

TEST(RawWideReplayedLoad, TheSixtyFourLaneAnswerIsNeverLargerAndKeepsRealReaders) {
    // Width knowledge only removes the "the high word may survive a compare" assumption. A load
    // with a reader that has nothing to do with a recycled pair stays numeric at any width:
    //   5  v_mov_b32 v0, s18
    const auto reader = program({.in_loop = {0x7e000212u}});
    EXPECT_TRUE(flagged(reader, /*wave64*/ true));
    uint32_t pc = 0;
    EXPECT_EQ(numeric_blocker(reader, &pc, true), "numeric-reader");
    EXPECT_EQ(pc, 5u);
    EXPECT_FALSE(rdna2_raw_wave_wide_data_loads(reader, true).empty());
    // And a plain copy of loaded words into EXEC is one at any width (#4574).
    const auto copied = program({.in_loop = {0xbefe0412u, 0x7e000301u, 0xbefe04c1u}});
    EXPECT_EQ(numeric_blocker(copied, &pc, true), "derived-value-enters-exec");
    // Over every shape this file builds a few of, the 64-lane set is a subset of the default.
    const std::vector<Shape> shapes{
        {},
        {.in_loop = {0x7e000212u}},
        {.load_offset_word = 0x28000000u},
        {.in_loop = {0xbefe0412u, 0x7e000301u, 0xbefe04c1u}},
        {.after_mask = {0x87a87e10u, 0xbf850000u}},
        {.in_loop = {0xbeea0412u, 0x7d820204u, 0x87a87e6au, 0xbf850000u}},
        {.in_loop = {0xbefe0428u},
         .move_mask_to_vcc = false,
         .after_mask = {0xbefe0410u, 0xbefe04c1u, 0xbe900480u, 0xbe920480u}},
        {.compare = false},
    };
    for (size_t index = 0; index < shapes.size(); ++index) {
        const auto instructions = program(shapes[index]);
        const auto narrow = rdna2_raw_wide_data_loads(instructions, true);
        const auto wide = rdna2_raw_wide_data_loads(instructions);
        EXPECT_TRUE(std::includes(wide.begin(), wide.end(), narrow.begin(), narrow.end()))
            << "shape " << index;
    }
}
