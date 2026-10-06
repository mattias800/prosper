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
std::string numeric_blocker(const std::vector<Rdna2Inst>& instructions, uint32_t* pc = nullptr) {
    for (const RawWideLoadDiagnosis& row : rdna2_raw_wide_data_load_diagnoses(instructions))
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

bool flagged(const std::vector<Rdna2Inst>& instructions) {
    const auto loads = rdna2_raw_wide_data_loads(instructions);
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
// Returns whether the load is flagged, and its numeric blocker through `kind`.
bool prefixed_load_is_flagged(const std::vector<uint32_t>& prefix,
                              const std::vector<uint32_t>& in_loop, std::string* kind = nullptr) {
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
        if (row.load_pc == load_pc && kind) *kind = row.numeric_kind;
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
