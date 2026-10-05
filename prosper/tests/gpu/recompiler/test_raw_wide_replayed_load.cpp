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
    // False for the two source-relative moves: they read through M0 and write a named register.
    bool writes_unnamed_or_leaves_cfg = true;
};
// One encoding of every instruction rdna2_escapes_decoded_effects names.
constexpr Escape kEscapes[] = {
    {"s_setpc_b64 s[60:61]", 0xbe80203cu, Rdna2Format::SOP1, kSop1OpcodeSetpcB64},
    {"s_swappc_b64 s[60:61], s[40:41]", 0xbebc2128u, Rdna2Format::SOP1, kSop1OpcodeSwappcB64},
    {"s_rfe_b64 s[60:61]", 0xbe80223cu, Rdna2Format::SOP1, kSop1OpcodeRfeB64},
    {"s_movrels_b32 s60, s40", 0xbebc2e28u, Rdna2Format::SOP1, kSop1OpcodeMovrelsB32, false},
    {"s_movrels_b64 s[60:61], s[40:41]", 0xbebc2f28u, Rdna2Format::SOP1, kSop1OpcodeMovrelsB64,
     false},
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
    // between executions. A source-relative move cannot, and must not cost the load its proof:
    // s_movrels_b32 is an instruction the emitter lowers.
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
    // In the loop body, with the loaded words live. Every escape stops here, the source-relative
    // moves included: SGPR[src + M0] may be one of those words. Before #4529 only the transfers
    // and the call stopped the numeric walk, and only the transfers stopped the cheap one.
    for (const Escape& escape : kEscapes) {
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
                << escape.name << " reads through M0; it cannot move the entry pointer";
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
