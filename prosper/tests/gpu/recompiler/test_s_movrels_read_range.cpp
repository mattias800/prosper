// #4538: s_movrels_b32 sD, sS reads SGPR[S + M0], and the lowering (rdna2_movrels.cpp) is a select
// over every register from sS up to s105. The analyses that decide what a program reads charged it
// sS alone, so a register above the base could be proven dead across a merge -- and a scratch write
// hardware skips could be run unconditionally in front of the relative read that observes it.
//
// Every case here is a pair: the relative read, and the same program with an ordinary instruction
// in its place, so no arm can pass because the proof it asks about never fires at all.
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t kMovrelsB32S60S40 = 0xbebc2e28u;   // s_movrels_b32 s60, s40
constexpr uint32_t kMovB32S60S40 = 0xbebc0328u;   // s_mov_b32 s60, s40
constexpr uint32_t kEnd = 0xbf810000u;   // s_endpgm

std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> result;
    EXPECT_EQ(rdna2_walk(code.data(), code.size(), result), code.size());
    return result;
}

Rdna2Inst one(uint32_t word) {
    return rdna2_decode_one(&word, 1);
}

bool linearized(const std::vector<uint32_t>& code, uint32_t branch_pc) {
    const auto safe = safe_execz_branches_for_test(code.data(), code.size());
    return std::find(safe.begin(), safe.end(), branch_pc) != safe.end();
}

}   // namespace

TEST(SMovrelsReadRange, WidthIsEveryRegisterTheLoweringMayRead) {
    const Rdna2Inst relative = one(kMovrelsB32S60S40);
    ASSERT_EQ(relative.fmt, Rdna2Format::SOP1);
    ASSERT_EQ(relative.opcode, kSop1OpcodeMovrelsB32);
    ASSERT_EQ(relative.src[0].kind, OperandKind::SGPR);
    ASSERT_EQ(relative.src[0].value, 40);
    ASSERT_EQ(relative.dst.value, 60);
    EXPECT_EQ(scalar_alu_source_words(relative, 0), 66u) << "s40..s105";
    EXPECT_EQ(scalar_alu_source_words(one(0xbebc2e00u), 0), 106u) << "s0..s105";
    EXPECT_EQ(scalar_alu_source_words(one(0xbebc2e69u), 0), 1u) << "s105 alone";
    // The pair form reads SGPR[S + M0] too. It is not lowered, and gets the same range.
    const Rdna2Inst pair = one(0xbebc2f28u);
    ASSERT_EQ(pair.opcode, kSop1OpcodeMovrelsB64);
    EXPECT_EQ(scalar_alu_source_words(pair, 0), 66u);
    EXPECT_EQ(scalar_write_width(pair), 2u) << "and it writes a pair";
    EXPECT_EQ(scalar_write_width(relative), 1u);

    // Controls: the range belongs to the relative SOURCE forms only.
    EXPECT_EQ(scalar_alu_source_words(one(kMovB32S60S40), 0), 1u) << "s_mov_b32";
    const Rdna2Inst relative_destination = one(0xbebc3028u);
    ASSERT_EQ(relative_destination.opcode, kSop1OpcodeMovreldB32);
    EXPECT_EQ(scalar_alu_source_words(relative_destination, 0), 1u)
        << "s_movreld_b32 reads exactly the source it names";
    // A base that is not an SGPR is refused by the lowering; nothing is claimed about it.
    const Rdna2Inst from_vcc = one(0xbebc2e6au);
    ASSERT_EQ(from_vcc.opcode, kSop1OpcodeMovrelsB32);
    ASSERT_EQ(from_vcc.src[0].value, 106);
    EXPECT_EQ(scalar_alu_source_words(from_vcc, 0), 1u);
}

TEST(SMovrelsReadRange, ARegisterInTheRangeIsNotDeadAcrossTheRead) {
    const auto relative = decode({kMovrelsB32S60S40, kEnd});
    const auto dead = [&](int reg) { return sgpr_dead_at_merge(relative, 0, reg); };
    EXPECT_FALSE(dead(40)) << "the base";
    ScalarMergeBlocker blocker;
    EXPECT_FALSE(sgpr_dead_at_merge(relative, 0, 50, ScalarMergeProof::AnyRead, &blocker))
        << "s50 is SGPR[40 + 10]";
    EXPECT_EQ(blocker.pc, 0u);
    EXPECT_FALSE(dead(105)) << "the last register the select reaches";
    // The destination is a candidate like any other, and is read before it is replaced.
    EXPECT_FALSE(dead(60));
    // Outside the range nothing changed: below the base, and VCC_LO just past s105.
    EXPECT_TRUE(dead(39));
    EXPECT_TRUE(dead(106));
    // The index. No operand names M0, here or in any of its other readers, so it is never dead.
    EXPECT_FALSE(dead(124));

    // Control: an ordinary move at the same place reads its one source and kills its destination.
    const auto ordinary = decode({kMovB32S60S40, kEnd});
    EXPECT_FALSE(sgpr_dead_at_merge(ordinary, 0, 40));
    EXPECT_TRUE(sgpr_dead_at_merge(ordinary, 0, 50));
    EXPECT_TRUE(sgpr_dead_at_merge(ordinary, 0, 60));
}

TEST(SMovrelsReadRange, ASkippedScratchWriteInTheRangeIsNotLinearized) {
    // pc 0  s_cbranch_execz +2          hardware skips the write when EXEC is empty
    // pc 1  s_mov_b32 s5, 0x1f4
    // pc 3  (reader)
    // pc 4  v_mov_b32 v0, s0
    // pc 5  s_endpgm
    // Linearizing drops the branch and runs the scalar write unconditionally, which is only sound
    // when nothing after the merge can read s5.
    const auto program = [](uint32_t reader) {
        return std::vector<uint32_t>{0xbf880002u, 0xbe8503ffu, 0x000001f4u,
                                     reader,      0x7e000200u, kEnd};
    };
    {
        const auto shape = decode(program(0xbe802e04u));
        ASSERT_EQ(shape[0].fmt, Rdna2Format::SOPP);
        ASSERT_EQ(shape[0].opcode, 0x08u);
        ASSERT_EQ(shape[0].pc + shape[0].len_dwords + shape[0].simm16, 3u);
        ASSERT_EQ(shape[1].dst.value, 5);
        ASSERT_EQ(shape[2].opcode, kSop1OpcodeMovrelsB32);
        ASSERT_EQ(shape[2].src[0].value, 4);
    }
    EXPECT_FALSE(linearized(program(0xbe802e04u), 0))
        << "s_movrels_b32 s0, s4 reads s5 when M0 is 1";
    // Controls. A relative read whose range starts above s5 cannot observe it, and an ordinary
    // move never could; both keep the branch linearizable, so the refusal above is the range.
    EXPECT_TRUE(linearized(program(0xbe802e06u), 0)) << "s_movrels_b32 s0, s6";
    EXPECT_TRUE(linearized(program(0xbe800304u), 0)) << "s_mov_b32 s0, s4";
}

TEST(SMovrelsReadRange, ASkippedWriteOfTheIndexIsNotLinearized) {
    // pc 0  s_cbranch_execz +2
    // pc 1  s_buffer_load_dword (destination), s[8:11], null
    // pc 3  s_movrels_b32 s0, s4
    // pc 4  v_mov_b32 v0, s0
    // pc 5  s_endpgm
    // A skipped load is linearized when every word it writes is dead at the merge. M0 is what
    // the relative read after the merge indexes with.
    const auto program = [](uint32_t load_word) {
        return std::vector<uint32_t>{0xbf880002u, load_word,   0xfa000000u,
                                     0xbe802e04u, 0x7e000200u, kEnd};
    };
    {
        const auto shape = decode(program(0xf4201f04u));
        ASSERT_EQ(shape[1].fmt, Rdna2Format::SMEM);
        ASSERT_EQ(shape[1].opcode, 0x8u);
        ASSERT_EQ(shape[1].dst.value, 124);
        ASSERT_EQ(shape[0].pc + shape[0].len_dwords + shape[0].simm16, 3u);
        ASSERT_EQ(shape[2].opcode, kSop1OpcodeMovrelsB32);
    }
    EXPECT_FALSE(linearized(program(0xf4201f04u), 0)) << "the load writes M0";
    // Control: the same load into s3, below the relative read's base, is still linearized.
    ASSERT_EQ(decode(program(0xf42000c4u))[1].dst.value, 3);
    EXPECT_TRUE(linearized(program(0xf42000c4u), 0));
}

TEST(SMovrelsReadRange, ABranchOnARelativeReadIsNotWorkgroupUniform) {
    // pc 0  s_buffer_load_dword s16, s[8:11], null    uniform: launch SGPRs
    // pc 2  v_readfirstlane_b32 s0, v0                this wave's value
    // pc 3  s_mov_b32 m0, s0
    // pc 4  (definition of s12)
    // pc 5  s_cmp_le_u32 s12, 1
    // pc 6  s_mov_b64 exec, -1
    // pc 7  s_cbranch_scc1 +1
    // Every register from s16 up is uniform here: the load, and untouched launch values. Which
    // of them a relative read returns is chosen by M0, and M0 differs per wave, so the range
    // being uniform does not make the result uniform. The destination is deliberately BELOW the
    // base: inside the range it would be one of its own candidates, and the proof would fail
    // for that reason instead of this one.
    const auto program = [](uint32_t definition) {
        return decode({0xf4200404u, 0xfa000000u, 0x7e000500u, 0xbefc0300u, definition, 0xbf0b810cu,
                       0xbefe04c1u, 0xbf850001u, 0xbf800000u, kEnd});
    };
    const auto relative = program(0xbe8c2e10u);   // s_movrels_b32 s12, s16
    ASSERT_EQ(relative[3].opcode, kSop1OpcodeMovrelsB32);
    ASSERT_EQ(relative[3].dst.value, 12);
    ASSERT_EQ(relative[3].src[0].value, 16);
    ASSERT_EQ(relative[2].dst.value, 124) << "the move before it writes M0";
    ASSERT_EQ(relative[4].fmt, Rdna2Format::SOPC);
    ASSERT_EQ(relative[4].src[0].value, 12);
    ASSERT_EQ(relative[6].pc, 7u);
    EXPECT_FALSE(scc_branch_is_workgroup_uniform(relative, 7));
    // Control: the same branch on a plain copy of the uniform load is proven.
    const auto ordinary = program(0xbe8c0310u);   // s_mov_b32 s12, s16
    ASSERT_EQ(ordinary[3].opcode, kSop1OpcodeMovB32);
    ASSERT_EQ(ordinary[3].dst.value, 12);
    EXPECT_TRUE(scc_branch_is_workgroup_uniform(ordinary, 7));
}

TEST(SMovrelsReadRange, ALoadAddressedThroughARelativeReadIsNotWorkgroupUniform) {
    // The same question through the other slice: a sole load outside the branch's own block,
    // traced back through the straight-line entry prefix. Here the relative read produces a
    // word of the load's descriptor.
    // pc 0  v_readfirstlane_b32 s0, v0
    // pc 1  s_mov_b32 m0, s0
    // pc 2  (definition of s8)
    // pc 3  s_buffer_load_dword s16, s[8:11], null    the sole definition of s16
    // pc 5  s_cbranch_execz -> pc 7
    // pc 6  s_nop
    // pc 7  s_cmp_le_u32 s16, 1                       a branch target: the block starts here
    // pc 8  s_mov_b64 exec, -1
    // pc 9  s_cbranch_scc1 +1
    const auto program = [](uint32_t definition) {
        return decode({0x7e000500u, 0xbefc0300u, definition, 0xf4200404u, 0xfa000000u, 0xbf880001u,
                       0xbf800000u, 0xbf0b8110u, 0xbefe04c1u, 0xbf850001u, 0xbf800000u, kEnd});
    };
    const auto relative = program(0xbe882e28u);   // s_movrels_b32 s8, s40
    ASSERT_EQ(relative[2].opcode, kSop1OpcodeMovrelsB32);
    ASSERT_EQ(relative[2].dst.value, 8);
    ASSERT_EQ(relative[2].src[0].value, 40);
    ASSERT_EQ(relative[3].fmt, Rdna2Format::SMEM);
    ASSERT_EQ(relative[3].src[0].value, 8);
    ASSERT_EQ(relative[4].pc + relative[4].len_dwords + relative[4].simm16, 7u)
        << "the execz targets the compare, so the load is outside the branch's block";
    ASSERT_EQ(relative[8].pc, 9u);
    EXPECT_FALSE(scc_branch_is_workgroup_uniform(relative, 9));
    // Control: a plain copy of a launch register into the descriptor is proven.
    const auto ordinary = program(0xbe880328u);   // s_mov_b32 s8, s40
    ASSERT_EQ(ordinary[2].opcode, kSop1OpcodeMovB32);
    ASSERT_EQ(ordinary[2].dst.value, 8);
    EXPECT_TRUE(scc_branch_is_workgroup_uniform(ordinary, 9));
}
