// Selected halves feed integer arithmetic; the untouched destination half and mask state survive.
#include "gpu/recompiler/ir/lower/vop3_integer16_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t opcode = 0x303, uint32_t selectors = 0) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP3;
    in.opcode = opcode;
    in.dst = {OperandKind::VGPR, 1};
    in.src[0] = {OperandKind::VGPR, 2};
    in.src[1] = {OperandKind::VGPR, 3};
    in.src[2] = {OperandKind::VGPR, 5};
    in.vop3p_opsel = static_cast<uint8_t>(selectors);
    return in;
}
struct Result {
    uint32_t opcode;
    std::vector<uint32_t> operands;
};
std::map<uint32_t, Result> results(const LoweringFixture& t, size_t begin) {
    std::map<uint32_t, Result> out;
    for (size_t pc = begin; pc < t.b.code.size();) {
        const size_t count = t.b.code[pc] >> 16;
        if (count < 3 || count > t.b.code.size() - pc) {
            ADD_FAILURE() << "bad instruction at " << pc;
            break;
        }
        out[t.b.code[pc + 2]] = {t.b.code[pc] & 0xffffu,
                                 {t.b.code.begin() + pc + 3, t.b.code.begin() + pc + count}};
        pc += count;
    }
    return out;
}
void sources(LoweringFixture& t) {
    t.rs.vreg[2] = t.b.uconst(0xabcd0007);
    t.rs.vreg[3] = t.b.uconst(0xef120009);
    t.rs.vreg[5] = t.b.uconst(0x3412000b);
}
}   // namespace
TEST(IrLowerVop3Integer16, HandlesExactlyTwentyLeavesWithAuthoritativeHalfAndSourceFacts) {
    size_t handled = 0;
    for (uint32_t opcode = 0; opcode < 1024; ++opcode) {
        LoweringFixture t;
        sources(t);
        const auto old = t.b.uconst(0xdeadbeef);
        const auto begin = t.code_words();
        if (!ir::lower_vop3_integer16(t.b, t.rs, instruction(opcode), t.ok, false, old)) {
            EXPECT_TRUE(t.ok);
            EXPECT_FALSE(t.rs.vreg.contains(1));
            EXPECT_EQ(t.code_words(), begin);
            continue;
        }
        ++handled;
        SCOPED_TRACE(opcode);
        ASSERT_TRUE(t.ok);
        const auto& e = rdna2_instruction_effects(Rdna2Format::VOP3, opcode);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_TRUE(e.vdst_half);
        EXPECT_EQ(e.vdst_dwords, 1u);
        const auto count = rdna2_valu_source_count(Rdna2Format::VOP3, opcode);
        EXPECT_GE(count, 2u);
        EXPECT_LE(count, 3u);
        EXPECT_TRUE(t.rs.vreg.contains(1));
    }
    EXPECT_EQ(handled, 20u);
}
TEST(IrLowerVop3Integer16, AddSelectsEachSourceHalfAndPreservesTheOtherDestinationHalf) {
    for (uint32_t selection = 0; selection < 8; ++selection) {
        const uint32_t opsel = (selection & 3) | ((selection >> 2) << 3);
        LoweringFixture t;
        sources(t);
        const auto old = t.b.uconst(0xdeadbeef);
        const auto begin = t.code_words();
        ASSERT_TRUE(
            ir::lower_vop3_integer16(t.b, t.rs, instruction(0x303, opsel), t.ok, false, old));
        ASSERT_TRUE(t.ok);
        const auto ops = results(t, begin);
        const auto& d = ops.at(t.rs.vreg.at(1));
        EXPECT_EQ(d.opcode, Op_BitwiseOr);
        ASSERT_EQ(d.operands.size(), 2u);
        const auto& keep = ops.at(d.operands[0]);
        EXPECT_EQ(keep.opcode, Op_BitwiseAnd);
        EXPECT_EQ(
            keep.operands,
            (std::vector<uint32_t>{old, t.b.uconst_cache.at((opsel & 8) ? 0xffffu : 0xffff0000u)}));
        auto r = d.operands[1];
        if (opsel & 8) {
            const auto& shifted = ops.at(r);
            EXPECT_EQ(shifted.opcode, Op_ShiftLeftLogical);
            ASSERT_EQ(shifted.operands.size(), 2u);
            EXPECT_EQ(shifted.operands[1], t.b.uconst_cache.at(16));
            r = shifted.operands[0];
        }
        const auto& mask = ops.at(r);
        EXPECT_EQ(mask.opcode, Op_BitwiseAnd);
        ASSERT_EQ(mask.operands.size(), 2u);
        EXPECT_EQ(mask.operands[1], t.b.uconst_cache.at(0xffff));
        const auto& sum = ops.at(mask.operands[0]);
        EXPECT_EQ(sum.opcode, Op_IAdd);
        ASSERT_EQ(sum.operands.size(), 2u);
        for (uint32_t source = 0; source < 2; ++source) {
            const auto& half = ops.at(sum.operands[source]);
            EXPECT_EQ(half.opcode, Op_BitwiseAnd);
            ASSERT_EQ(half.operands.size(), 2u);
            EXPECT_EQ(half.operands[1], t.b.uconst_cache.at(0xffff));
            auto raw = half.operands[0];
            if ((opsel >> source) & 1) {
                const auto& shift = ops.at(raw);
                EXPECT_EQ(shift.opcode, Op_ShiftRightLogical);
                ASSERT_EQ(shift.operands.size(), 2u);
                EXPECT_EQ(shift.operands[1], t.b.uconst_cache.at(16));
                raw = shift.operands[0];
            }
            EXPECT_EQ(raw, t.rs.vreg.at(int(2 + source)));
        }
    }
}
TEST(IrLowerVop3Integer16, InlineSourcesPrecedeArithmeticAndFinalMaskStaysLate) {
    LoweringFixture t;
    auto in = instruction();
    in.src[0] = {OperandKind::InlineInt, 7};
    in.src[1] = {OperandKind::InlineInt, 9};
    in.src[2] = {OperandKind::None, 0};
    const auto old = t.b.uconst(0xdeadbeef);
    const bool had_mask = t.constant_cached(0xffff);
    const auto begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_integer16(t.b, t.rs, in, t.ok, false, old));
    ASSERT_TRUE(t.ok);
    const auto ops = results(t, begin);
    const auto& d = ops.at(t.rs.vreg.at(1));
    const auto& mask = ops.at(d.operands.at(1));
    EXPECT_EQ(mask.opcode, Op_BitwiseAnd);
    const auto sum = mask.operands.at(0);
    EXPECT_EQ(ops.at(sum).opcode, Op_IAdd);
    EXPECT_EQ(ops.at(sum).operands,
              (std::vector<uint32_t>{t.b.uconst_cache.at(7), t.b.uconst_cache.at(9)}));
    EXPECT_LT(t.b.uconst_cache.at(7), t.b.uconst_cache.at(9));
    if (!had_mask) EXPECT_LT(sum, t.b.uconst_cache.at(0xffff));
}
TEST(IrLowerVop3Integer16, InlineBusKeepsHalfFloatBitsAndIntegerSignExtension) {
    uint32_t word = 0;
    EXPECT_TRUE(inline_16bit_operand_dword({OperandKind::InlineFloat, 242}, word));
    EXPECT_EQ(word, 0x00003c00u);   //1.0h: bias15 and zero fraction, not repeated in the high half.
    EXPECT_TRUE(inline_16bit_operand_dword({OperandKind::InlineInt, 7}, word));
    EXPECT_EQ(word, 7u);
    EXPECT_TRUE(inline_16bit_operand_dword({OperandKind::InlineInt, -1}, word));
    EXPECT_EQ(word, 0xffffffffu);
    EXPECT_TRUE(inline_16bit_operand_dword({OperandKind::InlineInt, -2}, word));
    EXPECT_EQ(word, 0xfffffffeu);
    EXPECT_FALSE(inline_16bit_operand_dword({OperandKind::VGPR, 2}, word));
    EXPECT_EQ(word, 0xfffffffeu);
}
TEST(IrLowerVop3Integer16, UnusedThirdSourceIsIgnoredButRequiredSourceErrorsSurvive) {
    for (auto opcode : {0x303u, 0x340u}) {
        const auto count = rdna2_valu_source_count(Rdna2Format::VOP3, opcode);
        for (uint32_t bad = 0; bad < 3; ++bad) {
            LoweringFixture t;
            sources(t);
            auto in = instruction(opcode);
            in.src[bad] = {OperandKind::None, 0};
            const auto old = t.b.uconst(0xdeadbeef);
            EXPECT_TRUE(ir::lower_vop3_integer16(t.b, t.rs, in, t.ok, false, old));
            EXPECT_EQ(t.ok, bad >= count);
            EXPECT_TRUE(t.rs.vreg.contains(1));
        }
    }
}
TEST(IrLowerVop3Integer16, EveryAbsNegModifierRefusesBeforeReadingOrWriting) {
    for (uint32_t flag = 0; flag < 6; ++flag) {
        LoweringFixture t;
        sources(t);
        auto in = instruction(0x340);
        if (flag < 3)
            in.src_abs[flag] = true;
        else
            in.src_neg[flag - 3] = true;
        const auto old = t.b.uconst(0xdeadbeef);
        const auto begin = t.code_words();
        EXPECT_TRUE(ir::lower_vop3_integer16(t.b, t.rs, in, t.ok, false, old));
        EXPECT_FALSE(t.ok);
        EXPECT_FALSE(t.rs.vreg.contains(1));
        EXPECT_EQ(t.code_words(), begin);
    }
}
TEST(IrLowerVop3Integer16, ExecVccSccAndFinalPredicationRemainWithTheCaller) {
    LoweringFixture t;
    sources(t);
    const auto old = t.b.uconst(0xdeadbeef);
    t.rs.exec = t.b.ucmp(Op_IEqual, t.b.uconst(0), t.b.uconst(1));
    t.rs.vcc = t.b.btrue();
    t.rs.scc = t.b.btrue();
    const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
    const auto begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_integer16(t.b, t.rs, instruction(0x340, 8), t.ok, false, old));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.rs.exec, exec);
    EXPECT_EQ(t.rs.vcc, vcc);
    EXPECT_EQ(t.rs.scc, scc);
    const auto ops = results(t, begin);
    EXPECT_EQ(ops.at(t.rs.vreg.at(1)).opcode, Op_BitwiseOr);
    for (const auto& [id, op] : ops) EXPECT_NE(op.opcode, Op_Select);
}
TEST(IrLowerVop3Integer16, HolesFloatCarryLaneAndOtherFormatsFallBackUntouched) {
    for (auto in :
         {instruction(0x306), instruction(0x34b), instruction(0x30f), instruction(0x360)}) {
        LoweringFixture t;
        const auto old = t.b.uconst(0xdeadbeef);
        const auto begin = t.code_words();
        EXPECT_FALSE(ir::lower_vop3_integer16(t.b, t.rs, in, t.ok, false, old));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), begin);
    }
    LoweringFixture t;
    auto in = instruction();
    in.fmt = Rdna2Format::VOP3P;
    const auto old = t.b.uconst(0xdeadbeef);
    const auto begin = t.code_words();
    EXPECT_FALSE(ir::lower_vop3_integer16(t.b, t.rs, in, t.ok, false, old));
    EXPECT_TRUE(t.ok);
    EXPECT_TRUE(t.rs.vreg.empty());
    EXPECT_EQ(t.code_words(), begin);
}
