// Pin numeric data source roles, original modifier guards and caller-owned mask state.
#include "gpu/recompiler/ir/lower/vop3_bit_utilities_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
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
Rdna2Inst instruction(uint32_t opcode = 0x363) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP3;
    in.opcode = opcode;
    in.dst = {OperandKind::VGPR, 1};
    in.src[0] = {OperandKind::VGPR, 2};
    in.src[1] = {OperandKind::VGPR, 3};
    in.src[2] = {OperandKind::VGPR, 5};
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
            ADD_FAILURE() << "invalid instruction size at " << pc;
            break;
        }
        out[t.b.code[pc + 2]] = {t.b.code[pc] & 0xffffu,
                                 {t.b.code.begin() + pc + 3, t.b.code.begin() + pc + count}};
        pc += count;
    }
    return out;
}
}   // namespace
TEST(IrLowerVop3BitUtilities, HandlesTwoExistingLaneWiseFullWordOperations) {
    for (uint32_t op : {0x363u, 0x364u}) {
        const auto& e = rdna2_instruction_effects(Rdna2Format::VOP3, op);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_FALSE(e.vdst_half);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_EQ(rdna2_valu_source_count(Rdna2Format::VOP3, op), 2u);
        LoweringFixture t;
        auto in = instruction(op);
        in.src[0] = {OperandKind::InlineInt, 7};
        in.src[1] = {OperandKind::InlineInt, 9};
        const auto begin = t.code_words();
        ASSERT_TRUE(ir::lower_vop3_bit_utilities(t.b, t.rs, in, t.ok, false));
        ASSERT_TRUE(t.ok);
        const auto ops = results(t, begin);
        EXPECT_EQ(ops.at(t.rs.vreg.at(1)).opcode, op == 0x363 ? Op_ShiftLeftLogical : Op_IAdd);
    }
}
TEST(IrLowerVop3BitUtilities, WidthAndOffsetAreIndependentlyMaskedBeforeConstructingTheField) {
    LoweringFixture t;
    auto in = instruction();
    in.src[0] = {OperandKind::InlineInt, 33};
    in.src[1] = {OperandKind::InlineInt, 63};
    const auto begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_bit_utilities(t.b, t.rs, in, t.ok, false));
    ASSERT_TRUE(t.ok);
    const auto ops = results(t, begin);
    const auto& shift = ops.at(t.rs.vreg.at(1));
    ASSERT_EQ(shift.operands.size(), 2u);
    const auto& offset = ops.at(shift.operands[1]);
    EXPECT_EQ(offset.opcode, Op_BitwiseAnd);
    EXPECT_EQ(offset.operands,
              (std::vector<uint32_t>{t.b.uconst_cache.at(63), t.b.uconst_cache.at(31)}));
    const auto& subtract = ops.at(shift.operands[0]);
    EXPECT_EQ(subtract.opcode, Op_ISub);
    ASSERT_EQ(subtract.operands.size(), 2u);
    EXPECT_EQ(subtract.operands[1], t.b.uconst_cache.at(1));
    const auto& first_shift = ops.at(subtract.operands[0]);
    EXPECT_EQ(first_shift.opcode, Op_ShiftLeftLogical);
    EXPECT_EQ(first_shift.operands[0], t.b.uconst_cache.at(1));
    const auto& width = ops.at(first_shift.operands[1]);
    EXPECT_EQ(width.opcode, Op_BitwiseAnd);
    EXPECT_EQ(width.operands,
              (std::vector<uint32_t>{t.b.uconst_cache.at(33), t.b.uconst_cache.at(31)}));
    EXPECT_LT(first_shift.operands[1], shift.operands[1]);
}
TEST(IrLowerVop3BitUtilities, BitCountConsumesS0AndAddsTheWholeS1Word) {
    LoweringFixture t;
    auto in = instruction(0x364);
    in.src[0] = {OperandKind::InlineInt, -1};
    in.src[1] = {OperandKind::InlineInt, 9};
    const auto begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_bit_utilities(t.b, t.rs, in, t.ok, false));
    ASSERT_TRUE(t.ok);
    const auto ops = results(t, begin);
    const auto& add = ops.at(t.rs.vreg.at(1));
    ASSERT_EQ(add.operands.size(), 2u);
    EXPECT_EQ(add.operands[1], t.b.uconst_cache.at(9));
    const auto& count = ops.at(add.operands[0]);
    EXPECT_EQ(count.opcode, Op_BitCount);
    EXPECT_EQ(count.operands, (std::vector<uint32_t>{t.b.uconst_cache.at(0xffffffffu)}));
}
TEST(IrLowerVop3BitUtilities, BfmModifierAndRawHalfSelectRefusalsPrecedeReadsAndWrites) {
    for (uint32_t flag = 0; flag < 14; ++flag) {
        LoweringFixture t;
        auto in = instruction();
        if (flag < 3)
            in.src_abs[flag] = true;
        else if (flag < 6)
            in.src_neg[flag - 3] = true;
        else if (flag == 6)
            in.clamp = true;
        else if (flag < 10)
            in.omod = flag - 6;
        else
            in.words[0] |= 1u << (11 + flag - 10);
        const auto begin = t.code_words(), constants = t.b.uconst_cache.size();
        EXPECT_TRUE(ir::lower_vop3_bit_utilities(t.b, t.rs, in, t.ok, false));
        EXPECT_FALSE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), begin);
        EXPECT_EQ(t.b.uconst_cache.size(), constants);
    }
}
TEST(IrLowerVop3BitUtilities, BothRequiredSourcesRetainErrorsButS2IsNeverRead) {
    for (uint32_t op : {0x363u, 0x364u})
        for (int bad = -1; bad < 2; ++bad) {
            LoweringFixture t;
            auto in = instruction(op);
            in.src[0] = {OperandKind::InlineInt, 7};
            in.src[1] = {OperandKind::InlineInt, 9};
            in.src[2] = {OperandKind::None, 0};
            if (bad >= 0) in.src[bad] = {OperandKind::None, 0};
            ASSERT_TRUE(ir::lower_vop3_bit_utilities(t.b, t.rs, in, t.ok, false));
            EXPECT_EQ(t.ok, bad < 0);
            EXPECT_TRUE(t.rs.vreg.contains(1));
        }
}
TEST(IrLowerVop3BitUtilities, CallerKeepsExecVccSccAndFinalPredication) {
    for (uint32_t op : {0x363u, 0x364u}) {
        LoweringFixture t;
        auto in = instruction(op);
        in.src[0] = {OperandKind::InlineInt, 7};
        in.src[1] = {OperandKind::InlineInt, 9};
        t.rs.exec = t.b.ucmp(Op_IEqual, t.b.uconst(0), t.b.uconst(1));
        t.rs.vcc = t.b.btrue();
        t.rs.scc = t.b.bfalse();
        const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
        const auto begin = t.code_words();
        ASSERT_TRUE(ir::lower_vop3_bit_utilities(t.b, t.rs, in, t.ok, false));
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(t.rs.exec, exec);
        EXPECT_EQ(t.rs.vcc, vcc);
        EXPECT_EQ(t.rs.scc, scc);
        const auto ops = results(t, begin);
        EXPECT_NE(ops.at(t.rs.vreg.at(1)).opcode, Op_Select);
    }
}
TEST(IrLowerVop3BitUtilities, OtherOperationsAndFormatsFallBackWithoutSideEffects) {
    for (auto in : {instruction(0x365), instruction(0x366), instruction(kVop3OpcodeAddCoU32),
                    instruction(0x303)}) {
        LoweringFixture t;
        const auto begin = t.code_words();
        EXPECT_FALSE(ir::lower_vop3_bit_utilities(t.b, t.rs, in, t.ok, false));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), begin);
    }
    LoweringFixture t;
    auto in = instruction();
    in.fmt = Rdna2Format::VOP2;
    const auto begin = t.code_words();
    EXPECT_FALSE(ir::lower_vop3_bit_utilities(t.b, t.rs, in, t.ok, false));
    EXPECT_TRUE(t.ok);
    EXPECT_TRUE(t.rs.vreg.empty());
    EXPECT_EQ(t.code_words(), begin);
}
