// Pin join direction, zero bypass, byte domains, ordered reads and existing modifier refusals.
#include "gpu/recompiler/ir/lower/vop3_align_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t opcode = kVop3OpcodeAlignbitB32) {
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
TEST(IrLowerVop3Align, BitResultJoinsS0HighAndS1LowWithZeroBypass) {
    LoweringFixture t;
    const auto a = t.b.uconst(0x11223344), b = t.b.uconst(0x55667788), c = t.b.uconst(13);
    t.rs.vreg[2] = a;
    t.rs.vreg[3] = b;
    t.rs.vreg[5] = c;
    const auto begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_align(t.b, t.rs, instruction(), t.ok, false));
    ASSERT_TRUE(t.ok);
    const auto ops = results(t, begin);
    const auto& d = ops.at(t.rs.vreg.at(1));
    EXPECT_EQ(d.opcode, Op_Select);
    ASSERT_EQ(d.operands.size(), 3u);
    EXPECT_EQ(d.operands[1], b);
    const auto& zero = ops.at(d.operands[0]);
    EXPECT_EQ(zero.opcode, Op_IEqual);
    ASSERT_EQ(zero.operands.size(), 2u);
    EXPECT_EQ(zero.operands[1], t.b.uconst_cache.at(0));
    const auto& count = ops.at(zero.operands[0]);
    EXPECT_EQ(count.opcode, Op_BitwiseAnd);
    EXPECT_EQ(count.operands, (std::vector<uint32_t>{c, t.b.uconst_cache.at(31)}));
    const auto& join = ops.at(d.operands[2]);
    EXPECT_EQ(join.opcode, Op_BitwiseOr);
    ASSERT_EQ(join.operands.size(), 2u);
    const auto& right = ops.at(join.operands[0]);
    const auto& left = ops.at(join.operands[1]);
    EXPECT_EQ(right.opcode, Op_ShiftRightLogical);
    EXPECT_EQ(left.opcode, Op_ShiftLeftLogical);
    EXPECT_EQ(right.operands, (std::vector<uint32_t>{b, zero.operands[0]}));
    ASSERT_EQ(left.operands.size(), 2u);
    EXPECT_EQ(left.operands[0], a);
    const auto& inverse = ops.at(left.operands[1]);
    EXPECT_EQ(inverse.opcode, Op_BitwiseAnd);
    ASSERT_EQ(inverse.operands.size(), 2u);
    EXPECT_EQ(inverse.operands[1], t.b.uconst_cache.at(31));
    const auto& subtraction = ops.at(inverse.operands[0]);
    EXPECT_EQ(subtraction.opcode, Op_ISub);
    EXPECT_EQ(subtraction.operands,
              (std::vector<uint32_t>{t.b.uconst_cache.at(32), zero.operands[0]}));
}
TEST(IrLowerVop3Align, ByteResultRetainsLowUpperAndZeroDomains) {
    LoweringFixture t;
    const auto a = t.b.uconst(0x11223344), b = t.b.uconst(0x55667788), c = t.b.uconst(13);
    t.rs.vreg[2] = a;
    t.rs.vreg[3] = b;
    t.rs.vreg[5] = c;
    const auto begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_align(t.b, t.rs, instruction(kVop3OpcodeAlignbyteB32), t.ok, false));
    ASSERT_TRUE(t.ok);
    const auto ops = results(t, begin);
    const auto& d = ops.at(t.rs.vreg.at(1));
    EXPECT_EQ(d.opcode, Op_Select);
    ASSERT_EQ(d.operands.size(), 3u);
    const auto& low_test = ops.at(d.operands[0]);
    EXPECT_EQ(low_test.opcode, Op_ULessThan);
    ASSERT_EQ(low_test.operands.size(), 2u);
    EXPECT_EQ(low_test.operands[1], t.b.uconst_cache.at(4));
    const auto& upper_or_zero = ops.at(d.operands[2]);
    EXPECT_EQ(upper_or_zero.opcode, Op_Select);
    ASSERT_EQ(upper_or_zero.operands.size(), 3u);
    EXPECT_EQ(upper_or_zero.operands[2], t.b.uconst_cache.at(0));
    const auto& upper_test = ops.at(upper_or_zero.operands[0]);
    EXPECT_EQ(upper_test.opcode, Op_ULessThan);
    EXPECT_EQ(upper_test.operands,
              (std::vector<uint32_t>{low_test.operands[0], t.b.uconst_cache.at(8)}));
    const auto& upper = ops.at(upper_or_zero.operands[1]);
    EXPECT_EQ(upper.opcode, Op_ShiftRightLogical);
    ASSERT_EQ(upper.operands.size(), 2u);
    EXPECT_EQ(upper.operands[0], a);
    const auto& safe = ops.at(upper.operands[1]);
    EXPECT_EQ(safe.opcode, Op_BitwiseAnd);
    ASSERT_EQ(safe.operands.size(), 2u);
    EXPECT_EQ(safe.operands[1], t.b.uconst_cache.at(31));
    const auto& low = ops.at(d.operands[1]);
    EXPECT_EQ(low.opcode, Op_Select);
    ASSERT_EQ(low.operands.size(), 3u);
    EXPECT_EQ(low.operands[1], b);
}
TEST(IrLowerVop3Align, EveryModifierRefusesBeforeReadingSourcesOrInsertingTheDestination) {
    for (auto op : {kVop3OpcodeAlignbitB32, kVop3OpcodeAlignbyteB32})
        for (unsigned flag = 0; flag < 8; ++flag) {
            LoweringFixture t;
            auto in = instruction(op);
            if (flag < 3)
                in.src_abs[flag] = true;
            else if (flag < 6)
                in.src_neg[flag - 3] = true;
            else if (flag == 6)
                in.clamp = true;
            else
                in.omod = 1;
            const auto begin = t.code_words();
            const auto constants = t.b.uconst_cache.size();
            EXPECT_TRUE(ir::lower_vop3_align(t.b, t.rs, in, t.ok, false));
            EXPECT_FALSE(t.ok);
            EXPECT_TRUE(t.rs.vreg.empty());
            EXPECT_EQ(t.code_words(), begin);
            EXPECT_EQ(t.b.uconst_cache.size(), constants);
        }
}
TEST(IrLowerVop3Align, SourceReadsStaySequentialBeforeCountArithmetic) {
    for (auto op : {kVop3OpcodeAlignbitB32, kVop3OpcodeAlignbyteB32}) {
        LoweringFixture t;
        auto in = instruction(op);
        for (size_t i = 0; i < 3; ++i) in.src[i] = {OperandKind::InlineInt, int(7 + i * 2)};
        ASSERT_TRUE(ir::lower_vop3_align(t.b, t.rs, in, t.ok, false));
        ASSERT_TRUE(t.ok);
        EXPECT_LT(t.b.uconst_cache.at(7), t.b.uconst_cache.at(9));
        EXPECT_LT(t.b.uconst_cache.at(9), t.b.uconst_cache.at(11));
        EXPECT_LT(t.b.uconst_cache.at(11), t.rs.vreg.at(1));
    }
}
TEST(IrLowerVop3Align, BadSourcesRetainTheOriginalFailureAndValueStage) {
    for (auto op : {kVop3OpcodeAlignbitB32, kVop3OpcodeAlignbyteB32})
        for (size_t bad = 0; bad < 3; ++bad) {
            LoweringFixture t;
            auto in = instruction(op);
            for (size_t i = 0; i < 3; ++i) in.src[i] = {OperandKind::InlineInt, int(7 + i * 2)};
            in.src[bad] = {OperandKind::None, 0};
            const auto begin = t.code_words();
            const auto constants = t.b.uconst_cache.size();
            EXPECT_TRUE(ir::lower_vop3_align(t.b, t.rs, in, t.ok, false));
            EXPECT_FALSE(t.ok);
            ASSERT_TRUE(t.rs.vreg.contains(1));
            const auto ops = results(t, begin);
            EXPECT_EQ(ops.at(t.rs.vreg.at(1)).opcode, Op_Select);
        }
}
TEST(IrLowerVop3Align, CallerOwnsExecVccAndDestinationPredication) {
    for (auto op : {kVop3OpcodeAlignbitB32, kVop3OpcodeAlignbyteB32}) {
        LoweringFixture t;
        auto in = instruction(op);
        for (size_t i = 0; i < 3; ++i) in.src[i] = {OperandKind::InlineInt, int(7 + i * 2)};
        t.rs.exec = t.b.ucmp(Op_IEqual, t.b.uconst(0), t.b.uconst(1));
        t.rs.vcc = t.b.btrue();
        const auto exec = t.rs.exec, vcc = t.rs.vcc;
        const auto begin = t.code_words();
        ASSERT_TRUE(ir::lower_vop3_align(t.b, t.rs, in, t.ok, false));
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(t.rs.exec, exec);
        EXPECT_EQ(t.rs.vcc, vcc);
        const auto ops = results(t, begin);
        EXPECT_NE(ops.at(t.rs.vreg.at(1)).operands.at(0), exec);
    }
}
TEST(IrLowerVop3Align, OtherOperationsAndFormatsFallBackUntouched) {
    for (auto in : {instruction(0x142), instruction(kVop3OpcodeAddCoU32), instruction(0x02)}) {
        LoweringFixture t;
        const auto begin = t.code_words();
        EXPECT_FALSE(ir::lower_vop3_align(t.b, t.rs, in, t.ok, false));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), begin);
    }
    LoweringFixture t;
    auto in = instruction();
    in.fmt = Rdna2Format::VOP2;
    const auto begin = t.code_words();
    EXPECT_FALSE(ir::lower_vop3_align(t.b, t.rs, in, t.ok, false));
    EXPECT_TRUE(t.ok);
    EXPECT_TRUE(t.rs.vreg.empty());
    EXPECT_EQ(t.code_words(), begin);
}
