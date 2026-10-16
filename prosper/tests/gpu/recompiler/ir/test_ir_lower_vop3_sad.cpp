// Pin the two existing operand-read stages and trace actual SAD destination results.
#include "gpu/recompiler/ir/lower/vop3_sad_lowering.hpp"
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
Rdna2Inst instruction(uint32_t opcode) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP3;
    in.opcode = opcode;
    in.dst = {OperandKind::VGPR, 1};
    for (int source = 0; source < 3; ++source)
        in.src[source] = {OperandKind::InlineInt, source == 0 ? 59 : source == 1 ? 61 : 62};
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
            ADD_FAILURE() << "invalid size at " << pc;
            break;
        }
        out[t.b.code[pc + 2]] = {t.b.code[pc] & 0xffffu,
                                 {t.b.code.begin() + pc + 3, t.b.code.begin() + pc + count}};
        pc += count;
    }
    return out;
}
}   // namespace
TEST(IrLowerVop3Sad, SubwordReadsAreSequentialAndPrecedeArithmetic) {
    for (auto opcode : {0x15au, 0x15bu, 0x15cu, 0x171u}) {
        SCOPED_TRACE(opcode);
        LoweringFixture t;
        const size_t begin = t.code_words();
        ASSERT_TRUE(ir::lower_vop3_sad(t.b, t.rs, instruction(opcode), t.ok, false));
        ASSERT_TRUE(t.ok);
        ASSERT_GT(t.b.code.size(), begin + 2);
        EXPECT_LT(t.b.uconst_cache.at(59), t.b.uconst_cache.at(61));
        EXPECT_LT(t.b.uconst_cache.at(61), t.b.uconst_cache.at(62));
        EXPECT_LT(t.b.uconst_cache.at(62), t.b.code[begin + 2]);
        const auto ops = results(t, begin);
        const auto d = t.rs.vreg.at(1);
        ASSERT_TRUE(ops.contains(d));
        EXPECT_EQ(ops.at(d).opcode, Op_IAdd);
        ASSERT_EQ(ops.at(d).operands.size(), 2u);
        EXPECT_EQ(ops.at(d).operands[1], t.b.uconst_cache.at(62));
        if (opcode == 0x15b) {
            const auto shifted = ops.at(d).operands[0];
            ASSERT_TRUE(ops.contains(shifted));
            EXPECT_EQ(ops.at(shifted).opcode, Op_ShiftLeftLogical);
            ASSERT_EQ(ops.at(shifted).operands.size(), 2u);
            EXPECT_EQ(ops.at(shifted).operands[1], t.b.uconst_cache.at(16));
        }
    }
}
TEST(IrLowerVop3Sad, SubwordErrorStillReadsAllSourcesThenReturnsZeroWithoutMath) {
    for (auto opcode : {0x15au, 0x15bu, 0x15cu, 0x171u})
        for (int invalid = -1; invalid < 3; ++invalid) {
            LoweringFixture t;
            auto in = instruction(opcode);
            if (invalid >= 0) in.src[invalid].kind = OperandKind::None;
            const size_t begin = t.code_words();
            ASSERT_TRUE(ir::lower_vop3_sad(t.b, t.rs, in, t.ok, false));
            EXPECT_EQ(t.ok, invalid < 0);
            for (int source = 0; source < 3; ++source)
                if (source != invalid)
                    EXPECT_TRUE(t.constant_cached(source == 0 ? 59 : source == 1 ? 61 : 62));
            if (invalid >= 0) {
                EXPECT_EQ(t.rs.vreg.at(1), t.b.uconst_cache.at(0));
                EXPECT_EQ(t.code_words(), begin);
            } else
                EXPECT_GT(t.code_words(), begin);
        }
}
TEST(IrLowerVop3Sad, DwordDestinationUsesUnsignedMagnitudeBeforeTheAccumulatorRead) {
    LoweringFixture t;
    const size_t begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_sad(t.b, t.rs, instruction(0x15d), t.ok, false));
    ASSERT_TRUE(t.ok);
    const auto ops = results(t, begin);
    const auto d = t.rs.vreg.at(1);
    ASSERT_TRUE(ops.contains(d));
    EXPECT_EQ(ops.at(d).opcode, Op_IAdd);
    ASSERT_EQ(ops.at(d).operands.size(), 2u);
    EXPECT_EQ(ops.at(d).operands[1], t.b.uconst_cache.at(62));
    const auto diff = ops.at(d).operands[0];
    ASSERT_TRUE(ops.contains(diff));
    EXPECT_EQ(ops.at(diff).opcode, Op_ISub);
    ASSERT_EQ(ops.at(diff).operands.size(), 2u);
    EXPECT_LT(diff, t.b.uconst_cache.at(62));
    for (size_t side = 0; side < 2; ++side) {
        const auto leg = ops.at(diff).operands[side];
        ASSERT_TRUE(ops.contains(leg));
        EXPECT_EQ(ops.at(leg).opcode, Op_ExtInst);
        EXPECT_EQ(ops.at(leg).operands,
                  (std::vector<uint32_t>{t.b.glsl, side == 0 ? Glsl_UMax : Glsl_UMin,
                                         t.b.uconst_cache.at(59), t.b.uconst_cache.at(61)}));
    }
}
TEST(IrLowerVop3Sad, DwordErrorRetainsItsExistingMathAndLateRead) {
    LoweringFixture t;
    auto in = instruction(0x15d);
    in.src[0].kind = OperandKind::None;
    const size_t begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_sad(t.b, t.rs, in, t.ok, false));
    EXPECT_FALSE(t.ok);
    EXPECT_TRUE(t.constant_cached(61));
    EXPECT_TRUE(t.constant_cached(62));
    EXPECT_GT(t.code_words(), begin);
    const auto ops = results(t, begin);
    EXPECT_EQ(ops.at(t.rs.vreg.at(1)).opcode, Op_IAdd);
}
TEST(IrLowerVop3Sad, ValueStageLeavesExecVccAndPredicationToItsCaller) {
    for (auto opcode : {0x15au, 0x15bu, 0x15cu, 0x15du, 0x171u}) {
        LoweringFixture t;
        t.rs.exec = t.b.ucmp(Op_IEqual, t.b.uconst(0), t.b.uconst(1));
        t.rs.exec_narrowed = true;
        t.rs.vcc = t.b.btrue();
        const auto exec = t.rs.exec, vcc = t.rs.vcc;
        const size_t begin = t.code_words();
        ASSERT_TRUE(ir::lower_vop3_sad(t.b, t.rs, instruction(opcode), t.ok, false));
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(t.rs.exec, exec);
        EXPECT_EQ(t.rs.vcc, vcc);
        const auto ops = results(t, begin);
        EXPECT_EQ(ops.at(t.rs.vreg.at(1)).opcode, Op_IAdd);
    }
}
TEST(IrLowerVop3Sad, ReservedForeignAndOtherFormatsStayWithTheCaller) {
    for (auto opcode : {0x176u, 0x172u}) {
        LoweringFixture t;
        const size_t begin = t.code_words();
        EXPECT_FALSE(ir::lower_vop3_sad(t.b, t.rs, instruction(opcode), t.ok, false));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), begin);
        EXPECT_FALSE(t.constant_cached(59));
    }
    LoweringFixture t;
    auto in = instruction(0x15a);
    in.fmt = Rdna2Format::VOP2;
    const size_t begin = t.code_words();
    EXPECT_FALSE(ir::lower_vop3_sad(t.b, t.rs, in, t.ok, false));
    EXPECT_TRUE(t.ok);
    EXPECT_TRUE(t.rs.vreg.empty());
    EXPECT_EQ(t.code_words(), begin);
}
