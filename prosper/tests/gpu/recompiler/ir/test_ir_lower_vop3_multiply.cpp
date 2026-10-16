// Follow D through low/extended product emission; preserve the caller-owned stages and failures.
#include "gpu/recompiler/ir/lower/vop3_multiply_lowering.hpp"
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
    in.src[0] = {OperandKind::VGPR, 2};
    in.src[1] = {OperandKind::VGPR, 3};
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
} // namespace
TEST(IrLowerVop3Multiply, DestinationComesFromTheRequestedProductWord) {
    for (auto opcode : {kVop3OpcodeMulLoU32, kVop3OpcodeMulHiU32, kVop3OpcodeMulHiI32}) {
        SCOPED_TRACE(opcode);
        LoweringFixture t;
        const auto a = t.b.uconst(0x80000001u), b = t.b.uconst(0xfffffffeu);
        t.rs.vreg[2] = a;
        t.rs.vreg[3] = b;
        const size_t begin = t.code_words();
        ASSERT_TRUE(ir::lower_vop3_multiply(t.b, t.rs, instruction(opcode), t.ok, false));
        ASSERT_TRUE(t.ok);
        ASSERT_TRUE(t.rs.vreg.contains(1));
        const auto ops = results(t, begin);
        auto product = t.rs.vreg.at(1);
        if (opcode == kVop3OpcodeMulHiI32) {
            ASSERT_TRUE(ops.contains(product));
            EXPECT_EQ(ops.at(product).opcode, Op_Bitcast);
            ASSERT_EQ(ops.at(product).operands.size(), 1u);
            product = ops.at(product).operands[0];
        }
        if (opcode != kVop3OpcodeMulLoU32) {
            ASSERT_TRUE(ops.contains(product));
            EXPECT_EQ(ops.at(product).opcode, Op_CompositeExtract);
            ASSERT_EQ(ops.at(product).operands.size(), 2u);
            EXPECT_EQ(ops.at(product).operands[1], 1u);
            product = ops.at(product).operands[0];
        }
        ASSERT_TRUE(ops.contains(product));
        const auto& mul = ops.at(product);
        EXPECT_EQ(mul.opcode, opcode == kVop3OpcodeMulLoU32   ? Op_IMul
                              : opcode == kVop3OpcodeMulHiU32 ? Op_UMulExtended
                                                              : Op_SMulExtended);
        ASSERT_EQ(mul.operands.size(), 2u);
        for (size_t source = 0; source < 2; ++source) {
            auto id = mul.operands[source];
            if (opcode == kVop3OpcodeMulHiI32) {
                ASSERT_TRUE(ops.contains(id));
                EXPECT_EQ(ops.at(id).opcode, Op_Bitcast);
                ASSERT_EQ(ops.at(id).operands.size(), 1u);
                id = ops.at(id).operands[0];
            }
            EXPECT_EQ(id, source == 0 ? a : b);
        }
        EXPECT_EQ(t.rs.vreg.size(), 3u);
    }
}
TEST(IrLowerVop3Multiply, ValueStageLeavesExecVccAndPredicationToItsCaller) {
    LoweringFixture t;
    t.rs.vreg[2] = t.b.uconst(3);
    t.rs.vreg[3] = t.b.uconst(5);
    t.rs.exec = t.b.ucmp(Op_IEqual, t.b.uconst(0), t.b.uconst(1));
    t.rs.vcc = t.b.btrue();
    const auto exec = t.rs.exec, vcc = t.rs.vcc;
    const size_t begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_multiply(t.b, t.rs, instruction(kVop3OpcodeMulLoU32), t.ok, false));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.rs.exec, exec);
    EXPECT_EQ(t.rs.vcc, vcc);
    for (const auto& [id, op] : results(t, begin)) EXPECT_NE(op.opcode, Op_Select);
}
TEST(IrLowerVop3Multiply, InvalidSourceRetainsTheExistingFailureSignal) {
    for (bool invalid : {false, true}) {
        LoweringFixture t;
        auto in = instruction(kVop3OpcodeMulLoU32);
        in.src[0] = {invalid ? OperandKind::None : OperandKind::InlineInt, 7};
        in.src[1] = {OperandKind::InlineInt, 9};
        EXPECT_TRUE(ir::lower_vop3_multiply(t.b, t.rs, in, t.ok, false));
        EXPECT_EQ(t.ok, !invalid);
    }
}
TEST(IrLowerVop3Multiply, OtherFormatsUnsupportedAndReservedOperationsFallBackUntouched) {
    for (auto in : {instruction(0x16b), instruction(kVop3OpcodeAddCoU32), instruction(0x02)}) {
        LoweringFixture t;
        const size_t begin = t.code_words();
        EXPECT_FALSE(ir::lower_vop3_multiply(t.b, t.rs, in, t.ok, false));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), begin);
    }
    LoweringFixture t;
    auto in = instruction(kVop3OpcodeMulLoU32);
    in.fmt = Rdna2Format::VOP2;
    const size_t begin = t.code_words();
    EXPECT_FALSE(ir::lower_vop3_multiply(t.b, t.rs, in, t.ok, false));
    EXPECT_TRUE(t.rs.vreg.empty());
    EXPECT_EQ(t.code_words(), begin);
}
