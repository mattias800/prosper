// Pin the relocated VOP3 value stage: source-read order, register insertion and caller-owned stages.
#include "gpu/recompiler/ir/lower/vop3_ternary_integer_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t op) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP3;
    in.opcode = op;
    in.dst.kind = OperandKind::VGPR;
    in.dst.value = 1;
    for (unsigned n = 0; n < 3; ++n) {
        in.src[n].kind = OperandKind::InlineInt;
        in.src[n].value = n == 0 ? 59 : n == 1 ? 61 : 62;
    }
    return in;
}
}   // namespace
TEST(IrLowerVop3TernaryInteger, ShiftIsEmittedBeforeTheLaterSourceWordsAreRead) {
    LoweringFixture t;
    auto in = instruction(0x346);
    for (uint32_t value : {59u, 61u, 62u}) EXPECT_FALSE(t.constant_cached(value));
    EXPECT_TRUE(t.rs.vreg.empty());
    const size_t before = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_ternary_integer(t.b, t.rs, in, t.ok, false));
    ASSERT_TRUE(t.ok);
    ASSERT_TRUE(t.rs.vreg.contains(1));
    uint32_t and_result = 0;
    for (size_t pc = before; pc < t.b.code.size();) {
        const size_t count = t.b.code[pc] >> 16;
        ASSERT_NE(count, 0u);
        ASSERT_LE(count, t.b.code.size() - pc);
        if ((t.b.code[pc] & 0xffffu) == Op_BitwiseAnd) and_result = t.b.code[pc + 2];
        pc += count;
    }
    ASSERT_NE(and_result, 0u);
    EXPECT_LT(t.b.uconst_cache.at(61), and_result);
    EXPECT_LT(and_result, t.b.uconst_cache.at(62));
    EXPECT_LT(and_result, t.b.uconst_cache.at(59));
}
TEST(IrLowerVop3TernaryInteger, ValueStageLeavesExecVccAndPredicationWithTheCaller) {
    LoweringFixture t;
    auto in = instruction(kVop3OpcodeAdd3U32);
    t.rs.exec = t.b.ucmp(Op_IEqual, t.b.uconst(0), t.b.uconst(1));
    t.rs.vcc = t.b.btrue();
    const auto old_exec = t.rs.exec, old_vcc = t.rs.vcc;
    const size_t before = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_ternary_integer(t.b, t.rs, in, t.ok, false));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.rs.exec, old_exec);
    EXPECT_EQ(t.rs.vcc, old_vcc);
    for (size_t pc = before; pc < t.b.code.size();) {
        const size_t count = t.b.code[pc] >> 16;
        ASSERT_NE(count, 0u);
        ASSERT_LE(count, t.b.code.size() - pc);
        EXPECT_NE(t.b.code[pc] & 0xffffu, Op_Select);
        pc += count;
    }
}
TEST(IrLowerVop3TernaryInteger, UnrecognizedSourceKeepsTheExistingFailureSignal) {
    for (bool invalid : {false, true}) {
        LoweringFixture t;
        auto in = instruction(0x346);
        if (invalid) in.src[1].kind = OperandKind::None;
        EXPECT_TRUE(ir::lower_vop3_ternary_integer(t.b, t.rs, in, t.ok, false));
        EXPECT_EQ(t.ok, !invalid);
    }
}
TEST(IrLowerVop3TernaryInteger, OtherFormatsAndReservedOpcodesStayWithTheCaller) {
    for (auto fmt : {Rdna2Format::VOP2, Rdna2Format::VOP3}) {
        LoweringFixture t;
        auto in = instruction(fmt == Rdna2Format::VOP3 ? 0x128 : 0x346);
        in.fmt = fmt;
        const size_t before = t.code_words();
        EXPECT_FALSE(ir::lower_vop3_ternary_integer(t.b, t.rs, in, t.ok, false));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), before);
    }
}
