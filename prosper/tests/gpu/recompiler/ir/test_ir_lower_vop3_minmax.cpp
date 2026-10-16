// Focus the relocated value stage: destination flow, sequential median reads and caller ownership.
#include "gpu/recompiler/ir/lower/vop3_minmax_lowering.hpp"
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
TEST(IrLowerVop3Minmax, MedianDestinationTracesThroughBothOrderedMinMaxLegs) {
    LoweringFixture t;
    const size_t begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_minmax(t.b, t.rs, instruction(0x159), t.ok, false));
    ASSERT_TRUE(t.ok);
    ASSERT_TRUE(t.rs.vreg.contains(1));
    const auto ops = results(t, begin);
    const auto d = t.rs.vreg.at(1);
    ASSERT_TRUE(ops.contains(d));
    const auto& outer = ops.at(d);
    EXPECT_EQ(outer.opcode, Op_ExtInst);
    ASSERT_EQ(outer.operands.size(), 4u);
    EXPECT_EQ(outer.operands[1], Glsl_UMax);
    const auto mn = outer.operands[2], tail = outer.operands[3];
    ASSERT_TRUE(ops.contains(mn));
    ASSERT_TRUE(ops.contains(tail));
    EXPECT_EQ(ops.at(mn).opcode, Op_ExtInst);
    EXPECT_EQ(ops.at(mn).operands,
              (std::vector<uint32_t>{t.b.glsl, Glsl_UMin, t.b.uconst_cache.at(59),
                                     t.b.uconst_cache.at(61)}));
    const auto& inner = ops.at(tail);
    EXPECT_EQ(inner.opcode, Op_ExtInst);
    ASSERT_EQ(inner.operands.size(), 4u);
    EXPECT_EQ(inner.operands[1], Glsl_UMin);
    EXPECT_EQ(inner.operands[3], t.b.uconst_cache.at(62));
    const auto mx = inner.operands[2];
    ASSERT_TRUE(ops.contains(mx));
    EXPECT_EQ(ops.at(mx).opcode, Op_ExtInst);
    EXPECT_EQ(ops.at(mx).operands,
              (std::vector<uint32_t>{t.b.glsl, Glsl_UMax, t.b.uconst_cache.at(59),
                                     t.b.uconst_cache.at(61)}));
}
TEST(IrLowerVop3Minmax, MedianReadsAllThreeSourcesSequentiallyBeforeArithmetic) {
    for (auto opcode : {0x158u, 0x159u}) {
        LoweringFixture t;
        for (auto value : {59u, 61u, 62u}) EXPECT_FALSE(t.constant_cached(value));
        const size_t begin = t.code_words();
        ASSERT_TRUE(ir::lower_vop3_minmax(t.b, t.rs, instruction(opcode), t.ok, false));
        ASSERT_TRUE(t.ok);
        ASSERT_GE(t.b.code.size() - begin, 3u);
        EXPECT_LT(t.b.uconst_cache.at(59), t.b.uconst_cache.at(61));
        EXPECT_LT(t.b.uconst_cache.at(61), t.b.uconst_cache.at(62));
        EXPECT_LT(t.b.uconst_cache.at(62), t.b.code[begin + 2]);
    }
}
TEST(IrLowerVop3Minmax, ValueStageLeavesExecVccAndPredicationToItsCaller) {
    LoweringFixture t;
    t.rs.exec = t.b.ucmp(Op_IEqual, t.b.uconst(0), t.b.uconst(1));
    t.rs.vcc = t.b.btrue();
    const auto exec = t.rs.exec, vcc = t.rs.vcc;
    const size_t begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_minmax(t.b, t.rs, instruction(0x153), t.ok, false));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.rs.exec, exec);
    EXPECT_EQ(t.rs.vcc, vcc);
    for (const auto& [id, op] : results(t, begin)) EXPECT_NE(op.opcode, Op_Select);
}
TEST(IrLowerVop3Minmax, InvalidSourceRetainsTheExistingFailureSignal) {
    for (bool invalid : {false, true}) {
        LoweringFixture t;
        auto in = instruction(0x159);
        if (invalid) in.src[2].kind = OperandKind::None;
        EXPECT_TRUE(ir::lower_vop3_minmax(t.b, t.rs, in, t.ok, false));
        EXPECT_EQ(t.ok, !invalid);
    }
}
TEST(IrLowerVop3Minmax, FloatReservedAndOtherFormatsStayWithTheCaller) {
    for (auto opcode : {0x157u, 0x128u}) {
        LoweringFixture t;
        const size_t begin = t.code_words();
        EXPECT_FALSE(ir::lower_vop3_minmax(t.b, t.rs, instruction(opcode), t.ok, false));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), begin);
    }
    LoweringFixture t;
    auto in = instruction(0x153);
    in.fmt = Rdna2Format::VOP2;
    const size_t begin = t.code_words();
    EXPECT_FALSE(ir::lower_vop3_minmax(t.b, t.rs, in, t.ok, false));
    EXPECT_TRUE(t.ok);
    EXPECT_TRUE(t.rs.vreg.empty());
    EXPECT_EQ(t.code_words(), begin);
}
