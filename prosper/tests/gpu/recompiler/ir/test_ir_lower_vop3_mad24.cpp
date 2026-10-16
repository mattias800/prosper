// Pin the original masked-product graph and late, full-width accumulator read.
#include "gpu/recompiler/ir/lower/vop3_mad24_lowering.hpp"
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
Rdna2Inst instruction(uint32_t opcode = 0x143) {
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
TEST(IrLowerVop3Mad24, DestinationIsMaskedProductPlusFullAccumulator) {
    LoweringFixture t;
    const auto a = t.b.uconst(0xff000002), b = t.b.uconst(0xaa000003), c = t.b.uconst(0xff000005);
    t.rs.vreg[2] = a;
    t.rs.vreg[3] = b;
    t.rs.vreg[5] = c;
    const size_t begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_mad24(t.b, t.rs, instruction(), t.ok, false));
    ASSERT_TRUE(t.ok);
    const auto ops = results(t, begin);
    const auto d = t.rs.vreg.at(1);
    ASSERT_TRUE(ops.contains(d));
    EXPECT_EQ(ops.at(d).opcode, Op_IAdd);
    ASSERT_EQ(ops.at(d).operands.size(), 2u);
    EXPECT_EQ(ops.at(d).operands[1], c);
    const auto product = ops.at(d).operands[0];
    ASSERT_TRUE(ops.contains(product));
    EXPECT_EQ(ops.at(product).opcode, Op_IMul);
    ASSERT_EQ(ops.at(product).operands.size(), 2u);
    for (size_t i = 0; i < 2; ++i) {
        const auto id = ops.at(product).operands[i];
        ASSERT_TRUE(ops.contains(id));
        EXPECT_EQ(ops.at(id).opcode, Op_BitwiseAnd);
        EXPECT_EQ(ops.at(id).operands,
                  (std::vector<uint32_t>{i == 0 ? a : b, t.b.uconst_cache.at(0xffffff)}));
    }
    EXPECT_EQ(t.rs.vreg.size(), 4u);
}
TEST(IrLowerVop3Mad24, MaskIsAllocatedBeforeSourcesAndAccumulatorAfterProduct) {
    LoweringFixture t;
    auto in = instruction();
    for (size_t i = 0; i < 3; ++i) in.src[i] = {OperandKind::InlineInt, int(7 + i * 2)};
    const auto begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_mad24(t.b, t.rs, in, t.ok, false));
    ASSERT_TRUE(t.ok);
    const auto ops = results(t, begin);
    const auto& add = ops.at(t.rs.vreg.at(1));
    EXPECT_EQ(add.opcode, Op_IAdd);
    ASSERT_EQ(add.operands.size(), 2u);
    EXPECT_LT(t.b.uconst_cache.at(0xffffff), t.b.uconst_cache.at(7));
    EXPECT_LT(t.b.uconst_cache.at(0xffffff), t.b.uconst_cache.at(9));
    EXPECT_LT(add.operands[0], t.b.uconst_cache.at(11));
    EXPECT_EQ(add.operands[1], t.b.uconst_cache.at(11));
}
TEST(IrLowerVop3Mad24, EveryBadSourceKeepsTheOriginalFailureAndValueStage) {
    for (size_t bad = 0; bad < 3; ++bad) {
        LoweringFixture t;
        auto in = instruction();
        for (size_t i = 0; i < 3; ++i) in.src[i] = {OperandKind::InlineInt, int(7 + i * 2)};
        in.src[bad] = {OperandKind::None, 0};
        const auto begin = t.code_words();
        EXPECT_TRUE(ir::lower_vop3_mad24(t.b, t.rs, in, t.ok, false));
        EXPECT_FALSE(t.ok);
        ASSERT_TRUE(t.rs.vreg.contains(1));
        const auto ops = results(t, begin);
        EXPECT_EQ(ops.at(t.rs.vreg.at(1)).opcode, Op_IAdd);
    }
}
TEST(IrLowerVop3Mad24, CommonExecVccAndPredicationRemainCallerOwned) {
    LoweringFixture t;
    auto in = instruction();
    for (size_t i = 0; i < 3; ++i) in.src[i] = {OperandKind::InlineInt, int(7 + i * 2)};
    t.rs.exec = t.b.ucmp(Op_IEqual, t.b.uconst(0), t.b.uconst(1));
    t.rs.vcc = t.b.btrue();
    const auto exec = t.rs.exec, vcc = t.rs.vcc;
    const auto begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_mad24(t.b, t.rs, in, t.ok, false));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.rs.exec, exec);
    EXPECT_EQ(t.rs.vcc, vcc);
    for (const auto& [id, op] : results(t, begin)) EXPECT_NE(op.opcode, Op_Select);
}
TEST(IrLowerVop3Mad24, UnsupportedSignedAndReservedFormsFallBackWithoutSideEffects) {
    for (auto in : {instruction(0x142), instruction(kVop3OpcodeAddCoU32), instruction(0x02)}) {
        LoweringFixture t;
        const auto begin = t.code_words();
        EXPECT_FALSE(ir::lower_vop3_mad24(t.b, t.rs, in, t.ok, false));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), begin);
    }
    LoweringFixture t;
    auto in = instruction();
    in.fmt = Rdna2Format::VOP2;
    const auto begin = t.code_words();
    EXPECT_FALSE(ir::lower_vop3_mad24(t.b, t.rs, in, t.ok, false));
    EXPECT_TRUE(t.ok);
    EXPECT_TRUE(t.rs.vreg.empty());
    EXPECT_EQ(t.code_words(), begin);
}
