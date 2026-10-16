// Pin all selector bytes, destination placement, prefetched sources and unchanged refusals.
#include "gpu/recompiler/ir/lower/vop3_permute_lowering.hpp"
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
Rdna2Inst instruction(uint32_t opcode = 0x344) {
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
TEST(IrLowerVop3Permute, FourSelectorsReadS2AndDestinationPlacesFourBytes) {
    LoweringFixture t;
    const auto a = t.b.uconst(0x11223344), b = t.b.uconst(0x55667788), c = t.b.uconst(0x04000501);
    t.rs.vreg[2] = a;
    t.rs.vreg[3] = b;
    t.rs.vreg[5] = c;
    const auto begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_permute(t.b, t.rs, instruction(), t.ok, false));
    ASSERT_TRUE(t.ok);
    const auto ops = results(t, begin);
    size_t selectors = 0;
    for (const auto& [id, op] : ops)
        if (op.opcode == Op_BitFieldUExtract && op.operands.at(0) == c) {
            ASSERT_EQ(op.operands.size(), 3u);
            const auto& offset = ops.at(op.operands[1]);
            EXPECT_EQ(offset.opcode, Op_ExtInst);
            EXPECT_EQ(offset.operands,
                      (std::vector<uint32_t>{t.b.glsl, Glsl_UMin,
                                             t.b.uconst_cache.at(uint32_t(selectors * 8)),
                                             t.b.uconst_cache.at(32)}));
            const auto& count = ops.at(op.operands[2]);
            EXPECT_EQ(count.opcode, Op_ExtInst);
            ASSERT_EQ(count.operands.size(), 4u);
            EXPECT_EQ(count.operands[0], t.b.glsl);
            EXPECT_EQ(count.operands[1], Glsl_UMin);
            EXPECT_EQ(count.operands[2], t.b.uconst_cache.at(8));
            const auto& remaining = ops.at(count.operands[3]);
            EXPECT_EQ(remaining.opcode, Op_ISub);
            EXPECT_EQ(remaining.operands,
                      (std::vector<uint32_t>{t.b.uconst_cache.at(32), op.operands[1]}));
            ++selectors;
        }
    EXPECT_EQ(selectors, 4u);
    auto result = t.rs.vreg.at(1);
    for (int byte = 3; byte >= 0; --byte) {
        ASSERT_TRUE(ops.contains(result));
        const auto& combine = ops.at(result);
        EXPECT_EQ(combine.opcode, Op_BitwiseOr);
        ASSERT_EQ(combine.operands.size(), 2u);
        const auto& placed = ops.at(combine.operands[1]);
        EXPECT_EQ(placed.opcode, Op_ShiftLeftLogical);
        ASSERT_EQ(placed.operands.size(), 2u);
        EXPECT_EQ(placed.operands[1], t.b.uconst_cache.at(uint32_t(byte * 8)));
        result = combine.operands[0];
    }
    EXPECT_EQ(result, t.b.uconst_cache.at(0));
}
TEST(IrLowerVop3Permute, AllThreeArgumentReadsPrecedeHelperArithmetic) {
    LoweringFixture t;
    auto in = instruction();
    for (size_t i = 0; i < 3; ++i) in.src[i] = {OperandKind::InlineInt, int(7 + i * 2)};
    const auto begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_permute(t.b, t.rs, in, t.ok, false));
    ASSERT_TRUE(t.ok);
    const auto ops = results(t, begin);
    bool found = false;
    for (const auto& [id, op] : ops)
        if (op.opcode == Op_BitFieldUExtract) {
            EXPECT_LT(t.b.uconst_cache.at(7), id);
            EXPECT_LT(t.b.uconst_cache.at(9), id);
            EXPECT_LT(t.b.uconst_cache.at(11), id);
            found = true;
            break;
        }
    EXPECT_TRUE(found);
}
TEST(IrLowerVop3Permute, EveryModifierRefusesBeforeReadingOrInsertingTheDestination) {
    for (unsigned flag = 0; flag < 8; ++flag) {
        LoweringFixture t;
        auto in = instruction();
        if (flag < 3)
            in.src_abs[flag] = true;
        else if (flag < 6)
            in.src_neg[flag - 3] = true;
        else if (flag == 6)
            in.clamp = true;
        else
            in.omod = 1;
        const auto begin = t.code_words(), constants = t.b.uconst_cache.size();
        EXPECT_TRUE(ir::lower_vop3_permute(t.b, t.rs, in, t.ok, false));
        EXPECT_FALSE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), begin);
        EXPECT_EQ(t.b.uconst_cache.size(), constants);
    }
}
TEST(IrLowerVop3Permute, EachBadSourceKeepsTheOriginalErrorAndValueStage) {
    for (size_t bad = 0; bad < 3; ++bad) {
        LoweringFixture t;
        auto in = instruction();
        for (size_t i = 0; i < 3; ++i) in.src[i] = {OperandKind::InlineInt, int(7 + i * 2)};
        in.src[bad] = {OperandKind::None, 0};
        const auto begin = t.code_words();
        EXPECT_TRUE(ir::lower_vop3_permute(t.b, t.rs, in, t.ok, false));
        EXPECT_FALSE(t.ok);
        ASSERT_TRUE(t.rs.vreg.contains(1));
        const auto ops = results(t, begin);
        EXPECT_EQ(ops.at(t.rs.vreg.at(1)).opcode, Op_BitwiseOr);
    }
}
TEST(IrLowerVop3Permute, CallerRetainsExecVccAndDestinationPredication) {
    LoweringFixture t;
    auto in = instruction();
    for (size_t i = 0; i < 3; ++i) in.src[i] = {OperandKind::InlineInt, int(7 + i * 2)};
    t.rs.exec = t.b.ucmp(Op_IEqual, t.b.uconst(0), t.b.uconst(1));
    t.rs.vcc = t.b.btrue();
    const auto exec = t.rs.exec, vcc = t.rs.vcc;
    const auto begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_permute(t.b, t.rs, in, t.ok, false));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.rs.exec, exec);
    EXPECT_EQ(t.rs.vcc, vcc);
    const auto ops = results(t, begin);
    EXPECT_EQ(ops.at(t.rs.vreg.at(1)).opcode, Op_BitwiseOr);
}
TEST(IrLowerVop3Permute, OtherOperationsAndFormatsFallBackWithoutSideEffects) {
    for (auto in : {instruction(0x143), instruction(kVop3OpcodeAddCoU32), instruction(0x02)}) {
        LoweringFixture t;
        const auto begin = t.code_words();
        EXPECT_FALSE(ir::lower_vop3_permute(t.b, t.rs, in, t.ok, false));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), begin);
    }
    LoweringFixture t;
    auto in = instruction();
    in.fmt = Rdna2Format::VOP2;
    const auto begin = t.code_words();
    EXPECT_FALSE(ir::lower_vop3_permute(t.b, t.rs, in, t.ok, false));
    EXPECT_TRUE(t.ok);
    EXPECT_TRUE(t.rs.vreg.empty());
    EXPECT_EQ(t.code_words(), begin);
}
