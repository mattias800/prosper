// Pin whole-dword packing, selected inline bits and unchanged source/modifier boundaries.
#include "gpu/recompiler/ir/lower/vop3_pack16_lowering.hpp"
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
Rdna2Inst instruction(uint32_t opcode = 0x311) {
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
TEST(IrLowerVop3Pack16, HandlesOnlyTheExistingFullWordTwoSourceOperation) {
    const auto& e = rdna2_instruction_effects(Rdna2Format::VOP3, 0x311);
    EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
    EXPECT_EQ(e.vdst_dwords, 1u);
    EXPECT_FALSE(e.vdst_half);
    EXPECT_EQ(rdna2_valu_source_count(Rdna2Format::VOP3, 0x311), 2u);
    LoweringFixture t;
    auto in = instruction();
    in.src[0] = {OperandKind::InlineInt, 7};
    in.src[1] = {OperandKind::InlineInt, 9};
    const auto begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_pack16(t.b, t.rs, in, t.ok, false));
    ASSERT_TRUE(t.ok);
    const auto ops = results(t, begin);
    const auto& combine = ops.at(t.rs.vreg.at(1));
    EXPECT_EQ(combine.opcode, Op_BitwiseOr);
    ASSERT_EQ(combine.operands.size(), 2u);
    EXPECT_EQ(combine.operands[0], t.b.uconst_cache.at(7));
    const auto& shift = ops.at(combine.operands[1]);
    EXPECT_EQ(shift.opcode, Op_ShiftLeftLogical);
    EXPECT_EQ(shift.operands,
              (std::vector<uint32_t>{t.b.uconst_cache.at(9), t.b.uconst_cache.at(16)}));
}
TEST(IrLowerVop3Pack16, InlineSourcesSelectTheirActualHalfWidthBusBits) {
    for (uint32_t select = 0; select < 16; ++select) {
        LoweringFixture t;
        auto in = instruction();
        in.vop3p_opsel = select;
        in.src[0] = {OperandKind::InlineFloat, 242};
        in.src[1] = {OperandKind::InlineInt, -2};
        const auto begin = t.code_words();
        ASSERT_TRUE(ir::lower_vop3_pack16(t.b, t.rs, in, t.ok, false));
        ASSERT_TRUE(t.ok);
        const auto ops = results(t, begin);
        const auto& combine = ops.at(t.rs.vreg.at(1));
        EXPECT_EQ(combine.operands[0], t.b.uconst_cache.at((select & 1) ? 0 : 0x3c00));
        const auto& shift = ops.at(combine.operands[1]);
        EXPECT_EQ(shift.operands,
                  (std::vector<uint32_t>{t.b.uconst_cache.at((select & 2) ? 0xffff : 0xfffe),
                                         t.b.uconst_cache.at(16)}));
    }
}
TEST(IrLowerVop3Pack16, ModeledModifierRefusalsPrecedeReadsAndDestinationInsertion) {
    for (uint32_t flag = 0; flag < 6; ++flag) {
        LoweringFixture t;
        auto in = instruction();
        if (flag < 2)
            in.src_abs[flag] = true;
        else if (flag < 4)
            in.src_neg[flag - 2] = true;
        else if (flag == 4)
            in.clamp = true;
        else
            in.omod = 1;
        const auto begin = t.code_words(), constants = t.b.uconst_cache.size();
        EXPECT_TRUE(ir::lower_vop3_pack16(t.b, t.rs, in, t.ok, false));
        EXPECT_FALSE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), begin);
        EXPECT_EQ(t.b.uconst_cache.size(), constants);
    }
}
TEST(IrLowerVop3Pack16, RequiredBadSourceRefusesButTheUnusedThirdSourceIsNotRead) {
    for (int bad = -1; bad < 2; ++bad) {
        LoweringFixture t;
        auto in = instruction();
        in.src[0] = {OperandKind::InlineInt, 7};
        in.src[1] = {OperandKind::InlineInt, 9};
        in.src[2] = {OperandKind::None, 0};
        in.src_abs[2] = in.src_neg[2] = true;
        if (bad >= 0) in.src[bad] = {OperandKind::None, 0};
        ASSERT_TRUE(ir::lower_vop3_pack16(t.b, t.rs, in, t.ok, false));
        EXPECT_EQ(t.ok, bad < 0);
        EXPECT_TRUE(t.rs.vreg.contains(1));
    }
}
TEST(IrLowerVop3Pack16, CallerKeepsExecVccSccAndFinalPredication) {
    LoweringFixture t;
    auto in = instruction();
    in.src[0] = {OperandKind::InlineInt, 7};
    in.src[1] = {OperandKind::InlineInt, 9};
    t.rs.exec = t.b.ucmp(Op_IEqual, t.b.uconst(0), t.b.uconst(1));
    t.rs.vcc = t.b.btrue();
    t.rs.scc = t.b.bfalse();
    const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
    const auto begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_pack16(t.b, t.rs, in, t.ok, false));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.rs.exec, exec);
    EXPECT_EQ(t.rs.vcc, vcc);
    EXPECT_EQ(t.rs.scc, scc);
    const auto ops = results(t, begin);
    EXPECT_EQ(ops.at(t.rs.vreg.at(1)).opcode, Op_BitwiseOr);
}
TEST(IrLowerVop3Pack16, OtherOperationsAndFormatsFallBackWithoutSideEffects) {
    for (auto in : {instruction(0x303), instruction(kVop3OpcodeAddCoU32), instruction(0x02)}) {
        LoweringFixture t;
        const auto begin = t.code_words();
        EXPECT_FALSE(ir::lower_vop3_pack16(t.b, t.rs, in, t.ok, false));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), begin);
    }
    LoweringFixture t;
    auto in = instruction();
    in.fmt = Rdna2Format::VOP3P;
    const auto begin = t.code_words();
    EXPECT_FALSE(ir::lower_vop3_pack16(t.b, t.rs, in, t.ok, false));
    EXPECT_TRUE(t.ok);
    EXPECT_TRUE(t.rs.vreg.empty());
    EXPECT_EQ(t.code_words(), begin);
}
