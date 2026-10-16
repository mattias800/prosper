// Pin unsigned source words, independent saturation, publication and the existing caller boundary.
#include "gpu/recompiler/ir/lower/vop3_pack_saturate_lowering.hpp"
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
Rdna2Inst instruction(uint32_t opcode = 0x36a) {
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
TEST(IrLowerVop3PackSaturate, HandlesTheExistingFullWordTwoSourceOperation) {
    LoweringFixture t;
    auto in = instruction();
    const auto& e = rdna2_instruction_effects(in.fmt, in.opcode);
    EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
    EXPECT_EQ(e.vdst_dwords, 1u);
    EXPECT_FALSE(e.vdst_half);
    EXPECT_EQ(rdna2_valu_source_count(in.fmt, in.opcode), 2u);
    in.src[0] = {OperandKind::InlineInt, 7};
    in.src[1] = {OperandKind::InlineInt, 9};
    const auto begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_pack_saturate(t.b, t.rs, in, t.ok, false));
    ASSERT_TRUE(t.ok);
    const auto ops = results(t, begin);
    const auto& combine = ops.at(t.rs.vreg.at(1));
    EXPECT_EQ(combine.opcode, Op_BitwiseOr);
    ASSERT_EQ(combine.operands.size(), 2u);
    const auto low = combine.operands[0];
    const auto& lo = ops.at(low);
    EXPECT_EQ(lo.opcode, Op_ExtInst);
    EXPECT_EQ(lo.operands, (std::vector<uint32_t>{t.b.glsl, Glsl_UMin, t.b.uconst_cache.at(7),
                                                  t.b.uconst_cache.at(0xffff)}));
    const auto& shift = ops.at(combine.operands[1]);
    EXPECT_EQ(shift.opcode, Op_ShiftLeftLogical);
    ASSERT_EQ(shift.operands.size(), 2u);
    EXPECT_EQ(shift.operands[1], t.b.uconst_cache.at(16));
    const auto high = shift.operands[0];
    const auto& hi = ops.at(high);
    EXPECT_EQ(hi.opcode, Op_ExtInst);
    EXPECT_EQ(hi.operands, (std::vector<uint32_t>{t.b.glsl, Glsl_UMin, t.b.uconst_cache.at(9),
                                                  t.b.uconst_cache.at(0xffff)}));
    EXPECT_LT(low, high);
    EXPECT_LT(high, combine.operands[1]);
}
TEST(IrLowerVop3PackSaturate, InlineFloatAndNegativeIntegerKeepTheirFullUnsignedBusWords) {
    LoweringFixture t;
    auto in = instruction();
    in.src[0] = {OperandKind::InlineFloat, 242};
    in.src[1] = {OperandKind::InlineInt, -1};
    const auto begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_pack_saturate(t.b, t.rs, in, t.ok, false));
    ASSERT_TRUE(t.ok);
    const auto ops = results(t, begin);
    const auto& combine = ops.at(t.rs.vreg.at(1));
    EXPECT_EQ(ops.at(combine.operands[0]).operands,
              (std::vector<uint32_t>{t.b.glsl, Glsl_UMin, t.b.uconst_cache.at(0x3f800000),
                                     t.b.uconst_cache.at(0xffff)}));
    const auto& shift = ops.at(combine.operands[1]);
    EXPECT_EQ(ops.at(shift.operands[0]).operands,
              (std::vector<uint32_t>{t.b.glsl, Glsl_UMin, t.b.uconst_cache.at(0xffffffff),
                                     t.b.uconst_cache.at(0xffff)}));
}
TEST(IrLowerVop3PackSaturate, RequiredBadSourcesKeepErrorsAndTheUnusedThirdSourceIsIgnored) {
    for (int bad = -1; bad < 2; ++bad) {
        LoweringFixture t;
        auto in = instruction();
        in.src[0] = {OperandKind::InlineInt, 7};
        in.src[1] = {OperandKind::InlineInt, 9};
        in.src[2] = {OperandKind::None, 0};
        if (bad >= 0) in.src[bad] = {OperandKind::None, 0};
        ASSERT_TRUE(ir::lower_vop3_pack_saturate(t.b, t.rs, in, t.ok, false));
        EXPECT_EQ(t.ok, bad < 0);
        EXPECT_TRUE(t.rs.vreg.contains(1));
    }
}
TEST(IrLowerVop3PackSaturate, ExistingModifierAndClampBoundariesRemainWithTheCaller) {
    LoweringFixture t;
    auto in = instruction();
    in.src[0] = {OperandKind::InlineInt, 7};
    in.src[1] = {OperandKind::InlineInt, 9};
    for (uint32_t k = 0; k < 3; ++k) in.src_abs[k] = in.src_neg[k] = true;
    in.clamp = true;
    in.omod = 3;
    in.vop3p_opsel = 15;
    in.words[0] = 15u << 11;
    ASSERT_TRUE(ir::lower_vop3_pack_saturate(t.b, t.rs, in, t.ok, false));
    EXPECT_TRUE(t.ok);
    EXPECT_TRUE(t.rs.vreg.contains(1));
}
TEST(IrLowerVop3PackSaturate, CallerKeepsExecVccSccAndFinalPredication) {
    LoweringFixture t;
    auto in = instruction();
    in.src[0] = {OperandKind::InlineInt, 7};
    in.src[1] = {OperandKind::InlineInt, 9};
    t.rs.exec = t.b.bfalse();
    t.rs.exec_narrowed = true;
    t.rs.vcc = t.b.btrue();
    t.rs.scc = t.b.bfalse();
    const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
    const auto begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_pack_saturate(t.b, t.rs, in, t.ok, false));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.rs.exec, exec);
    EXPECT_EQ(t.rs.vcc, vcc);
    EXPECT_EQ(t.rs.scc, scc);
    const auto ops = results(t, begin);
    EXPECT_EQ(ops.at(t.rs.vreg.at(1)).opcode, Op_BitwiseOr);
}
TEST(IrLowerVop3PackSaturate, OtherOperationsAndFormatsFallBackWithoutSideEffects) {
    for (auto in : {instruction(0x369), instruction(0x36b), instruction(kVop3OpcodeAddCoU32),
                    instruction(0x311)}) {
        LoweringFixture t;
        const auto begin = t.code_words();
        EXPECT_FALSE(ir::lower_vop3_pack_saturate(t.b, t.rs, in, t.ok, false));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), begin);
    }
    LoweringFixture t;
    auto in = instruction();
    in.fmt = Rdna2Format::VOP3P;
    const auto begin = t.code_words();
    EXPECT_FALSE(ir::lower_vop3_pack_saturate(t.b, t.rs, in, t.ok, false));
    EXPECT_TRUE(t.ok);
    EXPECT_TRUE(t.rs.vreg.empty());
    EXPECT_EQ(t.code_words(), begin);
}
