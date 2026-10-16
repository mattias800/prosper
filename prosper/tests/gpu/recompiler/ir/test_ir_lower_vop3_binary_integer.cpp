// Canonical e64 data operations read two real words and keep all mask and destination stages
// outside the value lowering. Phantom operands or modifiers must not gain data authority.
#include "gpu/recompiler/ir/lower/vop3_binary_integer_lowering.hpp"
#include "gpu/recompiler/ir/ops/vector_binary_integer.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstddef>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t op = 0x125) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP3;
    in.opcode = op;
    in.dst = {OperandKind::VGPR, 1};
    in.src[0] = {OperandKind::InlineInt, 59};
    in.src[1] = {OperandKind::InlineInt, 61};
    in.n_src = 2;
    return in;
}
bool lower(LoweringFixture& t, const Rdna2Inst& in) {
    return ir::lower_vop3_binary_integer(t.b, t.rs, in, t.ok, false);
}
}   // namespace
TEST(IrLowerVop3BinaryInteger, ExactlyFifteenDataAliasesShareTheirOriginalBuilders) {
    ASSERT_EQ(ir::vector_binary_integer_defs().size(), 15u);
    for (const auto& def : ir::vector_binary_integer_defs()) {
        const auto* old = ir::find_vector_op(Rdna2Format::VOP2, def.opcode - 0x100);
        ASSERT_NE(old, nullptr);
        EXPECT_EQ(def.build, old->build);
        const auto& effects = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(effects.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(effects.vdst_dwords, 1u);
        EXPECT_FALSE(effects.vdst_half);
        EXPECT_FALSE(effects.accumulates);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 2u);
    }
    size_t matched = 0;
    for (uint32_t op = 0; op < 1024; ++op) {
        LoweringFixture t;
        const auto before = t.code_words();
        if (lower(t, instruction(op))) {
            ++matched;
            EXPECT_TRUE(t.ok);
            EXPECT_TRUE(t.rs.vreg.contains(1));
        } else {
            EXPECT_TRUE(t.ok);
            EXPECT_TRUE(t.rs.vreg.empty());
            EXPECT_EQ(t.code_words(), before);
        }
    }
    EXPECT_EQ(matched, 15u);
}
TEST(IrLowerVop3BinaryInteger, BothInputsAreMaterializedBeforeMathWithoutReadingUnusedSrc2) {
    for (const auto& def : ir::vector_binary_integer_defs()) {
        LoweringFixture t;
        auto in = instruction(def.opcode);
        in.src[2] = {OperandKind::Special, 251};   // Not an actual operand of a canonical encoding.
        ASSERT_TRUE(lower(t, in));
        ASSERT_TRUE(t.ok);
        ASSERT_TRUE(t.constant_cached(59));
        ASSERT_TRUE(t.constant_cached(61));
        EXPECT_LT(t.b.uconst_cache.at(59), t.b.uconst_cache.at(61));
        EXPECT_LT(t.b.uconst_cache.at(61), t.rs.vreg.at(1));
    }
}
TEST(IrLowerVop3BinaryInteger, ReservedModifiersAndUnusedFieldsRefuseBeforePublication) {
    for (const auto& def : ir::vector_binary_integer_defs())
        for (uint32_t flag = 0; flag < 18; ++flag) {
            LoweringFixture t;
            auto in = instruction(def.opcode);
            if (flag < 3)
                in.src_abs[flag] = true;
            else if (flag < 6)
                in.src_neg[flag - 3] = true;
            else if (flag < 9)
                in.omod = static_cast<uint8_t>(flag - 5);
            else if (flag < 13)
                in.words[0] |= 1u << (flag + 2);
            else if (flag == 13)
                in.clamp = true;
            else if (flag == 14)
                in.words[1] |= 255u << 18;
            else if (flag == 15)
                in.has_dpp = true;
            else if (flag == 16)
                in.has_sdwa = true;
            else
                in.has_modifier = true;
            const auto before = t.code_words();
            ASSERT_TRUE(lower(t, in));
            EXPECT_FALSE(t.ok) << "opcode " << def.opcode << " flag " << flag;
            EXPECT_TRUE(t.rs.vreg.empty());
            EXPECT_EQ(t.code_words(), before);
            EXPECT_FALSE(t.constant_cached(59));
            EXPECT_FALSE(t.constant_cached(61));
        }
}
TEST(IrLowerVop3BinaryInteger, MissingOperandFailsWithoutPublishingAResult) {
    for (unsigned slot : {0u, 1u}) {
        LoweringFixture t;
        auto in = instruction();
        in.src[slot].kind = OperandKind::None;
        EXPECT_TRUE(lower(t, in));
        EXPECT_FALSE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
    }
}
TEST(IrLowerVop3BinaryInteger, TheCallerKeepsExecVccAndInactiveDestinationPredication) {
    LoweringFixture t;
    t.rs.exec = t.b.bfalse();
    t.rs.exec_narrowed = true;
    t.rs.vcc = t.b.btrue();
    t.rs.scc = t.b.bfalse();
    const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
    const auto before = t.code_words();
    ASSERT_TRUE(lower(t, instruction()));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.rs.exec, exec);
    EXPECT_EQ(t.rs.vcc, vcc);
    EXPECT_EQ(t.rs.scc, scc);
    for (size_t pc = before; pc < t.b.code.size();) {
        const auto size = t.b.code[pc] >> 16;
        ASSERT_NE(size, 0u);
        EXPECT_NE(t.b.code[pc] & 0xffffu, Op_Select);
        pc += size;
    }
}
TEST(IrLowerVop3BinaryInteger, DecoderUsesActualBinaryDataSourcesAndPreservesOtherShapes) {
    for (const auto& def : ir::vector_binary_integer_defs()) {
        const auto in = LoweringFixture::decode({0xd4000001u | (def.opcode << 16), 0x00020702u});
        EXPECT_EQ(in.n_src, 2u);
        EXPECT_EQ(in.src[2].kind, OperandKind::None);
    }
    // Hand ISA shapes: two real data words, full word result, no third operand.
    for (uint32_t op : {0x312u, 0x363u, 0x364u, 0x36bu}) {
        const auto in = LoweringFixture::decode({0xd4000001u | (op << 16), 0x00020702u});
        EXPECT_EQ(in.n_src, 2u);
        EXPECT_EQ(in.src[2].kind, OperandKind::None);
    }
    // Carry-in, DIV_SCALE, wide MAD, accumulating FMA and cross-lane forms have three sources.
    for (uint32_t op : {0x128u, 0x16du, 0x176u, 0x34bu, 0x377u, 0x270u}) {
        const auto in = LoweringFixture::decode({0xd4000001u | (op << 16), 0x00020702u});
        EXPECT_EQ(in.n_src, 3u) << "opcode " << op;
    }
    // Relative MOV is architecturally unary. Correct decoding grants no lowering/proof authority.
    for (uint32_t op : {0x181u, 0x1c2u}) {
        const auto unary = LoweringFixture::decode({0xd4000001u | (op << 16), 0x00020702u});
        EXPECT_EQ(unary.n_src, 1u);
        EXPECT_EQ(unary.src[1].kind, OperandKind::None);
        EXPECT_EQ(unary.src[2].kind, OperandKind::None);
        LoweringFixture t;
        EXPECT_FALSE(lower(t, unary));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
    }
    const auto carry = LoweringFixture::decode({0xd70f0001u, 0x00020702u});
    EXPECT_EQ(carry.n_src, 2u);   // The existing carry-out inventory remains unchanged.
    EXPECT_EQ(carry.sdst.kind, OperandKind::SGPR);
}
TEST(IrLowerVop3BinaryInteger, ForeignFormatsCarryAndUnimplementedMathRemainUnmatched) {
    for (uint32_t op : {0x101u, 0x109u, 0x10au, 0x10cu, 0x128u, 0x30fu}) {
        LoweringFixture t;
        EXPECT_FALSE(lower(t, instruction(op)));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
    }
    LoweringFixture t;
    auto in = instruction();
    in.fmt = Rdna2Format::VOP2;
    EXPECT_FALSE(lower(t, in));
    EXPECT_TRUE(t.ok);
}
