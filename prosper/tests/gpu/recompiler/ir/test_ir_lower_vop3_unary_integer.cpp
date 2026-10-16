// Numeric e64 unary stage: canonical words only, with all authority and predication left outside.
#include "gpu/recompiler/ir/lower/vop3_unary_integer_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/ops/vector_unary_integer_e64.hpp"
#include "gpu/recompiler/ir/ops/vector_integer_unary.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t op = 0x1b7, uint32_t source = 258) {
    return LoweringFixture::decode({0xd4000001u | (op << 16), source});
}
}   // namespace
TEST(IrLowerVop3UnaryInteger, FourAliasesShareTheExistingBuildersWithoutExtendingTheOldLookup) {
    ASSERT_EQ(vector_unary_integer_e64_defs().size(), 4u);
    for (const auto& def : vector_unary_integer_e64_defs()) {
        const auto* old = find_vector_integer_unary_op(Rdna2Format::VOP1, def.opcode - 0x180);
        ASSERT_NE(old, nullptr);
        EXPECT_EQ(def.build, old->build);
        EXPECT_EQ(find_vector_integer_unary_op(Rdna2Format::VOP3, def.opcode), nullptr);
        const auto& effects = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 1u);
        EXPECT_EQ(effects.vdst_dwords, 1u);
        EXPECT_FALSE(effects.vdst_half || effects.accumulates || effects.implicit_reads ||
                     effects.implicit_writes);
        EXPECT_EQ(effects.lane_flow, Rdna2LaneFlow::LaneWise);
    }
}
TEST(IrLowerVop3UnaryInteger, ReadsOnlyItsOneSourceBeforePublishingTheNumericWord) {
    for (const auto& def : vector_unary_integer_e64_defs()) {
        LoweringFixture t;
        t.set_sgpr(50, 0x81234567);
        EXPECT_TRUE(lower_vop3_unary_integer(t.b, t.rs, instruction(def.opcode, 50), t.ok, false));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.contains(1));
        EXPECT_EQ(t.rs.sreg.size(), 1u);
        EXPECT_FALSE(t.rs.sreg.contains(0));
    }
}
TEST(IrLowerVop3UnaryInteger, UnsupportedModifiersAndUnusedFieldsRefuseBeforePublication) {
    for (const auto& def : vector_unary_integer_e64_defs())
        for (uint32_t word : {0u, 1u})
            for (uint32_t bit = word ? 9 : 8; bit < (word ? 32u : 16u); ++bit) {
                std::vector<uint32_t> words{0xd4000001u | (def.opcode << 16), 258, 0x12345678};
                words[word] |= 1u << bit;
                const auto in = LoweringFixture::decode(words);
                LoweringFixture t;
                const auto before = t.code_words();
                EXPECT_TRUE(lower_vop3_unary_integer(t.b, t.rs, in, t.ok, false));
                EXPECT_FALSE(t.ok) << "opcode " << def.opcode << " word " << word << " bit " << bit;
                EXPECT_FALSE(t.rs.vreg.contains(1));
                EXPECT_EQ(t.code_words(), before);
            }
}
TEST(IrLowerVop3UnaryInteger, ARecognizedInstructionWithNoOperandDoesNotPublishAWord) {
    for (const auto& def : vector_unary_integer_e64_defs()) {
        LoweringFixture t;
        auto in = instruction(def.opcode);
        in.src[0] = {};
        EXPECT_TRUE(lower_vop3_unary_integer(t.b, t.rs, in, t.ok, false));
        EXPECT_FALSE(t.ok);
        EXPECT_FALSE(t.rs.vreg.contains(1));
    }
}
TEST(IrLowerVop3UnaryInteger, LeavesMasksAndInactiveDestinationPredicationWithTheCaller) {
    for (const auto& def : vector_unary_integer_e64_defs()) {
        LoweringFixture t;
        t.rs.exec = t.b.bfalse();
        t.rs.exec_narrowed = true;
        t.rs.vcc = t.b.btrue();
        t.rs.scc = t.b.bfalse();
        t.rs.vreg[2] = t.b.uconst(0x12345678);
        const auto old = t.b.uconst(0x87654321);
        t.rs.vreg[1] = old;
        const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
        EXPECT_TRUE(lower_vop3_unary_integer(t.b, t.rs, instruction(def.opcode), t.ok, false));
        EXPECT_TRUE(t.ok);
        EXPECT_NE(t.rs.vreg.at(1), old);
        EXPECT_EQ(t.rs.exec, exec);
        EXPECT_EQ(t.rs.vcc, vcc);
        EXPECT_EQ(t.rs.scc, scc);
        EXPECT_TRUE(t.rs.exec_narrowed);
    }
}
TEST(IrLowerVop3UnaryInteger, ArchitecturalUnaryArityDoesNotAdmitReservedMovOrOtherShapes) {
    for (uint32_t op : {0x181u, 0x1b7u, 0x1b8u, 0x1b9u, 0x1bau, 0x1a0u, 0x1c2u, 0x1d4u}) {
        const auto in = instruction(op);
        EXPECT_EQ(in.n_src, 1);
        EXPECT_EQ(in.src[1].kind, OperandKind::None);
        EXPECT_EQ(in.src[2].kind, OperandKind::None);
    }
    for (uint32_t op : {0x101u, 0x128u, 0x16du, 0x176u, 0x360u, 0x361u}) {
        const auto in = instruction(op);
        EXPECT_NE(in.n_src, 1) << "non-unary opcode " << op;
    }
    // Main's ISA arity covers relative/partial forms too; this data lowerer does not own them.
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::VOP3, 0x1c2).m0_relative,
              Rdna2M0Relative::Destination);
    EXPECT_TRUE(rdna2_instruction_effects(Rdna2Format::VOP3, 0x1d4).vdst_half);
    for (uint32_t op : {0x181u, 0x182u, 0x1bbu, 0x1c2u, 0x1c3u, 0x1d4u, 0x360u}) {
        LoweringFixture t;
        EXPECT_FALSE(lower_vop3_unary_integer(t.b, t.rs, instruction(op), t.ok, false));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
    }
    auto in = instruction();
    in.fmt = Rdna2Format::VOP1;
    LoweringFixture t;
    EXPECT_FALSE(lower_vop3_unary_integer(t.b, t.rs, in, t.ok, false));
    EXPECT_TRUE(t.ok);
}
