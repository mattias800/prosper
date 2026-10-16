// The numeric partial-word stage may change only the selected low destination half.
#include "gpu/recompiler/ir/lower/vop3_half_integer_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/ops/vector_half_integer_e64.hpp"
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t op = 0x1d2, uint32_t source = 258) {
    return LoweringFixture::decode({0xd4000001u | (op << 16), source});
}
}   // namespace
TEST(IrLowerVop3HalfInteger, TwoAliasesShareTheExistingPartialWordDefinitions) {
    ASSERT_EQ(vector_half_integer_e64_defs().size(), 2u);
    for (const auto& def : vector_half_integer_e64_defs()) {
        const auto* old =
            find_vector_half_integer_conversion_op(Rdna2Format::VOP1, def.opcode - 0x180);
        ASSERT_NE(old, nullptr);
        EXPECT_EQ(def.build, old->build);
        EXPECT_EQ(find_vector_half_integer_conversion_op(Rdna2Format::VOP3, def.opcode), nullptr);
        const auto& effects = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 1u);
        EXPECT_EQ(instruction(def.opcode).n_src, 1);
        EXPECT_TRUE(effects.vdst_half);
        EXPECT_EQ(effects.vdst_dwords, 1u);
        EXPECT_FALSE(effects.accumulates || effects.implicit_reads || effects.implicit_writes);
        EXPECT_EQ(effects.lane_flow, Rdna2LaneFlow::LaneWise);
    }
}
TEST(IrLowerVop3HalfInteger, ReadsOneSourceAndRetainsTheOldHighHalf) {
    for (const auto& def : vector_half_integer_e64_defs()) {
        LoweringFixture t;
        t.set_sgpr(50, 0x40003e00);
        const auto old = t.b.uconst(0x3555abcd);
        t.rs.vreg[1] = old;
        EXPECT_TRUE(
            lower_vop3_half_integer(t.b, t.rs, instruction(def.opcode, 50), t.ok, false, old));
        EXPECT_TRUE(t.ok);
        EXPECT_NE(t.rs.vreg.at(1), old);
        EXPECT_TRUE(t.constant_cached(0xffff0000));
        EXPECT_EQ(t.rs.sreg.size(), 1u);
        EXPECT_FALSE(t.rs.sreg.contains(0));
    }
}
TEST(IrLowerVop3HalfInteger, AllModifiersAndUnusedSourcesRefuseBeforeWriting) {
    for (const auto& def : vector_half_integer_e64_defs())
        for (uint32_t word : {0u, 1u})
            for (uint32_t bit = word ? 9 : 8; bit < (word ? 32u : 16u); ++bit) {
                std::vector<uint32_t> words{0xd4000001u | (def.opcode << 16), 258, 0x3c00};
                words[word] |= 1u << bit;
                const auto in = LoweringFixture::decode(words);
                LoweringFixture t;
                const auto old = t.b.uconst(0x3555abcd);
                const auto before = t.code_words();
                EXPECT_TRUE(lower_vop3_half_integer(t.b, t.rs, in, t.ok, false, old));
                EXPECT_FALSE(t.ok) << def.opcode << " word " << word << " bit " << bit;
                EXPECT_TRUE(t.rs.vreg.empty());
                EXPECT_EQ(t.code_words(), before);
            }
}
TEST(IrLowerVop3HalfInteger, MissingOperandDoesNotPublishAValue) {
    for (const auto& def : vector_half_integer_e64_defs()) {
        LoweringFixture t;
        auto in = instruction(def.opcode);
        in.src[0] = {};
        const auto old = t.b.uconst(0x3555abcd);
        EXPECT_TRUE(lower_vop3_half_integer(t.b, t.rs, in, t.ok, false, old));
        EXPECT_FALSE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
    }
}
TEST(IrLowerVop3HalfInteger, CallerRetainsMasksAndExecPredication) {
    for (const auto& def : vector_half_integer_e64_defs()) {
        LoweringFixture t;
        t.rs.exec = t.b.bfalse();
        t.rs.exec_narrowed = true;
        t.rs.vcc = t.b.btrue();
        t.rs.scc = t.b.bfalse();
        t.rs.vreg[2] = t.b.uconst(0x40003e00);
        const auto old = t.b.uconst(0x3555abcd);
        t.rs.vreg[1] = old;
        const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
        EXPECT_TRUE(lower_vop3_half_integer(t.b, t.rs, instruction(def.opcode), t.ok, false, old));
        EXPECT_TRUE(t.ok);
        EXPECT_NE(t.rs.vreg.at(1), old);
        EXPECT_EQ(t.rs.exec, exec);
        EXPECT_EQ(t.rs.vcc, vcc);
        EXPECT_EQ(t.rs.scc, scc);
        EXPECT_TRUE(t.rs.exec_narrowed);
    }
}
TEST(IrLowerVop3HalfInteger, OtherConversionsAndReservedShapesRemainUnmatched) {
    for (uint32_t op : {0x181u, 0x187u, 0x188u, 0x1d0u, 0x1d1u, 0x1d4u, 0x360u}) {
        LoweringFixture t;
        const auto old = t.b.uconst(0x3555abcd);
        const auto before = t.code_words();
        EXPECT_FALSE(lower_vop3_half_integer(t.b, t.rs, instruction(op), t.ok, false, old));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), before);
    }
    LoweringFixture t;
    auto in = instruction();
    in.fmt = Rdna2Format::VOP1;
    EXPECT_FALSE(lower_vop3_half_integer(t.b, t.rs, in, t.ok, false, t.b.uconst(0)));
    EXPECT_TRUE(t.ok);
}
