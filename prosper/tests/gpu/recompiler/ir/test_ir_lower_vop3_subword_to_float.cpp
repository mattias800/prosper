// Five exact subword results use the old builders while canonical guards refuse before publication.
#include "gpu/recompiler/ir/lower/vop3_subword_to_float_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/ops/vector_subword_to_float_e64.hpp"
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <initializer_list>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t opcode = 0x191, uint32_t source = 258) {
    return LoweringFixture::decode({0xd4000001u | (opcode << 16), source});
}
}   // namespace
TEST(IrLowerVop3SubwordToFloat, FiveAliasesShareBuildersWithoutExtendingTheOldConversionSpan) {
    ASSERT_EQ(vector_conversion_defs().size(), 13u);
    ASSERT_EQ(vector_subword_to_float_e64_defs().size(), 5u);
    for (const auto& def : vector_subword_to_float_e64_defs()) {
        const auto* old = find_vector_conversion_op(Rdna2Format::VOP1, def.opcode - 0x180);
        ASSERT_NE(old, nullptr);
        EXPECT_EQ(def.build, old->build);
        EXPECT_EQ(find_vector_conversion_op(Rdna2Format::VOP3, def.opcode), nullptr);
        const auto& effects = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 1u);
        EXPECT_EQ(effects.vdst_dwords, 1u);
        EXPECT_EQ(effects.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_FALSE(effects.vdst_half || effects.accumulates || effects.implicit_reads ||
                     effects.implicit_writes);
    }
}
TEST(IrLowerVop3SubwordToFloat, MaterializesOnlyTheArchitecturalSource) {
    for (const auto& def : vector_subword_to_float_e64_defs()) {
        LoweringFixture t;
        t.set_sgpr(50, 0x81234567);
        EXPECT_TRUE(lower_vop3_subword_to_float(t.b, t.rs, instruction(def.opcode, 50), t.ok));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.contains(1));
        EXPECT_EQ(t.rs.sreg.size(), 1u);
        EXPECT_FALSE(t.rs.sreg.contains(0));
    }
}
TEST(IrLowerVop3SubwordToFloat, ModifiersAndUnusedRawFieldsRefuseBeforeEmission) {
    for (const auto& def : vector_subword_to_float_e64_defs())
        for (uint32_t word : {0u, 1u})
            for (uint32_t bit = word ? 9 : 8; bit < (word ? 32u : 16u); ++bit) {
                std::vector<uint32_t> words{0xd4000001u | (def.opcode << 16), 258, 0x81234567};
                words[word] |= 1u << bit;
                const auto in = LoweringFixture::decode(words);
                LoweringFixture t;
                const auto before = t.code_words();
                EXPECT_TRUE(lower_vop3_subword_to_float(t.b, t.rs, in, t.ok));
                EXPECT_FALSE(t.ok) << def.opcode << " word " << word << " bit " << bit;
                EXPECT_FALSE(t.rs.vreg.contains(1));
                EXPECT_EQ(t.code_words(), before);
            }
}
TEST(IrLowerVop3SubwordToFloat, MissingOperandDoesNotPublishAResult) {
    for (const auto& def : vector_subword_to_float_e64_defs()) {
        LoweringFixture t;
        auto in = instruction(def.opcode);
        in.src[0] = {};
        EXPECT_TRUE(lower_vop3_subword_to_float(t.b, t.rs, in, t.ok));
        EXPECT_FALSE(t.ok);
        EXPECT_FALSE(t.rs.vreg.contains(1));
    }
}
TEST(IrLowerVop3SubwordToFloat, ModeBitsDoNotChangeTheExactByteConversionGraph) {
    for (const auto& def : vector_subword_to_float_e64_defs()) {
        std::vector<uint32_t> reference;
        for (uint32_t mode = 0; mode < 256; ++mode)
            for (uint32_t ieee : {0u, 1u}) {
                LoweringFixture t;
                t.b.compute_pgm_rsrc1 = (mode << 12) | (ieee << 23);
                t.rs.vreg[2] = t.b.uconst(0x81234567);
                ASSERT_TRUE(lower_vop3_subword_to_float(t.b, t.rs, instruction(def.opcode), t.ok));
                ASSERT_TRUE(t.ok);
                if (reference.empty()) reference = t.b.code;
                EXPECT_EQ(t.b.code, reference)
                    << def.opcode << " mode " << mode << " IEEE " << ieee;
            }
    }
}
TEST(IrLowerVop3SubwordToFloat, ExactResultsNeedNoModeGuessForComputeOrGraphics) {
    for (const auto& def : vector_subword_to_float_e64_defs()) {
        std::vector<uint32_t> reference;
        for (uint32_t rsrc1 : {0u, 0x20000u, 0x30000u, UINT32_MAX}) {
            LoweringFixture t;
            t.b.compute_pgm_rsrc1 = rsrc1;
            t.rs.vreg[2] = t.b.uconst(0x81234567);
            ASSERT_TRUE(lower_vop3_subword_to_float(t.b, t.rs, instruction(def.opcode), t.ok));
            ASSERT_TRUE(t.ok);
            if (reference.empty()) reference = t.b.code;
            EXPECT_EQ(t.b.code, reference);
        }
        LoweringFixture vertex;
        vertex.b.is_compute = false;
        vertex.b.is_vertex = true;
        vertex.b.vertex_float_mode = {};
        vertex.rs.vreg[2] = vertex.b.uconst(0x81234567);
        ASSERT_TRUE(
            lower_vop3_subword_to_float(vertex.b, vertex.rs, instruction(def.opcode), vertex.ok));
        EXPECT_TRUE(vertex.ok);
        EXPECT_EQ(vertex.b.code, reference);
    }
}
TEST(IrLowerVop3SubwordToFloat, LeavesDestinationPredicationAndMasksWithTheCaller) {
    for (const auto& def : vector_subword_to_float_e64_defs()) {
        LoweringFixture t;
        t.rs.exec = t.b.bfalse();
        t.rs.exec_narrowed = true;
        t.rs.vcc = t.b.btrue();
        t.rs.scc = t.b.bfalse();
        t.rs.vreg[2] = t.b.uconst(0x81234567);
        const auto old = t.b.uconst(0x87654321);
        t.rs.vreg[1] = old;
        const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
        EXPECT_TRUE(lower_vop3_subword_to_float(t.b, t.rs, instruction(def.opcode), t.ok));
        EXPECT_TRUE(t.ok);
        EXPECT_NE(t.rs.vreg.at(1), old);
        EXPECT_EQ(t.rs.exec, exec);
        EXPECT_EQ(t.rs.vcc, vcc);
        EXPECT_EQ(t.rs.scc, scc);
        EXPECT_TRUE(t.rs.exec_narrowed);
    }
}
TEST(IrLowerVop3SubwordToFloat, DoesNotClaimOtherConversionsOrReservedRoutes) {
    for (uint32_t opcode : {0x181u, 0x185u, 0x190u, 0x195u, 0x1c2u, 0x1d4u}) {
        LoweringFixture t;
        EXPECT_FALSE(lower_vop3_subword_to_float(t.b, t.rs, instruction(opcode), t.ok));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
    }
    auto in = instruction();
    in.fmt = Rdna2Format::VOP1;
    LoweringFixture t;
    EXPECT_FALSE(lower_vop3_subword_to_float(t.b, t.rs, in, t.ok));
    EXPECT_TRUE(t.ok);
}
