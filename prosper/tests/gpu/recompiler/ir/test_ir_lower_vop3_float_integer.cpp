// Numeric stage contracts, independently of the caller's source proofs and final EXEC write.
#include <vector>
#include "gpu/recompiler/ir/lower/vop3_float_integer_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/ops/vector_float_integer_e64.hpp"
#include "gpu/recompiler/ir/float_integer_e64_cases.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <gtest/gtest.h>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace cases = prosper::test::ir::float_integer_e64;
TEST(IrLowerVop3FloatInteger, TwoAliasesReuseTheOldBuildersAndUnaryEffects) {
    ASSERT_EQ(vector_float_integer_e64_defs().size(), 2u);
    EXPECT_EQ(vector_conversion_defs().size(), 13u);
    for (const auto& def : vector_float_integer_e64_defs()) {
        const auto* old = find_vector_conversion_op(Rdna2Format::VOP1, def.opcode - 0x180);
        ASSERT_NE(old, nullptr);
        EXPECT_EQ(def.build, old->build);
        EXPECT_EQ(find_vector_conversion_op(def.fmt, def.opcode), nullptr);
        const auto& effects = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 1u);
        EXPECT_EQ(effects.vdst_dwords, 1u);
        EXPECT_FALSE(effects.vdst_half || effects.accumulates || effects.implicit_reads ||
                     effects.implicit_writes);
        EXPECT_EQ(effects.lane_flow, Rdna2LaneFlow::LaneWise);
    }
}
TEST(IrLowerVop3FloatInteger, AbsNegAndIgnoredOmodNeedNoFloatLaunchMode) {
    for (const auto& def : vector_float_integer_e64_defs())
        for (uint32_t modifiers = 0; modifiers < 4; ++modifiers)
            for (uint32_t omod = 0; omod < 4; ++omod) {
                LoweringFixture t;
                t.b.compute_pgm_rsrc1 = UINT32_MAX;
                t.set_sgpr(50, 0xbfe00000);
                cases::Form f;
                f.source = 50;
                const auto in =
                    LoweringFixture::decode(cases::words(def.opcode, f, modifiers, omod));
                const auto exec = t.rs.exec;
                EXPECT_TRUE(lower_vop3_float_integer(t.b, t.rs, in, t.ok));
                EXPECT_TRUE(t.ok);
                EXPECT_TRUE(t.rs.vreg.contains(1));
                EXPECT_EQ(t.rs.sreg.size(), 1u);
                EXPECT_FALSE(t.rs.sreg.contains(0));
                EXPECT_EQ(t.rs.exec, exec);
            }
}
TEST(IrLowerVop3FloatInteger, ClampUnusedFieldsAndOpSelRefuseBeforePublication) {
    for (const auto& def : vector_float_integer_e64_defs())
        for (uint32_t word = 0; word < 2; ++word)
            for (uint32_t bit = 9; bit < (word ? 32u : 16u); ++bit) {
                if (word && bit >= 27 && bit <= 29) continue; // OMOD and SRC0 NEG are valid.
                auto raw = cases::words(def.opcode);
                raw[word] |= 1u << bit;
                LoweringFixture t;
                const auto before = t.code_words();
                const auto in = LoweringFixture::decode(raw);
                EXPECT_TRUE(lower_vop3_float_integer(t.b, t.rs, in, t.ok));
                EXPECT_FALSE(t.ok) << word << ':' << bit;
                EXPECT_FALSE(t.rs.vreg.contains(1));
                EXPECT_EQ(t.code_words(), before);
            }
}
TEST(IrLowerVop3FloatInteger, MissingOrUnsupportedOperandsDoNotPublishAResult) {
    for (const auto& def : vector_float_integer_e64_defs())
        for (bool missing : {false, true}) {
            auto in = LoweringFixture::decode(cases::words(def.opcode));
            in.src[0] = missing ? Operand{} : Operand{OperandKind::Special, 251};
            LoweringFixture t;
            EXPECT_TRUE(lower_vop3_float_integer(t.b, t.rs, in, t.ok));
            EXPECT_FALSE(t.ok);
            EXPECT_FALSE(t.rs.vreg.contains(1));
        }
}
TEST(IrLowerVop3FloatInteger, UnknownFormatAndAdjacentConversionsStayOutsideTheFamily) {
    for (uint32_t op : {0x181u, 0x185u, 0x186u, 0x18au, 0x18cu, 0x1d2u}) {
        const auto in = LoweringFixture::decode(cases::words(op));
        LoweringFixture t;
        EXPECT_FALSE(lower_vop3_float_integer(t.b, t.rs, in, t.ok));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
    }
    auto in = LoweringFixture::decode(cases::words(0x187));
    in.fmt = Rdna2Format::VOP1;
    LoweringFixture t;
    EXPECT_FALSE(lower_vop3_float_integer(t.b, t.rs, in, t.ok));
    EXPECT_TRUE(t.ok);
}
TEST(IrLowerVop3FloatInteger, PlainFlrLookupReusesTheExactOldDefinitionOutsideBothRegistries) {
    const auto* old = find_vector_conversion_op(Rdna2Format::VOP1, 0x0d);
    ASSERT_NE(old, nullptr);
    const auto* alias = find_vector_cvt_flr_e64_op(Rdna2Format::VOP3, 0x18d);
    ASSERT_NE(alias, nullptr);
    EXPECT_EQ(alias->build, old->build);
    EXPECT_EQ(alias->fmt, Rdna2Format::VOP3);
    EXPECT_EQ(alias->opcode, 0x18du);
    EXPECT_EQ(find_vector_float_integer_e64_op(Rdna2Format::VOP3, 0x18d), nullptr);
    EXPECT_EQ(vector_float_integer_e64_defs().size(), 2u);
    EXPECT_EQ(vector_conversion_defs().size(), 13u);
    for (uint32_t op : {0x181u, 0x185u, 0x186u, 0x187u, 0x188u, 0x18cu, 0x1d2u})
        EXPECT_EQ(find_vector_cvt_flr_e64_op(Rdna2Format::VOP3, op), nullptr);
    EXPECT_EQ(find_vector_cvt_flr_e64_op(Rdna2Format::VOP1, 0x0d), nullptr);
    const auto& effects = rdna2_instruction_effects(Rdna2Format::VOP3, 0x18d);
    EXPECT_EQ(rdna2_valu_source_count(Rdna2Format::VOP3, 0x18d), 1u);
    EXPECT_EQ(effects.vdst_dwords, 1u);
    EXPECT_FALSE(effects.vdst_half || effects.accumulates || effects.implicit_reads ||
                 effects.implicit_writes);
    EXPECT_EQ(effects.lane_flow, Rdna2LaneFlow::LaneWise);
}
TEST(IrLowerVop3FloatInteger, PlainFlrRejectsEveryModifierBeforeReadingOrPublishing) {
    for (uint32_t word = 0; word < 2; ++word)
        for (uint32_t bit = word ? 9u : 8u; bit < (word ? 32u : 16u); ++bit) {
            std::vector<uint32_t> words{0xd58d0001, 0x102};
            words[word] |= 1u << bit;
            LoweringFixture t;
            const auto before = t.code_words();
            const auto in = LoweringFixture::decode(words);
            ASSERT_TRUE(lower_vop3_float_integer(t.b, t.rs, in, t.ok));
            EXPECT_FALSE(t.ok) << word << ':' << bit;
            EXPECT_FALSE(t.rs.vreg.contains(1));
            EXPECT_EQ(t.code_words(), before);
        }
}
TEST(IrLowerVop3FloatInteger, PlainFlrPublishesOnlyVdstAndKeepsUnsupportedSourcesTransactional) {
    for (uint32_t prefix : {0xd58d0001u, 0xd18d0001u}) {
        LoweringFixture t;
        t.set_sgpr(50, 0xbfc00000);
        auto in = LoweringFixture::decode({prefix, 50});
        const auto exec = t.rs.exec;
        ASSERT_TRUE(lower_vop3_float_integer(t.b, t.rs, in, t.ok));
        ASSERT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.contains(1));
        EXPECT_EQ(t.rs.sreg.size(), 1u);
        EXPECT_EQ(t.rs.exec, exec);
        for (uint32_t flag = 0; flag < 5; ++flag) {
            LoweringFixture bad;
            auto unsupported = in;
            if (flag == 0) unsupported.src[0] = {};
            if (flag == 1) unsupported.src[0] = {OperandKind::Special, 251};
            if (flag == 2) unsupported.has_modifier = true;
            if (flag == 3) unsupported.has_sdwa = true;
            if (flag == 4) unsupported.has_dpp = true;
            ASSERT_TRUE(lower_vop3_float_integer(bad.b, bad.rs, unsupported, bad.ok));
            EXPECT_FALSE(bad.ok);
            EXPECT_FALSE(bad.rs.vreg.contains(1));
        }
    }
}
