// The exact half-step domain avoids all FP rounding/FTZ/saturation guesses, not just a fixture.
#include "gpu/recompiler/ir/cvt_rpi_exact_e64_cases.hpp"
#include "gpu/recompiler/ir/lower/vop3_cvt_rpi_exact_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_float_integer_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/ops/vector_cvt_rpi_exact_e64.hpp"
#include "gpu/recompiler/ir/ops/vector_float_integer_e64.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <gtest/gtest.h>
#include <initializer_list>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace cases = prosper::test::ir::cvt_rpi_exact;

TEST(IrLowerVop3CvtRpiExact, SeparateAliasSharesTheOldBuilderWithoutChangingEitherRegistry) {
    const auto* def = find_vector_cvt_rpi_exact_e64_op(Rdna2Format::VOP3, 0x18c);
    const auto* old = find_vector_conversion_op(Rdna2Format::VOP1, 0x0c);
    ASSERT_NE(def, nullptr);
    ASSERT_NE(old, nullptr);
    EXPECT_EQ(def->build, old->build);
    EXPECT_EQ(def->fmt, Rdna2Format::VOP3);
    EXPECT_EQ(def->opcode, 0x18cu);
    EXPECT_EQ(vector_conversion_defs().size(), 13u);
    EXPECT_EQ(vector_float_integer_e64_defs().size(), 2u);
    EXPECT_EQ(find_vector_conversion_op(Rdna2Format::VOP3, 0x18c), nullptr);
    EXPECT_EQ(find_vector_cvt_flr_e64_op(Rdna2Format::VOP3, 0x18c), nullptr);
    EXPECT_EQ(find_vector_cvt_rpi_exact_e64_op(Rdna2Format::VOP1, 0x0c), nullptr);
    for (uint32_t op : {0x181u, 0x18bu, 0x18du, 0x187u, 0x188u})
        EXPECT_EQ(find_vector_cvt_rpi_exact_e64_op(Rdna2Format::VOP3, op), nullptr);
    const auto& effects = rdna2_instruction_effects(def->fmt, def->opcode);
    EXPECT_STREQ(effects.name, "v_cvt_rpi_i32_f32");
    EXPECT_EQ(rdna2_valu_source_count(def->fmt, def->opcode), 1u);
    EXPECT_EQ(effects.vdst_dwords, 1u);
    EXPECT_FALSE(effects.vdst_half || effects.accumulates || effects.implicit_reads ||
                 effects.implicit_writes);
}
TEST(IrLowerVop3CvtRpiExact, HandHalfTiesAndFiniteBoundsPublishOnlyTheIntegerResult) {
    const auto* def = find_vector_cvt_rpi_exact_e64_op(Rdna2Format::VOP3, 0x18c);
    ASSERT_NE(def, nullptr);
    for (const auto& h : cases::kHand) {
        ASSERT_EQ(cases::reference(h.bits), Known(h.answer));
        LoweringFixture t;
        auto in = LoweringFixture::decode(cases::words(255, h.bits));
        ASSERT_TRUE(lower_vop3_float_integer(t.b, t.rs, in, t.ok));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.contains(1));
        EXPECT_EQ(eval_conversion_op(*def, h.bits), Known(h.answer));
    }
}
TEST(IrLowerVop3CvtRpiExact, OnlyEightArchitecturalInlineFloatsMatchTheDomain) {
    for (uint32_t selector = 128; selector <= 248; ++selector) {
        LoweringFixture t;
        const auto before = t.b.code;
        const auto in = LoweringFixture::decode(cases::words(selector));
        const bool wanted = selector >= 240 && selector <= 247;
        EXPECT_EQ(lower_vop3_cvt_rpi_exact(t.b, t.rs, in, t.ok), wanted) << selector;
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(t.rs.vreg.contains(1), wanted);
        if (!wanted) EXPECT_EQ(t.b.code, before);
    }
}
TEST(IrLowerVop3CvtRpiExact, NonHalfStepNonfiniteSubnormalAndSaturationNeighborsStayUnmatched) {
    for (const uint32_t bits :
         {1u, 0x80000001u, 0x007fffffu, 0x807fffffu, 0x00800000u, 0x3e800000u, 0x3effffffu,
          0x3f000001u, 0xbf000001u, 0x4b000000u, 0xcb000000u, 0x4f000000u, 0xcf000000u, 0x7f800000u,
          0xff800000u, 0x7fc00001u, 0xff800001u}) {
        LoweringFixture t;
        const auto before = t.b.code;
        const auto in = LoweringFixture::decode(cases::words(255, bits));
        EXPECT_FALSE(lower_vop3_cvt_rpi_exact(t.b, t.rs, in, t.ok)) << bits;
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.b.code, before);
    }
}
TEST(IrLowerVop3CvtRpiExact, RegisterConstantsAndDecodedImmediateCannotBypassRawSourceClass) {
    for (uint32_t selector : {50u, 106u, 124u, 125u, 256u, 257u, 258u})
        for (bool forged : {false, true}) {
            LoweringFixture t;
            t.set_sgpr(50, 0x3fc00000);
            t.rs.vreg[2] = t.b.uconst(0x3fc00000);
            auto in = LoweringFixture::decode(cases::words(selector));
            if (forged) in.src[0] = {OperandKind::InlineFloat, 240};
            const auto before = t.b.code;
            EXPECT_FALSE(lower_vop3_cvt_rpi_exact(t.b, t.rs, in, t.ok));
            EXPECT_TRUE(t.ok);
            EXPECT_FALSE(t.rs.vreg.contains(1));
            EXPECT_EQ(t.b.code, before);
        }
}
TEST(IrLowerVop3CvtRpiExact, LiteralPresenceLengthAndDecodedFlagsNeverInventAValue) {
    for (uint32_t arm = 0; arm < 4; ++arm) {
        LoweringFixture t;
        auto in = LoweringFixture::decode(cases::words(255, 0x3fc00000));
        if (arm == 0) in.has_literal = false;
        if (arm == 1) in.len_dwords = 2;
        if (arm == 2) in = LoweringFixture::decode({0xd58c0001u, 255});
        if (arm == 3) {
            in = LoweringFixture::decode(cases::words(240));
            in.has_literal = true;
            in.len_dwords = 3;
        }
        const auto before = t.b.code;
        EXPECT_FALSE(lower_vop3_cvt_rpi_exact(t.b, t.rs, in, t.ok));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.b.code, before);
    }
}
TEST(IrLowerVop3CvtRpiExact, EveryRawModifierAndUnusedBitRefusesWithoutPublication) {
    for (uint32_t word : {0u, 1u})
        for (uint32_t bit = word ? 9u : 8u; bit < (word ? 32u : 16u); ++bit) {
            auto raw = cases::words(240);
            raw[word] |= 1u << bit;
            raw.push_back(0x3fc00000);
            LoweringFixture t;
            const auto before = t.b.code;
            const auto in = LoweringFixture::decode(raw);
            const bool claimed = lower_vop3_cvt_rpi_exact(t.b, t.rs, in, t.ok);
            EXPECT_EQ(t.ok, !claimed) << word << ':' << bit;
            EXPECT_TRUE(t.rs.vreg.empty());
            EXPECT_EQ(t.b.code, before);
        }
}
TEST(IrLowerVop3CvtRpiExact, EarlierFailureAndCallerMaskStateStayUnchanged) {
    for (bool prior_ok : {false, true}) {
        LoweringFixture t;
        t.rs.exec = t.b.bfalse();
        t.rs.exec_narrowed = true;
        t.rs.vcc = t.b.btrue();
        t.rs.scc = t.b.bfalse();
        const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
        t.b.stage_reject_pc = 71;
        t.b.stage_reject_reason = "earlier-source-rejection";
        const auto before = t.b.code;
        t.ok = prior_ok;
        const auto in = LoweringFixture::decode(cases::words(240));
        EXPECT_TRUE(lower_vop3_cvt_rpi_exact(t.b, t.rs, in, t.ok));
        EXPECT_EQ(t.ok, prior_ok);
        EXPECT_EQ(t.rs.vreg.contains(1), prior_ok);
        EXPECT_EQ(t.rs.exec, exec);
        EXPECT_EQ(t.rs.vcc, vcc);
        EXPECT_EQ(t.rs.scc, scc);
        EXPECT_TRUE(t.rs.exec_narrowed);
        EXPECT_EQ(t.b.stage_reject_pc, 71u);
        EXPECT_EQ(t.b.stage_reject_reason, "earlier-source-rejection");
        if (!prior_ok) EXPECT_EQ(t.b.code, before);
    }
}
TEST(IrLowerVop3CvtRpiExact, DecodedModifierAndWidthFlagsRefuseBeforeReadingOrPublishing) {
    for (uint32_t arm = 0; arm < 9; ++arm) {
        LoweringFixture t;
        auto in = LoweringFixture::decode(cases::words(240));
        if (arm == 0) in.has_modifier = true;
        if (arm == 1) in.has_sdwa = true;
        if (arm == 2) in.has_dpp = true;
        if (arm == 3) in.src_abs[1] = true;
        if (arm == 4) in.src_neg[2] = true;
        if (arm == 5) in.omod = 1;
        if (arm == 6) in.clamp = true;
        if (arm == 7) in.n_src = 2;
        if (arm == 8) in.dst.kind = OperandKind::SGPR;
        const auto before = t.b.code;
        EXPECT_TRUE(lower_vop3_cvt_rpi_exact(t.b, t.rs, in, t.ok));
        EXPECT_FALSE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.b.code, before);
    }
}
