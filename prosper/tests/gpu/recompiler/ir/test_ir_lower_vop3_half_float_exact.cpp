// Bounded width conversion: integer bit answers are independent of the shared builder/eval.
#include "gpu/recompiler/ir/half_float_exact_e64_cases.hpp"
#include "gpu/recompiler/ir/lower/vop3_binary_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_half_float_exact_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/ops/vector_float_integer_e64.hpp"
#include "gpu/recompiler/ir/ops/vector_half_float_exact_e64.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <initializer_list>
#include <optional>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace cases = prosper::test::ir::half_float_exact;

TEST(IrLowerVop3HalfFloatExact, SeparateAliasReusesTheOldBuilderAndExistingEffects) {
    const auto* def = find_vector_half_float_exact_e64_op(Rdna2Format::VOP3, 0x18b);
    const auto* old = find_vector_conversion_op(Rdna2Format::VOP1, 0x0b);
    ASSERT_NE(def, nullptr);
    ASSERT_NE(old, nullptr);
    EXPECT_EQ(def->build, old->build);
    EXPECT_EQ(def->fmt, Rdna2Format::VOP3);
    EXPECT_EQ(def->opcode, 0x18bu);
    EXPECT_EQ(vector_conversion_defs().size(), 13u);
    EXPECT_EQ(vector_float_integer_e64_defs().size(), 2u);
    EXPECT_EQ(find_vector_conversion_op(Rdna2Format::VOP3, 0x18b), nullptr);
    EXPECT_EQ(find_vector_half_float_exact_e64_op(Rdna2Format::VOP1, 0x0b), nullptr);
    for (auto op : {0x181u, 0x185u, 0x186u, 0x18au, 0x18cu, 0x18du, 0x1d4u})
        EXPECT_EQ(find_vector_half_float_exact_e64_op(Rdna2Format::VOP3, op), nullptr);
    const auto& e = rdna2_instruction_effects(def->fmt, def->opcode);
    EXPECT_STREQ(e.name, "v_cvt_f32_f16");
    EXPECT_EQ(rdna2_valu_source_count(def->fmt, def->opcode), 1u);
    EXPECT_EQ(e.vdst_dwords, 1u);
    EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
    EXPECT_FALSE(e.vdst_half || e.accumulates || e.implicit_reads || e.implicit_writes);
}
TEST(IrLowerVop3HalfFloatExact, EveryNormalHalfMatchesTheIndependentBitEmbedding) {
    const auto* def = find_vector_half_float_exact_e64_op(Rdna2Format::VOP3, 0x18b);
    ASSERT_NE(def, nullptr);
    for (uint32_t sign : {0u, 0x8000u})
        for (uint32_t magnitude = 0x0400; magnitude <= 0x7bff; ++magnitude) {
            const auto raw = sign | magnitude;
            ASSERT_EQ(eval_conversion_op(*def, raw), cases::reference(raw)) << raw;
        }
    EXPECT_FALSE(eval_conversion_op(*def, std::nullopt));
}
TEST(IrLowerVop3HalfFloatExact, HandNormalBoundsAndSignsPublishOnlyTheFullDestination) {
    for (const auto& h : cases::kHand) {
        ASSERT_EQ(cases::reference(h.half), Known(h.answer));
        LoweringFixture t;
        const auto in = LoweringFixture::decode(cases::words(255, h.half));
        const BinaryFloatSource unused_source;
        const BinaryFloatResult unused_result;
        ASSERT_TRUE(lower_vop3_float_values_erased(t.b, t.rs, in, t.ok, false, unused_source,
                                                   unused_result));
        EXPECT_TRUE(t.ok);
        ASSERT_EQ(t.rs.vreg.size(), 1u);
        EXPECT_TRUE(t.rs.vreg.contains(1));
        EXPECT_TRUE(t.constant_cached(h.half));
    }
}
TEST(IrLowerVop3HalfFloatExact, InlineSourceUsesHalfWidthIncludingTheDocumentedTwoPiConstant) {
    for (uint32_t selector = 128; selector <= 248; ++selector) {
        LoweringFixture t;
        const auto before = t.b.code;
        const auto in = LoweringFixture::decode(cases::words(selector));
        const bool wanted = selector >= 240;
        EXPECT_EQ(lower_vop3_half_float_exact(t.b, t.rs, in, t.ok), wanted) << selector;
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(t.rs.vreg.contains(1), wanted);
        if (wanted)
            EXPECT_TRUE(t.constant_cached(cases::kInlineHalf.at(selector - 240)));
        else
            EXPECT_EQ(t.b.code, before);
    }
    EXPECT_EQ(cases::reference(0x3118), Known(0x3e230000));
    EXPECT_NE(cases::reference(0x3118), Known(0x3e22f983));
}
TEST(IrLowerVop3HalfFloatExact, ZeroDenormalNonfiniteAndNoncanonicalLiteralWordsStayUnmatched) {
    for (uint32_t raw :
         {0u, 0x8000u, 1u, 0x8001u, 0x03ffu, 0x83ffu, 0x7c00u, 0xfc00u, 0x7c01u, 0x7e00u, 0xfc01u,
          0xffffu, 0x13c00u, 0x3f800000u, 0x3e22f983u, UINT32_MAX}) {
        LoweringFixture t;
        const auto before = t.b.code;
        EXPECT_FALSE(lower_vop3_half_float_exact(
            t.b, t.rs, LoweringFixture::decode(cases::words(255, raw)), t.ok))
            << raw;
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.b.code, before);
    }
}
TEST(IrLowerVop3HalfFloatExact, KnownRegisterWordsAndForgedDecodedConstantsDoNotSupplySources) {
    for (uint32_t selector : {50u, 106u, 124u, 125u, 128u, 129u, 248u + 3u, 256u, 257u, 258u})
        for (bool forged : {false, true}) {
            LoweringFixture t;
            t.set_sgpr(50, 0x3c00);
            t.rs.vreg[2] = t.b.uconst(0x3c00);
            auto in = LoweringFixture::decode(cases::words(selector));
            if (forged) in.src[0] = {OperandKind::InlineFloat, 242};
            const auto before = t.b.code;
            EXPECT_FALSE(lower_vop3_half_float_exact(t.b, t.rs, in, t.ok));
            EXPECT_TRUE(t.ok);
            EXPECT_FALSE(t.rs.vreg.contains(1));
            EXPECT_EQ(t.b.code, before);
        }
}
TEST(IrLowerVop3HalfFloatExact, LiteralPresenceLengthAndInlineCoherenceCannotInventHalfBits) {
    for (uint32_t arm = 0; arm < 5; ++arm) {
        LoweringFixture t;
        auto in = LoweringFixture::decode(cases::words(255, 0x3c00));
        if (arm == 0) in.has_literal = false;
        if (arm == 1) in.len_dwords = 2;
        if (arm == 2) in = LoweringFixture::decode({0xd58b0001u, 255});
        if (arm == 3) {
            in = LoweringFixture::decode(cases::words(242));
            in.has_literal = true;
            in.len_dwords = 3;
        }
        if (arm == 4) in.src[0] = {OperandKind::InlineFloat, 242};
        const auto before = t.b.code;
        EXPECT_FALSE(lower_vop3_half_float_exact(t.b, t.rs, in, t.ok));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.b.code, before);
    }
}
TEST(IrLowerVop3HalfFloatExact, RawModifiersAndUnusedFieldsNeverPublish) {
    for (uint32_t word : {0u, 1u})
        for (uint32_t bit = word ? 9u : 8u; bit < (word ? 32u : 16u); ++bit) {
            auto raw = cases::words(242);
            raw[word] |= 1u << bit;
            raw.push_back(0x3c00);
            LoweringFixture t;
            const auto before = t.b.code;
            const bool claimed =
                lower_vop3_half_float_exact(t.b, t.rs, LoweringFixture::decode(raw), t.ok);
            EXPECT_EQ(t.ok, !claimed) << word << ':' << bit;
            EXPECT_TRUE(t.rs.vreg.empty());
            EXPECT_EQ(t.b.code, before);
        }
}
TEST(IrLowerVop3HalfFloatExact, DecodedFlagsAndOperandWidthsRefuseBeforeEmission) {
    for (uint32_t arm = 0; arm < 12; ++arm) {
        LoweringFixture t;
        auto in = LoweringFixture::decode(cases::words(242));
        if (arm == 0) in.has_modifier = true;
        if (arm == 1) in.has_sdwa = true;
        if (arm == 2) in.has_dpp = true;
        if (arm == 3) in.src_abs[1] = true;
        if (arm == 4) in.src_neg[2] = true;
        if (arm == 5) in.omod = 1;
        if (arm == 6) in.clamp = true;
        if (arm == 7) in.n_src = 2;
        if (arm == 8) in.dst.kind = OperandKind::SGPR;
        if (arm == 9) in.vop3p_opsel = 1;
        if (arm == 10) in.vop3p_opsel_hi = 1;
        if (arm == 11) in.vop3p_neg_hi = 1;
        const auto before = t.b.code;
        EXPECT_TRUE(lower_vop3_half_float_exact(t.b, t.rs, in, t.ok));
        EXPECT_FALSE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.b.code, before);
    }
}
TEST(IrLowerVop3HalfFloatExact, EarlierFailureAndCallerOwnedMaskStateRemainUnchanged) {
    for (bool prior_ok : {false, true}) {
        LoweringFixture t;
        t.rs.exec = t.b.bfalse();
        t.rs.exec_narrowed = true;
        t.rs.vcc = t.b.btrue();
        t.rs.scc = t.b.bfalse();
        const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
        t.b.stage_reject_pc = 71;
        t.b.stage_reject_reason = "earlier-source-rejection";
        t.ok = prior_ok;
        const auto before = t.b.code;
        EXPECT_TRUE(lower_vop3_half_float_exact(t.b, t.rs,
                                                LoweringFixture::decode(cases::words(242)), t.ok));
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
