// Exact constants avoid MODE-dependent integer-to-float rounding; no register proof is widened.
#include "gpu/recompiler/ir/exact_integer_float_e64_cases.hpp"
#include "gpu/recompiler/ir/lower/vop3_exact_integer_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_float_integer_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/ops/vector_exact_integer_float_e64.hpp"
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/ir/ops/vector_float_integer_e64.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <initializer_list>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace cases = prosper::test::ir::exact_integer_float;

TEST(IrLowerVop3ExactIntegerFloat, TwoAliasesReuseTheOldBuildersAndCentralEffects) {
    ASSERT_EQ(vector_exact_integer_float_e64_defs().size(), 2u);
    EXPECT_EQ(vector_conversion_defs().size(), 13u);
    EXPECT_EQ(vector_float_integer_e64_defs().size(), 2u);
    for (const auto& def : vector_exact_integer_float_e64_defs()) {
        const auto* old = find_vector_conversion_op(Rdna2Format::VOP1, def.opcode - 0x180);
        ASSERT_NE(old, nullptr);
        EXPECT_EQ(def.build, old->build);
        EXPECT_EQ(find_vector_conversion_op(def.fmt, def.opcode), nullptr);
        const auto& effects = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 1u);
        EXPECT_EQ(effects.vdst_dwords, 1u);
        EXPECT_EQ(effects.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_FALSE(effects.vdst_half || effects.accumulates || effects.implicit_reads ||
                     effects.implicit_writes);
    }
}
TEST(IrLowerVop3ExactIntegerFloat, LiteralHandTablePinsExactnessSignednessAndZeroRefusal) {
    for (const auto& def : vector_exact_integer_float_e64_defs())
        for (const auto& h : cases::kHand) {
            const auto want = def.opcode == 0x185 ? h.signed_bits : h.unsigned_bits;
            ASSERT_EQ(cases::reference_bits(h.raw, def.opcode == 0x185), want);
            LoweringFixture t;
            const auto before = t.b.code;
            const auto in = LoweringFixture::decode(cases::words(def.opcode, 255, h.raw));
            EXPECT_EQ(lower_vop3_float_integer(t.b, t.rs, in, t.ok), want.has_value())
                << def.opcode << ' ' << h.raw;
            EXPECT_TRUE(t.ok);
            EXPECT_EQ(t.rs.vreg.contains(1), want.has_value());
            if (want)
                EXPECT_EQ(eval_conversion_op(def, h.raw), want);
            else
                EXPECT_EQ(t.b.code, before);
        }
}
TEST(IrLowerVop3ExactIntegerFloat, EveryInlineIntegerSelectorHasAnExplicitDomain) {
    for (const auto& def : vector_exact_integer_float_e64_defs()) {
        uint32_t accepted = 0;
        for (uint32_t selector = 128; selector <= 208; ++selector) {
            LoweringFixture t;
            const auto in = LoweringFixture::decode(cases::words(def.opcode, selector));
            const bool want = selector != 128 && (def.opcode == 0x185 || selector <= 192);
            EXPECT_EQ(lower_vop3_float_integer(t.b, t.rs, in, t.ok), want) << selector;
            EXPECT_EQ(t.rs.vreg.contains(1), want);
            EXPECT_TRUE(t.ok);
            accepted += want;
        }
        EXPECT_EQ(accepted, def.opcode == 0x185 ? 80u : 64u);
    }
}
TEST(IrLowerVop3ExactIntegerFloat, BothDecodedVop3PrefixesHaveIdenticalValueGraphs) {
    for (const auto& def : vector_exact_integer_float_e64_defs())
        for (uint32_t selector : {129u, 255u}) {
            std::vector<uint32_t> reference;
            for (bool legacy : {false, true}) {
                LoweringFixture t;
                const auto in = LoweringFixture::decode(
                    cases::words(def.opcode, selector, 0x80000000, false, legacy));
                ASSERT_EQ(in.fmt, Rdna2Format::VOP3);
                ASSERT_EQ(in.opcode, def.opcode);
                ASSERT_TRUE(lower_vop3_float_integer(t.b, t.rs, in, t.ok));
                ASSERT_TRUE(t.ok);
                if (reference.empty()) reference = t.b.code;
                EXPECT_EQ(t.b.code, reference);
            }
        }
}
TEST(IrLowerVop3ExactIntegerFloat, KnownRegistersAndInlineFloatsRemainUnmatched) {
    for (const auto& def : vector_exact_integer_float_e64_defs())
        for (uint32_t selector : {0u, 50u, 106u, 124u, 125u, 126u, 242u, 256u, 258u, 511u}) {
            LoweringFixture t;
            t.set_sgpr(50, 1);
            t.rs.vreg[2] = t.b.uconst(1);
            const auto before = t.b.code;
            const auto in = LoweringFixture::decode(cases::words(def.opcode, selector));
            EXPECT_FALSE(lower_vop3_float_integer(t.b, t.rs, in, t.ok)) << selector;
            EXPECT_TRUE(t.ok);
            EXPECT_FALSE(t.rs.vreg.contains(1));
            EXPECT_EQ(t.b.code, before);
        }
}
TEST(IrLowerVop3ExactIntegerFloat, DecodedConstantCannotLaunderARegisterEncoding) {
    for (const auto& def : vector_exact_integer_float_e64_defs()) {
        LoweringFixture t;
        auto in = LoweringFixture::decode(cases::words(def.opcode, 258));
        in.src[0] = {OperandKind::InlineInt, 1};
        const auto before = t.b.code;
        EXPECT_FALSE(lower_vop3_float_integer(t.b, t.rs, in, t.ok));
        EXPECT_TRUE(t.ok);
        EXPECT_FALSE(t.rs.vreg.contains(1));
        EXPECT_EQ(t.b.code, before);
    }
}
TEST(IrLowerVop3ExactIntegerFloat, MissingLiteralAndMismatchedLengthCannotInventAWord) {
    for (const auto& def : vector_exact_integer_float_e64_defs())
        for (uint32_t arm = 0; arm < 4; ++arm) {
            LoweringFixture t;
            auto in = LoweringFixture::decode(cases::words(def.opcode, 255, 1));
            if (arm == 0) in.has_literal = false;
            if (arm == 1) in.len_dwords = 2;
            if (arm == 2) in = LoweringFixture::decode({0xd4000001u | (def.opcode << 16), 255});
            if (arm == 3) {
                in = LoweringFixture::decode(cases::words(def.opcode, 129));
                in.has_literal = true;
                in.literal = 1;
                in.len_dwords = 3;
            }
            const auto before = t.b.code;
            EXPECT_FALSE(lower_vop3_float_integer(t.b, t.rs, in, t.ok));
            EXPECT_TRUE(t.ok);
            EXPECT_FALSE(t.rs.vreg.contains(1));
            EXPECT_EQ(t.b.code, before);
        }
}
TEST(IrLowerVop3ExactIntegerFloat, RawUnusedFieldsAndEveryModifierRefuseWithoutPublication) {
    for (const auto& def : vector_exact_integer_float_e64_defs())
        for (uint32_t word : {0u, 1u})
            for (uint32_t bit = word ? 9u : 8u; bit < (word ? 32u : 16u); ++bit) {
                auto raw = cases::words(def.opcode, 129);
                raw[word] |= 1u << bit;
                raw.push_back(
                    1); // Supplies any raw-unused-field literal marker without truncation.
                LoweringFixture t;
                const auto before = t.b.code;
                const auto in = LoweringFixture::decode(raw);
                const bool claimed = lower_vop3_float_integer(t.b, t.rs, in, t.ok);
                EXPECT_EQ(t.ok, !claimed) << word << ':' << bit;
                EXPECT_FALSE(t.rs.vreg.contains(1));
                EXPECT_EQ(t.b.code, before);
            }
}
TEST(IrLowerVop3ExactIntegerFloat, EarlierFailureKeepsItsDiagnosticAndOldDestination) {
    for (const auto& def : vector_exact_integer_float_e64_defs()) {
        LoweringFixture t;
        const auto old = t.b.uconst(0x87654321);
        t.rs.vreg[1] = old;
        t.b.stage_reject_pc = 71;
        t.b.stage_reject_reason = "earlier-source-rejection";
        t.ok = false;
        const auto before = t.b.code;
        const auto in = LoweringFixture::decode(cases::words(def.opcode, 129));
        EXPECT_TRUE(lower_vop3_float_integer(t.b, t.rs, in, t.ok));
        EXPECT_FALSE(t.ok);
        EXPECT_EQ(t.rs.vreg.at(1), old);
        EXPECT_EQ(t.b.code, before);
        EXPECT_EQ(t.b.stage_reject_pc, 71u);
        EXPECT_EQ(t.b.stage_reject_reason, "earlier-source-rejection");
    }
}
TEST(IrLowerVop3ExactIntegerFloat, AllModeAndIeeeBytesLeaveTheExactGraphIdentical) {
    for (const auto& def : vector_exact_integer_float_e64_defs()) {
        std::vector<uint32_t> reference;
        for (uint32_t mode = 0; mode <= 256; ++mode)
            for (uint32_t ieee : {0u, 1u}) {
                LoweringFixture t;
                t.b.compute_pgm_rsrc1 = mode == 256 ? UINT32_MAX : (mode << 12) | (ieee << 23);
                const auto in = LoweringFixture::decode(cases::words(def.opcode, 255, 0x80000000));
                ASSERT_TRUE(lower_vop3_float_integer(t.b, t.rs, in, t.ok));
                ASSERT_TRUE(t.ok);
                if (reference.empty()) reference = t.b.code;
                EXPECT_EQ(t.b.code, reference) << mode << ' ' << ieee;
            }
    }
}
TEST(IrLowerVop3ExactIntegerFloat, PublicationLeavesExecAndScalarMasksWithTheCaller) {
    for (const auto& def : vector_exact_integer_float_e64_defs()) {
        LoweringFixture t;
        t.rs.exec = t.b.bfalse();
        t.rs.exec_narrowed = true;
        t.rs.vcc = t.b.btrue();
        t.rs.scc = t.b.bfalse();
        const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
        const auto in = LoweringFixture::decode(cases::words(def.opcode, 129));
        ASSERT_TRUE(lower_vop3_float_integer(t.b, t.rs, in, t.ok));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.contains(1));
        EXPECT_EQ(t.rs.exec, exec);
        EXPECT_EQ(t.rs.vcc, vcc);
        EXPECT_EQ(t.rs.scc, scc);
        EXPECT_TRUE(t.rs.exec_narrowed);
    }
}
