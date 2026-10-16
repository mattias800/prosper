// Numeric stage contracts, independently of the caller's source proofs and final EXEC write.
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
