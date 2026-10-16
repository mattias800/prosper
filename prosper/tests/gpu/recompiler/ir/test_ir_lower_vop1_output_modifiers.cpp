// The relocated VOP1 numeric stage must reject integer modifiers and retain producing-mode
// suppression without moving the caller's DPP or EXEC-predicated destination write.
#include "gpu/recompiler/ir/lower/vop1_output_modifiers.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <gtest/gtest.h>
#include <cstdint>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
TEST(IrLowerVop1OutputModifiers, AnUnmodifiedResultDoesNotRequireLaunchMode) {
    LoweringFixture t;
    t.b.is_compute = false;
    t.b.is_vertex = true;
    t.b.vertex_float_mode = {};
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP1;
    in.opcode = 0x11;
    const auto original = t.b.uconst(0x437f0000);
    auto result = original;
    const auto before = t.code_words();
    lower_vop1_output_modifiers(t.b, in, result, t.ok);
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(result, original);
    EXPECT_EQ(t.code_words(), before);
}
TEST(IrLowerVop1OutputModifiers, IntegerResultModifiersRefuseBeforeEmission) {
    for (uint32_t opcode : {0x07u, 0x37u})
        for (uint32_t scale = 0; scale < 4; ++scale)
            for (bool clamp : {false, true}) {
                if (!scale && !clamp) continue;
                LoweringFixture t;
                Rdna2Inst in{};
                in.fmt = Rdna2Format::VOP1;
                in.opcode = opcode;
                in.omod = scale;
                in.clamp = clamp;
                const auto original = t.b.uconst(0x81234567);
                auto result = original;
                const auto before = t.code_words();
                lower_vop1_output_modifiers(t.b, in, result, t.ok);
                EXPECT_FALSE(t.ok) << opcode << " " << scale << " " << clamp;
                EXPECT_EQ(result, original);
                EXPECT_EQ(t.code_words(), before);
            }
}
TEST(IrLowerVop1OutputModifiers, ByteResultsRespectBothModeSuppressionConditions) {
    for (uint32_t rsrc1 : {0u, 0x30000u, 0x800000u}) {
        LoweringFixture t;
        t.b.compute_pgm_rsrc1 = rsrc1;
        Rdna2Inst in{};
        in.fmt = Rdna2Format::VOP1;
        in.opcode = 0x11;
        in.omod = 1;
        const auto original = t.b.uconst(0x437f0000);
        auto result = original;
        const auto before = t.code_words();
        lower_vop1_output_modifiers(t.b, in, result, t.ok);
        EXPECT_TRUE(t.ok);
        if (rsrc1) {
            EXPECT_EQ(result, original);
            EXPECT_EQ(t.code_words(), before);
        } else {
            EXPECT_NE(result, original);
            EXPECT_GT(t.code_words(), before);
        }
        EXPECT_TRUE(t.rs.vreg.empty());
    }
}
