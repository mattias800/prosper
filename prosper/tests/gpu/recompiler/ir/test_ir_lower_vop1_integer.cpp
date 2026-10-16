// Integer-only VOP1 lowering owns its plain cases and preserves legacy modifier refusals.
#include "gpu/recompiler/ir/lower/vop1_integer_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>
#include <cstdint>

using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t op) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP1;
    in.opcode = op;
    return in;
}
}   // namespace
TEST(IrLowerVop1Integer, NotEmitsExactlyOneInstruction) {
    LoweringFixture t;
    const auto source = t.b.uconst(5), old = t.b.uconst(7);
    auto d = old;
    const auto words = t.code_words();
    EXPECT_TRUE(ir::lower_vop1_integer(t.b, instruction(0x37), source, d, t.ok));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.code_words() - words, 4u);
    EXPECT_NE(d, old);
}
TEST(IrLowerVop1Integer, CopyRetainsTheSourceIdWithoutInstructions) {
    LoweringFixture t;
    const auto source = t.b.uconst(5);
    auto d = t.b.uconst(7);
    const auto words = t.code_words();
    EXPECT_TRUE(ir::lower_vop1_integer(t.b, instruction(0x01), source, d, t.ok));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(d, source);
    EXPECT_EQ(t.code_words(), words);
}
TEST(IrLowerVop1Integer, ScansStillRefuseSubwordAndDppForms) {
    for (auto op : {kVop1OpcodeFfbhU32, kVop1OpcodeFfblB32})
        for (bool dpp : {false, true}) {
            SCOPED_TRACE(::testing::Message()
                         << "opcode " << op << " DPP " << dpp << " SDWA " << !dpp);
            LoweringFixture t;
            auto in = instruction(op);
            in.has_dpp = dpp;
            in.has_sdwa = !dpp;
            const auto source = t.b.uconst(5);
            auto d = t.b.uconst(7);
            const auto words = t.code_words();
            EXPECT_TRUE(ir::lower_vop1_integer(t.b, in, source, d, t.ok))
                << "the integer scan route owns this form";
            EXPECT_FALSE(t.ok) << "scan modifiers remain unsupported";
            EXPECT_EQ(t.code_words(), words);
        }
}
TEST(IrLowerVop1Integer, ReservedLaneAndOtherFormatsFallBackUntouched) {
    for (auto in : {instruction(0x02), Rdna2Inst{}}) {
        LoweringFixture t;
        const auto old = t.b.uconst(7);
        auto d = old;
        const auto words = t.code_words();
        EXPECT_FALSE(ir::lower_vop1_integer(t.b, in, old, d, t.ok));
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(d, old);
        EXPECT_EQ(t.code_words(), words);
    }
}

TEST(IrLowerVop1Integer, BfrevSelectsItsSourceAtTheRelocatedNumericStage) {
    for (uint8_t selector = 0; selector <= 5; ++selector) {
        LoweringFixture t;
        auto in = instruction(0x38);
        in.has_sdwa = true;
        in.sdwa_src0_sel = selector;
        const auto original = t.b.uconst(0x81234567);
        auto a = original;
        const auto before = t.code_words();
        ir::select_vop1_integer_source(t.b, in, a);
        EXPECT_NE(a, original);
        EXPECT_GT(t.code_words(), before);
        EXPECT_TRUE(t.ok);
    }
}
TEST(IrLowerVop1Integer, SourceSelectionLeavesOtherOperationsAndDwordFormsUntouched) {
    for (uint32_t op : {0x01u, 0x02u, 0x37u, 0x38u, 0x39u, 0x3au})
        for (bool sdwa : {false, true})
            for (uint8_t selector : {0, 6}) {
                if (op == 0x38 && sdwa && selector == 0) continue;
                LoweringFixture t;
                auto in = instruction(op);
                in.has_sdwa = sdwa;
                in.sdwa_src0_sel = selector;
                const auto original = t.b.uconst(0x81234567);
                auto a = original;
                const auto before = t.code_words();
                ir::select_vop1_integer_source(t.b, in, a);
                EXPECT_EQ(a, original);
                EXPECT_EQ(t.code_words(), before);
                EXPECT_TRUE(t.ok);
            }
}
