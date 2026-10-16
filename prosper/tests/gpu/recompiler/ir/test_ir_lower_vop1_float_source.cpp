// The relocated input stage stays before conversion and defers packed-half modifiers until
// unpacking. Real SDWA conversions pin its placement with independent, hand-worked answers.
#include "gpu/recompiler/ir/lower/vop1_float_source.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include <gtest/gtest.h>
#include <cstdint>
using namespace prosper::gpu;
using namespace prosper::test::ir;
TEST(IrLowerVop1FloatSource, UnmodifiedSourceKeepsItsSsaAndInstructionStream) {
    LoweringFixture t;
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP1;
    in.opcode = 0x07;
    auto source = t.b.uconst(0xbfe00000);
    const auto before = source;
    const auto words = t.code_words();
    ir::lower_vop1_float_source(t.b, in, source);
    EXPECT_EQ(source, before);
    EXPECT_EQ(t.code_words(), words);
}
TEST(IrLowerVop1FloatSource, PackedHalfModifiersWaitForTheHalfUnpack) {
    LoweringFixture t;
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP1;
    in.opcode = 0x0b;
    in.src_abs[0] = in.src_neg[0] = true;
    auto source = t.b.uconst(0x3c00bf00);
    const auto before = source;
    const auto words = t.code_words();
    ir::lower_vop1_float_source(t.b, in, source);
    EXPECT_EQ(source, before);
    EXPECT_EQ(t.code_words(), words);
}
TEST(IrLowerVop1FloatSource, SdwaFloatToIntegerAppliesAbsThenNegPerActiveLane) {
    for (uint32_t op : {0x07u, 0x08u})
        for (bool absolute : {false, true})
            for (bool negative : {false, true})
                for (uint32_t wave : {32u, 64u}) {
                    auto in = lane_inputs(411);
                    for (size_t lane = 0; lane < in.lanes(); ++lane)
                        in.s0[lane] = lane & 1 ? 0xbfe00000 : 0x3fe00000; // +/-1.75
                    const auto run = run_vector_op_per_lane(
                        {vop1(op, kVdst, 249), kVsrc0 | (6u << 8) | (6u << 16) |
                                                   (uint32_t(negative) << 20) |
                                                   (uint32_t(absolute) << 21)},
                        in, wave);
                    ASSERT_TRUE(run.error.empty()) << run.error;
                    ASSERT_EQ(run.d.size(), in.lanes());
                    for (size_t lane = 0; lane < in.lanes(); ++lane) {
                        const bool sign = ((lane & 1) && !absolute) != negative;
                        const uint32_t want = sign ? (op == 0x08 ? UINT32_MAX : 0) : 1;
                        EXPECT_EQ(run.d[lane], in.active[lane] ? want : in.old_d[lane]);
                    }
                }
}
