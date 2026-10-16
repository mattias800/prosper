// Missing launch state is not IEEE=0; either independently known suppression condition suffices.
#include "gpu/recompiler/ir/lower/float_output_modifiers.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include <gtest/gtest.h>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint8_t scale = 1) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP3;
    in.opcode = 0x103;
    in.pc = 10;
    in.omod = scale;
    return in;
}
}   // namespace
TEST(IrLowerFloatOutputModifiers, ComputeOutputDenormBitBelongsToTheResultPrecision) {
    for (auto width : {FloatResultWidth::F32, FloatResultWidth::F16})
        for (uint32_t mode = 0; mode < 256; ++mode) {
            LoweringFixture t;
            t.b.compute_pgm_rsrc1 = mode << 12;
            const auto m = float_output_modifier(t.b, instruction(2), width, t.ok);
            ASSERT_TRUE(t.ok);
            EXPECT_EQ(m.scale, mode & (width == FloatResultWidth::F32 ? 0x20 : 0x80) ? 0 : 2);
            if (m.scale)
                EXPECT_EQ(m.rounding, (mode >> (width == FloatResultWidth::F32 ? 0 : 2)) & 3);
        }
}
TEST(IrLowerFloatOutputModifiers, IeeeSuppressesScalingIndependentlyOfDenormMode) {
    for (auto width : {FloatResultWidth::F32, FloatResultWidth::F16}) {
        LoweringFixture t;
        t.b.compute_pgm_rsrc1 = 1u << 23;
        EXPECT_EQ(float_output_modifier(t.b, instruction(), width, t.ok).scale, 0);
        EXPECT_TRUE(t.ok);
    }
}
TEST(IrLowerFloatOutputModifiers, FragmentKeepsMissingAndObservedZeroSeparate) {
    for (int state = 0; state < 5; ++state) {
        LoweringFixture t;
        t.b.is_compute = false;
        t.b.is_fragment = true;
        if (state == 1) t.b.fragment_float_mode = {true, 0x20};
        if (state == 2) t.b.fragment_float_flags = {true, true, false};
        if (state == 3) t.b.fragment_float_mode = {true, 0};
        if (state == 4) {
            t.b.fragment_float_mode = {true, 0};
            t.b.fragment_float_flags = {true, false, false};
        }
        auto m = float_output_modifier(t.b, instruction(), FloatResultWidth::F32, t.ok);
        EXPECT_EQ(t.ok, state == 1 || state == 2 || state == 4);
        EXPECT_EQ(m.scale, state == 4 ? 1 : 0);
        if (!t.ok) {
            EXPECT_EQ(t.b.stage_reject_pc, 10u);
            EXPECT_EQ(t.b.stage_reject_reason, "float-omod-launch-mode-unavailable");
        }
    }
}
TEST(IrLowerFloatOutputModifiers, AbsentComputeModeOnlyMattersForAnActiveModifier) {
    for (uint8_t scale : {0, 1}) {
        LoweringFixture t;
        t.b.compute_pgm_rsrc1 = UINT32_MAX;
        const auto before = t.code_words();
        EXPECT_EQ(float_output_modifier(t.b, instruction(scale), FloatResultWidth::F32, t.ok).scale,
                  0);
        EXPECT_EQ(t.ok, scale == 0);
        EXPECT_EQ(t.code_words(), before);
    }
}
TEST(IrLowerFloatOutputModifiers, PhysicalComputeShellDoesNotSupplyANggGuestFloatMode) {
    LoweringFixture t;
    t.b.ngg_workgroup_shell = true;
    t.b.compute_pgm_rsrc1 = kDefaultComputePgmRsrc1;
    EXPECT_EQ(float_output_modifier(t.b, instruction(), FloatResultWidth::F32, t.ok).scale, 0);
    EXPECT_FALSE(t.ok);
    EXPECT_EQ(t.b.stage_reject_reason, "float-omod-launch-mode-unavailable");
}
TEST(IrLowerFloatOutputModifiers, VertexAndNggReadTheProducingGraphicsCarrier) {
    for (bool ngg : {false, true})
        for (auto width : {FloatResultWidth::F32, FloatResultWidth::F16})
            for (uint32_t mode = 0; mode < 256; ++mode)
                for (bool ieee : {false, true}) {
                    LoweringFixture t;
                    t.b.is_compute = ngg;
                    t.b.ngg_workgroup_shell = ngg;
                    t.b.compute_pgm_rsrc1 = UINT32_MAX;
                    t.b.vertex_float_mode = {true, uint8_t(mode), ieee};
                    const auto got = float_output_modifier(t.b, instruction(), width, t.ok);
                    EXPECT_TRUE(t.ok);
                    EXPECT_EQ(got.scale,
                              (mode & (width == FloatResultWidth::F32 ? 0x20 : 0x80)) || ieee ? 0
                                                                                              : 1);
                    if (got.scale)
                        EXPECT_EQ(got.rounding,
                                  (mode >> (width == FloatResultWidth::F32 ? 0 : 2)) & 3);
                }
}
TEST(IrLowerFloatOutputModifiers, UnrelatedLaunchBitsDoNotSelectAnOutputScalePolicy) {
    for (bool extra_bit : {false, true}) {
        LoweringFixture t;
        t.b.compute_pgm_rsrc1 = uint32_t(extra_bit) << 26;
        const auto m = float_output_modifier(t.b, instruction(), FloatResultWidth::F16, t.ok);
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(m.scale, 1);
        EXPECT_EQ(m.rounding, 0);
    }
    LoweringFixture ps;
    ps.b.is_compute = false;
    ps.b.is_fragment = true;
    ps.b.fragment_float_mode = {true, 0};
    ps.b.fragment_float_flags = {true, false, false};
    EXPECT_EQ(float_output_modifier(ps.b, instruction(), FloatResultWidth::F16, ps.ok).scale, 1);
    EXPECT_TRUE(ps.ok);
}
