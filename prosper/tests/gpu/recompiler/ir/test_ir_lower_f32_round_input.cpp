// CEIL/FLOOR must use the producing F32 input-denorm bit, never IEEE/output bits or
// a physical NGG compute launch. Missing MODE must not publish a guessed result.
#include "gpu/recompiler/ir/lower/f32_round_input.hpp"
#include "gpu/recompiler/ir/lower/vop1_unary_float_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/ops/vector_float_unary.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <gtest/gtest.h>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t opcode = 0x22) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP1;
    in.opcode = opcode;
    in.pc = 19;
    return in;
}
size_t lower_words(LoweringFixture& t, const Rdna2Inst& in, uint32_t source) {
    const auto* def = find_vector_float_unary_op(in.fmt, in.opcode);
    EXPECT_NE(def, nullptr);
    const auto before = t.code_words();
    EXPECT_NE(emit_f32_rounding_op(t.b, in, *def, source, t.ok), 0u);
    return t.code_words() - before;
}
} // namespace
TEST(IrLowerF32RoundInput, EffectsOwnTheSensitiveInstructionFacts) {
    unsigned count = 0;
    for (const auto& row : rdna2_instruction_effects_rows()) {
        if (!row.effects.f32_rounding_input_sensitive) continue;
        ++count;
        EXPECT_TRUE(row.fmt == Rdna2Format::VOP1 || row.fmt == Rdna2Format::VOP3);
        EXPECT_STREQ(row.effects.name,
                     row.opcode == 0x22 || row.opcode == 0x1a2 ? "v_ceil_f32" : "v_floor_f32");
        EXPECT_EQ(row.effects.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(row.effects.vdst_dwords, 1u);
    }
    EXPECT_EQ(count, 4u);
    for (uint32_t opcode : {0x0bu, 0x0cu, 0x0du, 0x20u, 0x21u, 0x23u})
        EXPECT_FALSE(
            rdna2_instruction_effects(Rdna2Format::VOP1, opcode).f32_rounding_input_sensitive);
}
TEST(IrLowerF32RoundInput, EveryComputeModeUsesOnlyTheInputDenormBit) {
    for (uint32_t opcode : {0x22u, 0x24u})
        for (uint32_t mode = 0; mode < 256; ++mode)
            for (bool ieee : {false, true}) {
                LoweringFixture t;
                t.b.compute_pgm_rsrc1 = (mode << 12) | (uint32_t(ieee) << 23);
                const auto words = lower_words(t, instruction(opcode), t.b.uconst(1));
                EXPECT_TRUE(t.ok);
                EXPECT_EQ(words == 14u, bool(mode & 0x10));
            }
}
TEST(IrLowerF32RoundInput, GraphicsAndPhysicalNggShellUseTheirOwnCarrier) {
    for (uint32_t domain = 0; domain < 4; ++domain)
        for (uint8_t mode : {0u, 0x10u, 0x20u, 0x30u}) {
            LoweringFixture t;
            const uint8_t other = mode ^ 0x10;
            t.b.is_compute = domain == 1 || domain == 3;
            t.b.is_fragment = domain == 2;
            t.b.ngg_workgroup_shell = domain == 1;
            if (domain == 3) {
                t.b.guest_stage = GuestShaderStage::Fragment;
                t.b.physical_domain = PhysicalExecutionDomain::Workgroup;
                ASSERT_TRUE(t.b.is_fragment_packet());
            }
            t.b.compute_pgm_rsrc1 = uint32_t(other) << 12;
            t.b.fragment_float_mode = {true, domain >= 2 ? mode : other};
            t.b.vertex_float_mode = {true, domain < 2 ? mode : other, true};
            const auto words = lower_words(t, instruction(), t.b.uconst(1));
            EXPECT_TRUE(t.ok);
            EXPECT_EQ(words == 14u, bool(mode & 0x10)) << "domain " << domain;
        }
}
TEST(IrLowerF32RoundInput, MissingPacketModeCannotBorrowOtherCarriers) {
    LoweringFixture t;
    t.b.guest_stage = GuestShaderStage::Fragment;
    t.b.physical_domain = PhysicalExecutionDomain::Workgroup;
    t.b.is_fragment = false;
    t.b.is_compute = true;
    t.b.vertex_float_mode = {true, 0x10, true};
    t.b.compute_pgm_rsrc1 = 0x30000;
    const auto* def = find_vector_float_unary_op(Rdna2Format::VOP1, 0x22);
    EXPECT_EQ(emit_f32_rounding_op(t.b, instruction(), *def, t.b.uconst(1), t.ok), 0u);
    EXPECT_FALSE(t.ok);
    EXPECT_EQ(t.b.stage_reject_reason, "f32-round-input-mode-unavailable");
}
TEST(IrLowerF32RoundInput, MissingModeRejectsBeforeDestinationPublication) {
    for (uint32_t opcode : {0x22u, 0x24u}) {
        LoweringFixture t;
        t.b.compute_pgm_rsrc1 = UINT32_MAX;
        auto destination = t.b.uconst(0xd00d);
        const auto old = destination, source = t.b.id();
        const auto before = t.code_words();
        EXPECT_TRUE(lower_vop1_unary_float(t.b, instruction(opcode), source, destination, t.ok));
        EXPECT_FALSE(t.ok);
        EXPECT_EQ(destination, old);
        EXPECT_EQ(t.code_words(), before);
        EXPECT_EQ(t.b.stage_reject_pc, 19u);
        EXPECT_EQ(t.b.stage_reject_reason, "f32-round-input-mode-unavailable");
    }
}
TEST(IrLowerF32RoundInput, KnownModeIndependentSourcesNeedNoModeGuess) {
    for (uint32_t bits : {0u, 0x80000000u, 0x00800000u, 0x80800000u, 0x3fa00000u}) {
        LoweringFixture t;
        t.b.compute_pgm_rsrc1 = UINT32_MAX;
        EXPECT_EQ(lower_words(t, instruction(), t.b.uconst(bits)), 14u);
        EXPECT_TRUE(t.ok);
    }
    for (uint32_t bits : {1u, 0x80000001u, 0x007fffffu, 0x807fffffu}) {
        LoweringFixture t;
        t.b.compute_pgm_rsrc1 = UINT32_MAX;
        const auto* def = find_vector_float_unary_op(Rdna2Format::VOP1, 0x22);
        EXPECT_EQ(emit_f32_rounding_op(t.b, instruction(), *def, t.b.uconst(bits), t.ok), 0u);
        EXPECT_FALSE(t.ok);
    }
}
TEST(IrLowerF32RoundInput, UnrelatedOperationsKeepTheirExistingEmission) {
    for (uint32_t opcode : {0x20u, 0x21u, 0x23u}) {
        LoweringFixture t;
        t.b.compute_pgm_rsrc1 = UINT32_MAX;
        EXPECT_EQ(lower_words(t, instruction(opcode), t.b.uconst(1)), 14u);
        EXPECT_TRUE(t.ok);
    }
}
TEST(IrLowerF32RoundInput, APhysicalComputeModeDoesNotAuthenticateMissingGraphicsMode) {
    LoweringFixture t;
    t.b.ngg_workgroup_shell = true;
    const auto* def = find_vector_float_unary_op(Rdna2Format::VOP1, 0x22);
    EXPECT_EQ(emit_f32_rounding_op(t.b, instruction(), *def, t.b.uconst(1), t.ok), 0u);
    EXPECT_FALSE(t.ok);
}
TEST(IrLowerF32RoundInput, AnEarlierSourceRefusalKeepsItsDiagnosticAndDestination) {
    LoweringFixture t;
    t.b.compute_pgm_rsrc1 = UINT32_MAX;
    t.b.stage_reject_pc = 7;
    t.b.stage_reject_reason = "earlier-source-refusal";
    t.ok = false;
    auto destination = t.b.uconst(0xd00d);
    const auto old = destination, source = t.b.id();
    const auto before = t.code_words();
    EXPECT_TRUE(lower_vop1_unary_float(t.b, instruction(), source, destination, t.ok));
    EXPECT_FALSE(t.ok);
    EXPECT_EQ(destination, old);
    EXPECT_EQ(t.code_words(), before);
    EXPECT_EQ(t.b.stage_reject_pc, 7u);
    EXPECT_EQ(t.b.stage_reject_reason, "earlier-source-refusal");
}
