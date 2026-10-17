// #5225: half FMA must round the exact product+sum once. A F32 intermediate
// loses the below-midpoint residue in both independently hand-derived issue witnesses.
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "fixtures/half_fma_reference.hpp"
#include "fixtures/half_fma_vectors.hpp"
#include "gpu/recompiler/ir/ops/half_fma_finite.hpp"
#include "gpu/recompiler/ir/lower/half_fma_finite.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/lower/native_half_fma_lowering.hpp"
#include "fixtures/half_fma_finite.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::half_fma;
TEST(IrHalfFmaFinite, HandRationalWitnessesCarryBorrowTiesAndSubnormalAnswers) {
    // b838=-135/256, 0b96=971/2^22, 36dc=439/1024. Exact sum is
    // 460193779/2^30, 13/2^30 below the 36db/36dc midpoint: choose 36db.
    // 3c01*3e00+8001 = 3075/2048-2^-24, just below the 3e01/3e02 midpoint.
    EXPECT_EQ(eval_half_fma_finite(0xb838, 0x0b96, 0x36dc)[0], Known(0x36db));
    EXPECT_EQ(eval_half_fma_finite(0x3c01, 0x3e00, 0x8001)[0], Known(0x3e01));
    // 65504*(1025/1024)-65504 = 2047/32, exactly half 53ff.
    EXPECT_EQ(eval_half_fma_finite(0x7bff, 0x3c01, 0xfbff)[0], Known(0x53ff));
    for (const auto& v : kHalfFmaVectors) {
        const auto want = reference(v.a, v.b, v.c);
        ASSERT_TRUE(want.eligible);
        EXPECT_EQ(want.bits, v.expected);
        const auto got = eval_half_fma_finite(v.a, v.b, v.c);
        EXPECT_EQ(got[1], Known(1));
        EXPECT_EQ(got[0], Known(v.expected));
    }
}
TEST(IrHalfFmaFinite, IndependentBitSerialOracleCoversLimbAndHalfExponentBoundaries) {
    // Decode shifts near limb carries: product shifts 31/32 and addend shifts
    // 31/32. RTE retained-bit shifts 63/64, and magnitude top bits 33/34.
    const uint16_t values[] = {0,      1,      2,      0x03ff, 0x0400, 0x07ff, 0x0800, 0x1fff,
                               0x2000, 0x23ff, 0x2400, 0x3bff, 0x3c00, 0x3c01, 0x3fff, 0x4000,
                               0x43ff, 0x4400, 0x47ff, 0x4800, 0x7bfe, 0x7bff};
    for (uint16_t a : values)
        for (uint16_t b : values)
            for (uint16_t c : {uint16_t(1), uint16_t(0x83ff), uint16_t(0x3c00), uint16_t(0xfbff)}) {
                const auto want = reference(a, b, c);
                const auto got = eval_half_fma_finite(a, b, c);
                if (!got[1]) {
                    ADD_FAILURE() << "unknown eligibility";
                    return;
                }
                EXPECT_EQ(got[1], Known(want.eligible));
                if (want.eligible) EXPECT_EQ(got[0], Known(want.bits));
            }
}
TEST(IrHalfFmaFinite, EveryIntermediateIsDefinedEvenForRejectedRawInputs) {
    Block block;
    Builder ir(block);
    const auto a = ir.input(Type::U32), b = ir.input(Type::U32), c = ir.input(Type::U32);
    const auto result = build_half_fma_finite(ir, a, b, c);
    uint32_t seed = 5225;
    for (uint32_t k = 0; k < 4096; ++k) {
        std::array<Known, 3> inputs;
        for (auto& input : inputs) input = seed = seed * 1664525u + 1013904223u;
        if (k < 16)
            inputs = {k & 1 ? 0xffffffffu : 0u, k & 2 ? 0x7c00u : 0x8000u,
                      k & 4 ? 0x7e01u : 0x7bffu};
        const auto got = evaluate(block, inputs);
        for (size_t j = 0; j < got.size(); ++j)
            if (!got[j]) {
                ADD_FAILURE() << "unknown instruction " << j;
                return;
            }
        const auto want =
            reference(uint16_t(inputs[0].value_or(0)), uint16_t(inputs[1].value_or(0)),
                      uint16_t(inputs[2].value_or(0)));
        EXPECT_EQ(got[result.eligible.index], Known(want.eligible));
        if (want.eligible) EXPECT_EQ(got[result.bits.index], Known(want.bits));
    }
    RecordProperty("kernel_ir_instructions", block.size());
}
TEST(IrHalfFmaFinite, MissingInputsDoNotInventKnownCorrectedResults) {
    for (size_t missing = 0; missing < 3; ++missing) {
        std::array<Known, 3> in = {0x3c01, 0x3e00, 0x8001};
        in[missing] = std::nullopt;
        const auto got = eval_half_fma_finite(in[0], in[1], in[2]);
        EXPECT_FALSE(got[0]);
        EXPECT_FALSE(got[1]);
    }
}
TEST(IrHalfFmaFinite, ZeroOverflowAndNonfiniteOperandsSelectTheLegacyDomain) {
    const uint16_t cases[][3] = {
        {0, 0x3c00, 0},      {0x8000, 0x3c00, 0},     {0x3c00, 0x3c00, 0xbc00}, {1, 1, 0},
        {0x8001, 1, 0},      {0x7bff, 0x4000, 0},     {0x7bff, 0x3c00, 0x4c00}, {0x7c00, 0x3c00, 0},
        {0x7c00, 0, 0x3c00}, {0x7e01, 0, 0x3c00},     {0xfc00, 0, 0},           {0x7e01, 0x3c00, 0},
        {0x3c00, 0x7fff, 1}, {0x3c00, 0x3c00, 0x7c00}};
    for (const auto& v : cases) {
        const auto got = eval_half_fma_finite(v[0], v[1], v[2]);
        EXPECT_EQ(got[1], Known(0));
        // No exceptional/zero result bits are claimed when eligibility is false.
    }
}
TEST(IrHalfFmaFinite, OnlyTheActualProducingStageCanSupplyTheBoundedHalfMode) {
    SpirvCompute b;
    Rdna2Inst in{};
    b.is_compute = true;
    for (uint32_t mode = 0; mode < 256; ++mode) {
        b.compute_pgm_rsrc1 = mode << 12;
        EXPECT_EQ(use_half_fma_finite(b, in), (mode & 0xcc) == 0xc0);
    }
    b.compute_pgm_rsrc1 = UINT32_MAX;
    EXPECT_FALSE(use_half_fma_finite(b, in));
    b.compute_pgm_rsrc1 = 0xc0000;
    b.ngg_workgroup_shell = true;
    EXPECT_FALSE(use_half_fma_finite(b, in)) << "physical CS mode is not producing GS MODE";
    b.ngg_workgroup_shell = false;
    b.ngg_one_lane = true;
    EXPECT_FALSE(use_half_fma_finite(b, in)) << "one-lane wrapper CS MODE is not producing GS MODE";
    b.vertex_float_mode.available = true;
    b.vertex_float_mode.value = 0xc0;
    EXPECT_TRUE(use_half_fma_finite(b, in));
    b.guest_stage = GuestShaderStage::Fragment;
    b.physical_domain = PhysicalExecutionDomain::Workgroup;
    EXPECT_FALSE(use_half_fma_finite(b, in)) << "packet uses PS MODE, not physical CS or GS";
    b.fragment_float_mode.available = true;
    b.fragment_float_mode.value = 0xc0;
    EXPECT_TRUE(use_half_fma_finite(b, in));
    in.clamp = true;
    EXPECT_FALSE(use_half_fma_finite(b, in));
    in.clamp = false;
    for (uint8_t omod : {1, 2, 3}) {
        in.omod = omod;
        EXPECT_FALSE(use_half_fma_finite(b, in));
    }
}
TEST(IrHalfFmaFinite, NativeCallbacksAreReadOnceAndResultModificationPrecedesPublication) {
    for (uint32_t mode : {0u, 0xc0u}) {
        prosper::test::ir::LoweringFixture t;
        t.b.compute_pgm_rsrc1 = mode << 12;
        const auto in = t.decode(words(Kind::Native));
        const auto old = t.b.uconst(0x35553a11);
        t.rs.vreg[in.dst.value] = old;
        std::array<size_t, 3> reads{};
        size_t result_calls = 0;
        const NativeHalfFmaSource read = [&](const Operand& operand) {
            const uint32_t slot = operand.value == prosper::test::ir::kVsrc0   ? 0
                                  : operand.value == prosper::test::ir::kVsrc1 ? 1
                                                                               : 2;
            ++reads[slot];
            const uint32_t raw[] = {0x3c01, 0x3e00, 0x8001};
            return t.b.uconst(raw[slot]);
        };
        const NativeHalfFmaResult result = [&](uint32_t value) {
            ++result_calls;
            EXPECT_EQ(reads, (std::array<size_t, 3>{1, 1, 1}));
            EXPECT_EQ(t.rs.vreg.at(in.dst.value), old);
            return value;
        };
        lower_native_half_fma(t.b, t.rs, in, t.ok, false, old, read, result);
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(reads, (std::array<size_t, 3>{1, 1, 1}));
        EXPECT_EQ(result_calls, 1u);
        EXPECT_NE(t.rs.vreg.at(in.dst.value), old);
    }
}
TEST(IrHalfFmaFinite, FailedResultCallbackDoesNotEmitTheExactKernel) {
    std::vector<uint32_t> old_code;
    for (uint32_t mode : {0u, 0xc0u}) {
        prosper::test::ir::LoweringFixture t;
        t.b.compute_pgm_rsrc1 = mode << 12;
        const auto in = t.decode(words(Kind::Native));
        const auto old = t.b.uconst(0x35553a11);
        const NativeHalfFmaSource read = [&](const Operand&) { return t.b.uconst(0x3c00); };
        const NativeHalfFmaResult result = [&](uint32_t value) {
            t.ok = false;
            return value;
        };
        lower_native_half_fma(t.b, t.rs, in, t.ok, false, old, read, result);
        EXPECT_FALSE(t.ok);
        if (!mode)
            old_code = t.b.code;
        else
            EXPECT_EQ(t.b.code, old_code);
    }
}
