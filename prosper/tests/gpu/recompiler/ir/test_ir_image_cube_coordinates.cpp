// Hand-worked coordinates pin the renderer's existing CUBE transform; the raw definition's
// actual emitted SPIR-V also runs per lane against its evaluation. This observes every
// intermediate, including vf/layer, independently of texture filtering and result writes.
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/ops/image_cube_coordinates.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <cstddef>
#include <optional>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
namespace {
uint32_t f(float value) {
    return std::bit_cast<uint32_t>(value);
}

TEST(IrImageCubeCoordinates, HandWorkedAxisClampsEvenFaceTiesAndStackedValues) {
    using A = std::array<Known, 4>;
    EXPECT_EQ(eval_image_cube_coordinates({f(1.25f), f(2.5f), f(2.5f)}),
              (A{f(0.25f), f(1.0f), f(2.0f), f(0.5f)}));
    EXPECT_EQ(eval_image_cube_coordinates({f(-0.5f), f(-0.25f), f(-0.5f)}),
              (A{f(-1.5f), f(0.0f), f(0.0f), f(0.0f)}));
    EXPECT_EQ(eval_image_cube_coordinates({f(2.5f), f(1.5f), f(4.5f)}),
              (A{f(1.5f), f(0.5f), f(4.0f), f(0.75f)}));
    EXPECT_EQ(eval_image_cube_coordinates({f(1.75f), f(1.25f), f(8.0f)}),
              (A{f(0.75f), f(0.25f), f(5.0f), f(0.875f)}));
    // RoundEven(1.5)=2 and RoundEven(3.5)=4, unlike truncation or rounding away from zero.
    EXPECT_EQ(eval_image_cube_coordinates({f(1.5f), f(1.5f), f(1.5f)})[2], Known{f(2.0f)});
    EXPECT_EQ(eval_image_cube_coordinates({f(1.5f), f(1.5f), f(3.5f)})[2], Known{f(4.0f)});
}

TEST(IrImageCubeCoordinates, UnknownSourcesAffectOnlyTheirDependentValues) {
    const auto x = eval_image_cube_coordinates({std::nullopt, f(1.5f), f(4.5f)});
    EXPECT_FALSE(x[0]);
    EXPECT_EQ(x[1], Known{f(0.5f)});
    EXPECT_EQ(x[2], Known{f(4.0f)});
    EXPECT_EQ(x[3], Known{f(0.75f)});
    const auto y = eval_image_cube_coordinates({f(1.25f), std::nullopt, f(4.5f)});
    EXPECT_EQ(y[0], Known{f(0.25f)});
    EXPECT_FALSE(y[1]);
    EXPECT_EQ(y[2], Known{f(4.0f)});
    EXPECT_FALSE(y[3]);
    const auto face = eval_image_cube_coordinates({f(1.25f), f(1.5f), std::nullopt});
    EXPECT_EQ(face[0], Known{f(0.25f)});
    EXPECT_EQ(face[1], Known{f(0.5f)});
    EXPECT_FALSE(face[2]);
    EXPECT_FALSE(face[3]);
}

TEST(IrImageCubeCoordinates, MinMaxEvaluationKeepsUnspecifiedDomainsUnknown) {
    // GLSL.std.450 FMin/FMax choose an implementation-dependent operand for a NaN. The
    // producing profile also does not authenticate signed-zero or infinite transport here.
    for (Opcode op : {Opcode::F32Min, Opcode::F32Max}) {
        for (uint32_t special : {0x7fc01234u, 0x7f801234u, 0x7f800000u, 0xff800000u}) {
            EXPECT_FALSE(evaluate_op(op, 0, special, f(1.0f), 0));
            EXPECT_FALSE(evaluate_op(op, 0, f(1.0f), special, 0));
        }
        for (uint32_t left : {0u, 0x80000000u})
            for (uint32_t right : {0u, 0x80000000u})
                EXPECT_FALSE(evaluate_op(op, 0, left, right, 0));
    }
    EXPECT_EQ(evaluate_op(Opcode::F32Min, 0, f(-1.0f), f(2.0f), 0), Known{f(-1.0f)});
    EXPECT_EQ(evaluate_op(Opcode::F32Max, 0, f(-1.0f), f(2.0f), 0), Known{f(2.0f)});
}

TEST(IrImageCubeCoordinates, ReadsThreeCoordinatesBeforeAllocatingConstants) {
    for (uint32_t first : {0u, 1u}) {
        const auto built = build_image_cube_coordinates(first);
        EXPECT_EQ(built.slots, (std::array<uint32_t, 3>{first, first + 1u, first + 2u}));
        ASSERT_GE(built.block.size(), 5u);
        for (size_t i = 0; i < 3u; ++i) EXPECT_EQ(built.block.insts()[i].op, Opcode::Input);
        EXPECT_EQ(built.block.insts()[3].op, Opcode::ConstU32);
        EXPECT_EQ(built.block.insts()[3].imm, f(1.0f));
        EXPECT_EQ(built.block.insts()[4].op, Opcode::ConstU32);
        EXPECT_EQ(built.block.insts()[4].imm, f(0.0f));
    }
}

TEST(IrImageCubeCoordinates, EveryEmittedIntermediateMatchesEvaluationPerLane) {
    constexpr uint32_t n = 64;
    const auto built = build_image_cube_coordinates();
    SpirvCompute b;
    b.diagnostic = {RecompileDiagnosticStage::Compute, 0x5d5428c000ull};
    b.begin(3u, nullptr, n, 1u, 1u, n, 0u, true, true);
    const std::array<uint32_t, 3> inputs{b.load_input(0), b.load_input(1), b.load_input(2)};
    const auto emitted = emit_spirv(b, built.block, inputs);
    for (uint32_t output = 0; output < 4u; ++output)
        b.store_output_word(emitted[built.values[output].index], 4u, output);
    std::vector<uint32_t> words;
    words.reserve(std::size_t{3} * n);
    std::vector<std::array<Known, 4>> expected;
    expected.reserve(n);
    for (uint32_t lane = 0; lane < n; ++lane) {
        // Interior face values avoid an unknown zero tie; y also spans both clamp branches.
        const std::array<uint32_t, 3> source{f(0.125f * static_cast<float>(lane) - 2.0f),
                                             f(0.125f + 0.25f * static_cast<float>(lane % 12u)),
                                             f(1.25f + static_cast<float>(lane % 4u))};
        words.insert(words.end(), source.begin(), source.end());
        expected.push_back(eval_image_cube_coordinates({source[0], source[1], source[2]}));
    }
    bpermute_oracle::Interpreter vm(b.finish());
    const auto got =
        vm.run_buffers(n, {{0u, words}, {1u, std::vector<uint32_t>(std::size_t{4} * n)}}, 1u);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(got.size(), std::size_t{4} * n);
    for (uint32_t lane = 0; lane < n; ++lane)
        for (uint32_t output = 0; output < 4u; ++output) {
            const auto want = expected[lane][output];
            if (!want) {
                ADD_FAILURE() << "unknown lane=" << lane << " output=" << output;
                continue;
            }
            EXPECT_EQ(got[std::size_t{4} * lane + output], *want)
                << "lane=" << lane << " output=" << output;
        }
}
}   // namespace
