// D16 evaluation must agree with emitted words where known. Unknown words are still
// checked half by half against independent adjacent bounds, never skipped as unknown (#5420).
#include "gpu/recompiler/ir/image_d16_cases.hpp"
#include "gpu/recompiler/ir/ops/image_d16.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstddef>
#include <ios>
#include <optional>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
namespace cases = prosper::test::ir::image_d16;
namespace {
constexpr std::array kForms{ImageD16Form::SelectedLow, ImageD16Form::SelectedPair,
                            ImageD16Form::GatherPair};
TEST(IrImageD16Values, OnlyHostGuaranteedExactConversionsAreKnown) {
    for (const auto& x : cases::kExamples) {
        // Exact subnormal bits remain mandatory in the execution oracle, but the module has no
        // denormal-preservation profile that would make them a portable evaluator guarantee.
        const bool known = x.f16 == x.other && ((x.f16 & 0x7fffu) == 0 || (x.f16 & 0x7c00u) != 0);
        const auto low = eval_image_d16_word(ImageD16Form::SelectedLow, x.f32);
        if (known)
            EXPECT_EQ(low, Known{x.f16}) << std::hex << x.f32;
        else
            EXPECT_FALSE(low) << std::hex << x.f32;
        for (const auto form : {ImageD16Form::SelectedPair, ImageD16Form::GatherPair}) {
            const auto word = eval_image_d16_word(form, x.f32, 0x3f800000u);
            if (known)
                EXPECT_EQ(word, Known{x.f16 | 0x3c000000u});
            else
                EXPECT_FALSE(word);
            EXPECT_FALSE(eval_image_d16_word(form, 0x3f800000u, 0x3f803001u));
        }
    }
    for (uint32_t inexact : {0x3f803001u, 0xbf803001u, 0x00000001u, 0x47800000u})
        EXPECT_FALSE(eval_image_d16_word(ImageD16Form::SelectedLow, inexact));
    EXPECT_FALSE(eval_image_d16_word(ImageD16Form::SelectedLow, 0x7fc12345u));
    EXPECT_FALSE(eval_image_d16_word(ImageD16Form::SelectedLow, std::nullopt));
    // The unrelated scalar/VALU/buffer-store primitive keeps its existing RTE evaluation.
    EXPECT_EQ(evaluate_op(Opcode::PackHalfLo, 0, 0x3f803000u, 0, 0), Known{0x3c02u});
    EXPECT_EQ(evaluate_op(Opcode::PackHalfLo, 0, 0x3f803001u, 0, 0), Known{0x3c02u});
    EXPECT_EQ(evaluate_op(Opcode::PackHalfLo, 0, 0x33800000u, 0, 0), Known{0x0001u});
    EXPECT_EQ(evaluate_op(Opcode::PackHalfLo, 0, 0xb3800000u, 0, 0), Known{0x8001u});
}
std::vector<uint32_t> run_block(ImageD16Form form, const std::vector<uint32_t>& inputs) {
    constexpr uint32_t lanes = 64;
    const auto built = build_image_d16_word(form);
    SpirvCompute b;
    b.begin(2, nullptr, lanes, 1, 1, 64, 0, true, true);
    const std::array ids{b.load_input(0), b.load_input(1)};
    const auto emitted = emit_spirv(b, built.block, ids);
    b.store_output_word(emitted[built.word.index], 1, 0);
    bpermute_oracle::Interpreter vm(b.finish());
    vm.half_rte_preserve_denorm = true;   // one allowed PackHalf2x16 profile, not a device promise
    const auto result =
        vm.run_buffers(lanes, {{0, inputs}, {1, std::vector<uint32_t>(lanes, 0xdeadu)}}, 1);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    return result;
}
TEST(IrImageD16Values, EveryFormMatchesEvaluationAndIndependentHalfBoundsPerLane) {
    std::vector<uint32_t> inputs;
    inputs.reserve(128);
    for (size_t lane = 0; lane < 64; ++lane) {
        auto low = cases::kExamples[lane % cases::kExamples.size()];
        auto high = cases::kExamples[(lane * 3 + 5) % cases::kExamples.size()];
        // The last lanes also exercise the just-past-tie inputs with the same hand-worked bounds.
        if (lane >= 32) {
            if (low.f32 == 0x3f803000u || low.f32 == 0xbf803000u) ++low.f32;
            if (high.f32 == 0x3f803000u || high.f32 == 0xbf803000u) ++high.f32;
        }
        inputs.push_back(low.f32);
        inputs.push_back(high.f32);
    }
    for (const auto form : kForms) {
        const auto result = run_block(form, inputs);
        ASSERT_EQ(result.size(), 64u);
        uint32_t known = 0, unknown = 0;
        for (size_t lane = 0; lane < result.size(); ++lane) {
            auto low = cases::kExamples[lane % cases::kExamples.size()];
            auto high = form == ImageD16Form::SelectedLow
                            ? cases::Example{0, 0}
                            : cases::kExamples[(lane * 3 + 5) % cases::kExamples.size()];
            low.f32 = inputs[lane * 2];
            if (form != ImageD16Form::SelectedLow) high.f32 = inputs[lane * 2 + 1];
            const auto evaluated = eval_image_d16_word(form, low.f32, high.f32);
            if (evaluated) {
                ++known;
                EXPECT_EQ(result[lane], *evaluated);
            } else
                ++unknown;
            const std::array expected{low, high};
            for (uint32_t part = 0; part < 2; ++part) {
                const auto& x = expected[part];
                const uint32_t got = (result[lane] >> (part * 16)) & 0xffffu;
                EXPECT_TRUE(got == x.f16 || got == x.other)
                    << "form=" << int(form) << " lane=" << lane << " half=" << part
                    << " inexact=" << (x.f16 != x.other) << std::hex << " input=" << x.f32
                    << " got=" << got << " neighbours=" << x.f16 << "," << x.other;
            }
        }
        EXPECT_GT(known, 0u);
        EXPECT_GT(unknown, 0u);
    }
}
TEST(IrImageD16Values, HalfSubnormalsStayUnknownWithoutPreservationButExactBitsAreChecked) {
    struct Example {
        uint32_t f32, half;
        bool known;
    };
    // Hand-worked min/max subnormals of both signs, then the adjacent normal boundary.
    constexpr std::array examples{
        Example{0x33800000u, 0x0001u, false}, Example{0xb3800000u, 0x8001u, false},
        Example{0x387fc000u, 0x03ffu, false}, Example{0xb87fc000u, 0x83ffu, false},
        Example{0x38800000u, 0x0400u, true},  Example{0xb8800000u, 0x8400u, true}};
    std::vector<uint32_t> inputs;
    inputs.reserve(128);
    for (size_t lane = 0; lane < 64; ++lane) {
        inputs.push_back(examples[lane % examples.size()].f32);
        inputs.push_back(examples[(lane + 1) % examples.size()].f32);
    }
    for (const auto form : kForms) {
        const auto words = run_block(form, inputs);
        ASSERT_EQ(words.size(), 64u);
        for (size_t lane = 0; lane < words.size(); ++lane) {
            const auto low = examples[lane % examples.size()];
            const auto high = form == ImageD16Form::SelectedLow
                                  ? Example{0, 0, true}
                                  : examples[(lane + 1) % examples.size()];
            const uint32_t expected = low.half | (high.half << 16);
            const auto evaluated = eval_image_d16_word(form, low.f32, high.f32);
            if (low.known && high.known)
                EXPECT_EQ(evaluated, Known{expected});
            else
                EXPECT_FALSE(evaluated) << "form=" << int(form) << " lane=" << lane;
            EXPECT_EQ(words[lane], expected) << "the independently preserving interpreter still "
                                                "checks exact bits on unknown words";
        }
    }
}
TEST(IrImageD16Values, DistinctPrimitivesKeepTheOldPackingPaths) {
    const auto low = build_image_d16_word(ImageD16Form::SelectedLow);
    const auto selected = build_image_d16_word(ImageD16Form::SelectedPair);
    const auto gather = build_image_d16_word(ImageD16Form::GatherPair);
    EXPECT_EQ(low.block.input_count(), 1u);
    EXPECT_EQ(selected.block.input_count(), 2u);
    EXPECT_EQ(gather.block.input_count(), 2u);
    EXPECT_EQ(low.block.at(low.word).op, Opcode::ImagePackHalfLo);
    EXPECT_EQ(selected.block.at(selected.word).op, Opcode::Or);
    EXPECT_EQ(gather.block.at(gather.word).op, Opcode::ImagePackHalf2);
}
}   // namespace
