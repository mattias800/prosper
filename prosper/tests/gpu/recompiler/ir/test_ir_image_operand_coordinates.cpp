// One MIMG spatial word: independent ISA/texel-space expectations and the actual emitted Block
// per lane against eval. The access and the EXEC-masked destination are tested on Vulkan.
#include "gpu/recompiler/ir/ops/image_operand_coordinates.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
namespace {
uint32_t f(float x) {
    return std::bit_cast<uint32_t>(x);
}
TEST(IrImageOperandCoordinates, HandWorkedScalingAndRawIdentity) {
    // 70648 8.2: FORCE_UNNORMALIZED changes spatial texels into normalized coordinates.
    EXPECT_EQ(eval_image_operand_coordinate(true, 4, f(3.0f)), Known{f(0.75f)});
    EXPECT_EQ(eval_image_operand_coordinate(true, 8, f(-2.0f)), Known{f(-0.25f)});
    // Gradients use the same per-axis scale: 0.5/64 and 0.25/8 are exact powers of two.
    EXPECT_EQ(eval_image_operand_coordinate(true, 64, f(0.5f)), Known{0x3c000000u});
    EXPECT_EQ(eval_image_operand_coordinate(true, 8, f(0.25f)), Known{0x3d000000u});
    // Round the reciprocal once: 1.5 * f32(1/3) lies below the midpoint above 0.5.
    EXPECT_EQ(eval_image_operand_coordinate(true, 3, f(1.5f)), Known{f(0.5f)});
    for (uint32_t raw : {0u, 0x80000000u, 0x7f800001u, 0xffc00001u, 0xdeadbeefu})
        EXPECT_EQ(eval_image_operand_coordinate(false, 0, raw), Known{raw});
}
TEST(IrImageOperandCoordinates, UnknownInputsDoNotAcquireValues) {
    EXPECT_FALSE(eval_image_operand_coordinate(false, 0, std::nullopt));
    EXPECT_FALSE(eval_image_operand_coordinate(true, 64, std::nullopt));
    EXPECT_EQ(eval_image_operand_coordinate(true, 0, std::nullopt), Known{0u});
}
std::vector<uint32_t> run(bool normalize, uint32_t extent, const std::vector<uint32_t>& in) {
    const auto built = build_image_operand_coordinate(normalize, extent);
    const uint32_t n = uint32_t(in.size());
    SpirvCompute b;
    b.begin(1, nullptr, n, 1, 1, n, 0, true, true);
    const uint32_t inputs[] = {b.load_input(0)};
    const auto ids = emit_spirv(b, built.block, inputs);
    b.store_output_word(ids[built.coordinate.index], 1, 0);
    bpermute_oracle::Interpreter vm(b.finish());
    auto out = vm.run_buffers(n, {{0u, in}, {1u, std::vector<uint32_t>(n)}}, 1u);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    return out;
}
TEST(IrImageOperandCoordinates, EveryModeMatchesEvaluationPerLane) {
    uint32_t seed = 5409;
    std::vector<uint32_t> bits, finite;
    for (uint32_t lane = 0; lane < 64; ++lane) {
        seed = seed * 1664525u + 1013904223u;
        bits.push_back(seed);
        finite.push_back(f(float(int(lane) - 32) * 0.5f));
    }
    for (bool normalize : {false, true})
        for (uint32_t extent : {0u, 3u, 8u, 64u, 1920u}) {
            const auto& in = normalize ? finite : bits;
            const auto out = run(normalize, extent, in);
            ASSERT_EQ(out.size(), in.size());
            for (size_t lane = 0; lane < in.size(); ++lane)
                EXPECT_EQ(Known{out[lane]},
                          eval_image_operand_coordinate(normalize, extent, in[lane]))
                    << "lane=" << lane << " normalize=" << normalize << " extent=" << extent;
        }
}
} // namespace
