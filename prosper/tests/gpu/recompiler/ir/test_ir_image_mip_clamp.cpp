// The nominal IMAGE_LOAD_MIP value preserves unsigned selection, the caller's count and the old
// input/constant order. Independent boundary values also catch signed-min and off-by-one mistakes.
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/ops/image_mip_clamp.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
namespace {
TEST(IrImageMipClamp, HandWorkedUnsignedBoundsAndCounts) {
    struct Case {
        uint32_t levels, requested, expected;
    };
    for (const auto c :
         {Case{3, 0, 0}, Case{3, 1, 1}, Case{3, 2, 2}, Case{3, 3, 2}, Case{3, 0x80000000u, 2},
          Case{6, 5, 5}, Case{6, 6, 5}, Case{7, 6, 6}, Case{7, 7, 6}, Case{7, 0xffffffffu, 6},
          Case{1, 0xffffffffu, 0}, Case{0xffffffffu, 0xffffffffu, 0xfffffffeu},
          Case{0xffffffffu, 0x80000000u, 0x80000000u}})
        EXPECT_EQ(eval_image_mip_clamp(c.requested, c.levels), Known{c.expected})
            << "levels=" << c.levels << " requested=" << c.requested;
}

TEST(IrImageMipClamp, UnknownSelectorRemainsUnknown) {
    for (uint32_t levels : {1u, 3u, 6u, 7u})
        EXPECT_FALSE(eval_image_mip_clamp(std::nullopt, levels));
}

TEST(IrImageMipClamp, InputPrecedesLastLevelConstantAndUnsignedMin) {
    const auto built = build_image_mip_clamp(7u);
    const auto& code = built.block.insts();
    ASSERT_EQ(code.size(), 3u);
    EXPECT_EQ(code[0].op, Opcode::Input);
    EXPECT_EQ(code[1].op, Opcode::ConstU32);
    EXPECT_EQ(code[1].imm, 6u);
    EXPECT_EQ(code[2].op, Opcode::UMin);
    EXPECT_EQ(code[2].args[0].index, 0u);
    EXPECT_EQ(code[2].args[1].index, 1u);
}

TEST(IrImageMipClamp, EveryEmittedLaneMatchesEvaluation) {
    constexpr uint32_t n = 64;
    const std::array<uint32_t, 11> edges{0, 1,  2,           3,           5,          6,
                                         7, 31, 0x7fffffffu, 0x80000000u, 0xffffffffu};
    std::vector<uint32_t> words;
    for (uint32_t lane = 0; lane < n; ++lane)
        words.push_back(lane < edges.size() ? edges[lane] : lane * 0x1020304u);
    for (uint32_t levels : {1u, 3u, 6u, 7u, 32u}) {
        const auto built = build_image_mip_clamp(levels);
        SpirvCompute b;
        b.diagnostic = {RecompileDiagnosticStage::Compute, 0x1a73500000ull + levels};
        b.begin(1u, nullptr, n, 1u, 1u, n, 0u, true, true);
        const std::array<uint32_t, 1> inputs{b.load_input(0)};
        const auto emitted = emit_spirv(b, built.block, inputs);
        b.store_output_word(emitted[built.lod.index], 1u, 0u);
        bpermute_oracle::Interpreter vm(b.finish());
        const auto got = vm.run_buffers(n, {{0u, words}, {1u, std::vector<uint32_t>(n)}}, 1u);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        ASSERT_EQ(got.size(), n);
        for (uint32_t lane = 0; lane < n; ++lane)
            EXPECT_EQ(Known{got[lane]}, eval_image_mip_clamp(words[lane], levels))
                << "levels=" << levels << " lane=" << lane;
    }
}
} // namespace
