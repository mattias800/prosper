// The lowering must consume the caller's exact selector id and emit one UMin against the
// admitted last level. It owns no source read, admission decision or image access.
#include "gpu/recompiler/ir/lower/image_mip_clamp_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
namespace {
TEST(IrLowerImageMipClamp, UsesTheCallerSelectorAndLastLevelInUnsignedMin) {
    for (uint32_t levels : {3u, 7u}) {
        SpirvCompute b;
        b.diagnostic = {RecompileDiagnosticStage::Compute, 0x1a73600000ull + levels};
        b.begin(3u, nullptr, 64u, 1u, 1u, 64u, 0u, true, true);
        const uint32_t requested = b.load_input(2u);
        const uint32_t lod = lower_image_mip_clamp(b, requested, levels);
        const uint32_t last = b.uconst(levels - 1u);
        const auto words = b.finish();
        size_t count = 0;
        for (size_t i = 5; i < words.size(); i += words[i] >> 16u) {
            if ((words[i] & 0xffffu) != 12u) continue; // OpExtInst
            if (words[i + 4] == Glsl_UMin) {
                ++count;
                EXPECT_EQ(words[i + 2], lod);
                EXPECT_EQ(words[i + 5], requested);
                EXPECT_EQ(words[i + 6], last);
            }
            EXPECT_NE(words[i + 4], Glsl_SMin);
            EXPECT_NE(words[i + 4], Glsl_UMax);
        }
        EXPECT_EQ(count, 1u);
    }
}
} // namespace
