// The CUBE lowering retains one instruction-order source read per address slot. Resource
// resolution/admission and all source-proof state remain the reader's caller-owned work.
#include "gpu/recompiler/ir/lower/image_cube_coordinates_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
namespace {
TEST(IrLowerImageCubeCoordinates, ReadsEachShiftedSlotOnceInOrder) {
    for (uint32_t first : {0u, 1u}) {
        SpirvCompute b;
        b.diagnostic = {RecompileDiagnosticStage::Compute, 0x5d5428d000ull + first};
        b.begin(3u, nullptr, 64u, 1u, 1u, 64u, 0u, true, true);
        std::vector<uint32_t> slots;
        const auto ids = lower_image_cube_coordinates(b, first, [&](uint32_t slot) {
            slots.push_back(slot);
            return b.load_input(slot - first);
        });
        EXPECT_EQ(slots, (std::vector<uint32_t>{first, first + 1u, first + 2u}));
        for (uint32_t id : ids) EXPECT_NE(id, 0u);
        const auto module = b.finish();
        std::array<uint32_t, 3> counts{};
        for (size_t i = 5; i < module.size(); i += module[i] >> 16u)
            if ((module[i] & 0xffffu) == Op_ExtInst) {
                const uint32_t ext = module[i + 4u];
                if (ext == Glsl_FMin) ++counts[0];
                if (ext == Glsl_FMax) ++counts[1];
                if (ext == Glsl_RoundEven) ++counts[2];
                EXPECT_NE(ext, Glsl_NMin);
                EXPECT_NE(ext, Glsl_NMax);
            }
        EXPECT_EQ(counts, (std::array<uint32_t, 3>{2u, 2u, 1u}));
    }
}
} // namespace
