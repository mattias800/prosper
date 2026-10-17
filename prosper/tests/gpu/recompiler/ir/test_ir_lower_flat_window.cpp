// These wrappers bind existing IDs and receive width facts; they resolve no guest resources.
#include "gpu/recompiler/ir/lower/flat_window_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <cstdint>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
namespace {
TEST(IrLowerFlatWindow, SubwordAccessKeepsTheOriginalByteAddressId) {
    SpirvCompute b;
    b.begin(1, nullptr, 32, 1, 1, 32, 0, true, true);
    const uint32_t byte0 = b.load_input(0);
    for (uint32_t bits : {8u, 16u}) {
        FlatAccessInfo access{};
        access.bits = bits;
        const auto result = lower_flat_window_component(b, byte0, access, 3);
        EXPECT_EQ(result[0], byte0);
        EXPECT_NE(result[1], byte0);
    }
}
TEST(IrLowerFlatWindow, DwordComponentZeroStillEmitsTheOriginalAdd) {
    SpirvCompute b;
    b.begin(1, nullptr, 32, 1, 1, 32, 0, true, true);
    const uint32_t byte0 = b.load_input(0);
    FlatAccessInfo access{};
    access.bits = 32;
    const auto result = lower_flat_window_component(b, byte0, access, 0);
    EXPECT_NE(result[0], byte0);
    EXPECT_NE(result[1], result[0]);
}
} // namespace
