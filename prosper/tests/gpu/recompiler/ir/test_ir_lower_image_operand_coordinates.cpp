// MIMG normalization bookkeeping: the assigned IR adapter has the raw adapter's exact refusal
// boundary, and a pass-through operand remains the same SPIR-V id without gaining a value.
#include "gpu/recompiler/ir/lower/image_operand_coordinates_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <cstdint>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
namespace {
TEST(IrLowerImageOperandCoordinates, IdentityPreservesTheInputIdAndExistingFailure) {
    for (bool initial : {false, true})
        for (bool float_id : {false, true}) {
            SpirvCompute b;
            b.begin(1, nullptr, 64, 1, 1, 64, 0);
            bool ok = initial;
            const uint32_t word =
                float_id ? b.fbin(Op_FAdd, b.uconst(0x3f800000u), b.uconst(0x40000000u))
                         : b.uconst(0x7f800001u);
            const auto words = b.code.size();
            EXPECT_EQ((LegacyImageSpatialNormalizer{b, ok, false})(word, 0), word);
            EXPECT_EQ((ImageOperandCoordinateNormalizer{b, ok, false})(word, 0), word);
            EXPECT_EQ(ok, initial);
            EXPECT_EQ(b.code.size(), words);
        }
}
TEST(IrLowerImageOperandCoordinates, OnlyNormalizedZeroExtentRefusesAndReturnsZero) {
    for (bool use_ir : {false, true}) {
        SpirvCompute b;
        b.begin(1, nullptr, 64, 1, 1, 64, 0);
        bool ok = true;
        const uint32_t word = b.uconst(0x40400000u);
        const uint32_t result = use_ir ? ImageOperandCoordinateNormalizer{b, ok, true}(word, 0)
                                       : LegacyImageSpatialNormalizer{b, ok, true}(word, 0);
        EXPECT_FALSE(ok);
        EXPECT_EQ(result, b.uconst(0));
    }
}
TEST(IrLowerImageOperandCoordinates, ScalingNeverRestoresAnExistingFailure) {
    for (bool use_ir : {false, true}) {
        SpirvCompute b;
        b.begin(1, nullptr, 64, 1, 1, 64, 0);
        bool ok = false;
        const uint32_t word = b.load_input(0);
        const uint32_t result = use_ir ? ImageOperandCoordinateNormalizer{b, ok, true}(word, 64)
                                       : LegacyImageSpatialNormalizer{b, ok, true}(word, 64);
        EXPECT_FALSE(ok);
        EXPECT_NE(result, 0u);
        EXPECT_NE(result, word);
    }
}
} // namespace
