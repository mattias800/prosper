// The MIMG coordinate lowering (ir/lower/image_coordinates.cpp): it calls the reader once per
// address slot, in slot order, returns the reader's ids for an unscaled Separate shape, and
// refuses (clears ok) exactly when a FORCE_UNNORMALIZED axis has no extent.
#include "gpu/recompiler/ir/lower/image_coordinates_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;

namespace {

struct Lowered {
    std::array<uint32_t, 3> ids{};
    std::vector<uint32_t> slots;
    std::vector<uint32_t> read_ids;
    bool ok = true;
};

Lowered lower(const ImageCoordinateShape& shape) {
    SpirvCompute b;
    b.diagnostic = {RecompileDiagnosticStage::Compute, 0x1a6d0000u};
    b.begin(3u, nullptr, 64u, 1u, 1u, 64u, 0u, true, true);
    Lowered out;
    out.ids = lower_image_coordinates(b, out.ok, shape, [&](uint32_t slot) {
        out.slots.push_back(slot);
        out.read_ids.push_back(b.load_input(slot));
        return out.read_ids.back();
    });
    return out;
}

TEST(IrLowerImageCoordinates, ReadsEachSlotOnceInOrder) {
    const Lowered three = lower({3, ImageCoordinateForm::Separate, false, {}});
    EXPECT_EQ(three.slots, (std::vector<uint32_t>{0, 1, 2}));
    EXPECT_EQ(three.ids,
              (std::array<uint32_t, 3>{three.read_ids[0], three.read_ids[1], three.read_ids[2]}));
    EXPECT_TRUE(three.ok);
    const Lowered packed = lower({2, ImageCoordinateForm::Packed16, false, {}});
    EXPECT_EQ(packed.slots, (std::vector<uint32_t>{0}));
    EXPECT_NE(packed.ids[0], packed.read_ids[0]);   // an extract, not the packed word
    EXPECT_NE(packed.ids[0], packed.ids[1]);
    EXPECT_TRUE(packed.ok);
}

TEST(IrLowerImageCoordinates, AZeroExtentRefusesOnlyWhenNormalizing) {
    EXPECT_TRUE(lower({2, ImageCoordinateForm::Separate, true, {4, 8, 0}}).ok);
    EXPECT_FALSE(lower({2, ImageCoordinateForm::Separate, true, {0, 8, 0}}).ok);
    EXPECT_FALSE(lower({3, ImageCoordinateForm::Separate, true, {4, 8, 0}}).ok);
    // The 2D shape never reads the third extent (the first arm), and an unscaled shape never
    // reads any.
    EXPECT_TRUE(lower({3, ImageCoordinateForm::Separate, false, {0, 0, 0}}).ok);
    // Scaling reads every slot first, like the unscaled shape.
    EXPECT_EQ(lower({3, ImageCoordinateForm::Separate, true, {2, 4, 8}}).slots,
              (std::vector<uint32_t>{0, 1, 2}));
}

}   // namespace
