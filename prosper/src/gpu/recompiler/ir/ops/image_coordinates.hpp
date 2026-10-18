// image_coordinates.hpp — the spatial coordinates of a MIMG 2D or 3D access, the first image
// definition (HOW_TO_ADD_A_FAMILY.md, "Image instructions (MIMG)").
//
// A definition here is VALUE only: how the address VGPRs become the coordinates the image access
// consumes. Which T#/S# the access resolves to (and so the extents and FORCE_UNNORMALIZED), which
// SPIR-V image operation runs, the result write and every MIMG source-proof step stay in the
// lowering and in emit_alu, where they were.
#pragma once

#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace prosper::gpu {
struct SpirvCompute;
}

namespace prosper::gpu::ir {

enum class ImageCoordinateForm : uint8_t {
    // One 32-bit address VGPR per axis, in axis order (AMD 70648, "Image Instructions", the
    // VADDR layout): x, y[, z].
    Separate,
    // A16 integer texel coordinates: x = VADDR[15:0], y = VADDR[31:16] of ONE address VGPR.
    // Only the 2D IMAGE_LOAD form reaches here; emit_alu refuses every other A16 shape.
    Packed16,
};

struct ImageCoordinateShape {
    uint32_t axes = 2;                                   // 2 (2D) or 3 (3D)
    ImageCoordinateForm form = ImageCoordinateForm::Separate;   // Packed16 needs axes == 2
    // S# FORCE_UNNORMALIZED on a SAMPLE: each coordinate is texel-space f32 and becomes
    // coordinate * (1 / extent). An axis whose extent is 0 has no such value; it is the constant 0
    // and the lowering refuses the instruction.
    bool normalize = false;
    std::array<uint32_t, 3> extent{};   // the T#'s width, height, depth
};

struct ImageCoordinateBlock {
    Block block;
    std::vector<uint32_t> inputs;   // input slot -> the MIMG address slot it reads
    std::array<Value, 3> coord;   // the first `axes` are valid
};

// Shared per-axis value, after the caller has read its word. This appends the original
// reciprocal constant before the multiply; it never resolves an address or changes proof state.
Value build_normalized_image_coordinate(Builder& ir, Value coordinate, uint32_t extent);

// Builds the definition. Inputs and constants are appended where the pre-IR emitter read the
// address VGPR or allocated the constant, so the emitted SPIR-V is the old SPIR-V.
ImageCoordinateBlock build_image_coordinates(const ImageCoordinateShape& shape);

// The coordinates for known address words (slot 0, 1, 2; an unread slot is ignored).
std::array<Known, 3> eval_image_coordinates(const ImageCoordinateShape& shape,
                                            const std::array<Known, 3>& address);

// Emits the definition; `read` is called for each address slot at its position in Block order.
using ImageAddressSlotReader = std::function<uint32_t(uint32_t slot)>;
std::array<uint32_t, 3> emit_image_coordinates(SpirvCompute& b, const ImageCoordinateShape& shape,
                                               const ImageAddressSlotReader& read);

}   // namespace prosper::gpu::ir
