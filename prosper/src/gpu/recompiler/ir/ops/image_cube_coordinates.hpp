// CUBE's pure stacked-face value transform. AMD 70648's image VADDR convention supplies the
// processed x/y/face tuple; its vertical face layout is the existing renderer contract
// (CONFIDENCE: MED). Resource/admission/proof state and the image access remain in emit_alu.
#pragma once

#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ops/image_coordinates.hpp"

#include <array>
#include <cstdint>

namespace prosper::gpu::ir {

struct ImageCubeCoordinateBlock {
    Block block;
    std::array<uint32_t, 3> slots;
    std::array<Value, 4> values; // uf, vf, layer, v6
};

ImageCubeCoordinateBlock build_image_cube_coordinates(uint32_t first_slot = 0);
std::array<Known, 4> eval_image_cube_coordinates(const std::array<Known, 3>& coordinates);
std::array<uint32_t, 4> emit_image_cube_coordinates(SpirvCompute& b, uint32_t first_slot,
                                                    const ImageAddressSlotReader& read);

} // namespace prosper::gpu::ir
