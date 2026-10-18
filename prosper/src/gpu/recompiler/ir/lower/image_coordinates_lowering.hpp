// image_coordinates_lowering.hpp — the spatial coordinates of a MIMG 2D or 3D access
// (HOW_TO_ADD_A_FAMILY.md, "Image instructions (MIMG)").
//
// emit_alu's MIMG case resolves the T#/S#, decides the access and every refusal, and keeps the
// MIMG source proof (`begin_mimg_source_proof`, `mimg_address_vgpr`, `read_mimg_source_word`). It
// hands this file the coordinate shape and a reader that performs exactly those two proof calls
// for one address slot, and gets back the coordinate ids the image access consumes. The value is
// the definition in ../ops/image_coordinates.cpp. Nothing here reads or writes provenance, marks
// or proof state.
#pragma once

#include "gpu/recompiler/ir/ops/image_coordinates.hpp"

#include <array>
#include <cstdint>

namespace prosper::gpu::ir {

// Emits the coordinates in emit_alu's order. A FORCE_UNNORMALIZED axis whose extent is 0 has no
// normalized value: it clears `ok`, as emit_alu's `normalized_spatial` did.
std::array<uint32_t, 3> lower_image_coordinates(SpirvCompute& b, bool& ok,
                                                const ImageCoordinateShape& shape,
                                                const ImageAddressSlotReader& read);

}   // namespace prosper::gpu::ir
