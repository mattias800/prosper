// image_coordinates_lowering.hpp — the spatial coordinates of a MIMG 2D or 3D access
// (HOW_TO_ADD_A_FAMILY.md, "Image instructions (MIMG)").
//
// emit_alu's MIMG case resolves the T#/S#, decides the access and every refusal, and keeps the
// MIMG source proof (`begin_mimg_source_proof`, `mimg_address_vgpr`, `read_mimg_source_word`). It
// hands this file a reader that performs exactly those two proof calls for one address slot, and
// gets back the coordinate ids the image access consumes. Nothing here reads or writes provenance,
// marks or proof state.
#pragma once

#include <array>
#include <cstdint>
#include <functional>

namespace prosper::gpu {
struct SpirvCompute;
struct Rdna2Inst;
struct ShaderResource;
}   // namespace prosper::gpu

namespace prosper::gpu::ir {

// emit_alu's `vread(cvg(slot))`: the MIMG address operand `slot`, read through the source proof.
using ImageAddressReader = std::function<uint32_t(uint32_t slot)>;
// emit_alu's `normalized_spatial` (relocation only).
using ImageCoordinateNormalizer = std::function<uint32_t(uint32_t coordinate, uint32_t extent)>;

std::array<uint32_t, 3> lower_image_coordinates_3d(SpirvCompute& b, const ShaderResource* res,
                                                   bool is_load, const ImageAddressReader& read,
                                                   const ImageCoordinateNormalizer& normalized_spatial);

std::array<uint32_t, 2> lower_image_coordinates_2d(SpirvCompute& b, const Rdna2Inst& in,
                                                   const ShaderResource* res, bool is_load,
                                                   const ImageAddressReader& read,
                                                   const ImageCoordinateNormalizer& normalized_spatial);

}   // namespace prosper::gpu::ir
