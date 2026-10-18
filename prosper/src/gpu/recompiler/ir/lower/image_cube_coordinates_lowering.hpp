// The CUBE value transform only. The caller retains admission, resource resolution, the image
// access and result writes. Its reader performs all MIMG source/proof work in the old order.
#pragma once

#include "gpu/recompiler/ir/ops/image_coordinates.hpp"

#include <array>
#include <cstdint>

namespace prosper::gpu::ir {

std::array<uint32_t, 4> lower_image_cube_coordinates(SpirvCompute& b, uint32_t ci,
                                                     const ImageAddressSlotReader& read);

} // namespace prosper::gpu::ir
