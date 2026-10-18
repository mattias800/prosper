// CUBE's value-only lowering. The caller-owned reader retains the source proof in Block order.
#include "gpu/recompiler/ir/lower/image_cube_coordinates_lowering.hpp"

#include "gpu/recompiler/ir/ops/image_cube_coordinates.hpp"

namespace prosper::gpu::ir {

std::array<uint32_t, 4> lower_image_cube_coordinates(SpirvCompute& b, uint32_t ci,
                                                     const ImageAddressSlotReader& read) {
    return emit_image_cube_coordinates(b, ci, read);
}

} // namespace prosper::gpu::ir
