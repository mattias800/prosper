// image_coordinates.cpp — the spatial coordinates of a MIMG 2D or 3D access
// (image_coordinates_lowering.hpp).
#include "gpu/recompiler/ir/lower/image_coordinates_lowering.hpp"

namespace prosper::gpu::ir {

std::array<uint32_t, 3> lower_image_coordinates(SpirvCompute& b, bool& ok,
                                                const ImageCoordinateShape& shape,
                                                const ImageAddressSlotReader& read) {
    const auto ids = emit_image_coordinates(b, shape, read);
    if (shape.normalize)
        for (uint32_t axis = 0; axis < shape.axes; ++axis)
            if (!shape.extent[axis]) ok = false;
    return ids;
}

}   // namespace prosper::gpu::ir
