// image_coordinates.cpp — the spatial coordinates of a MIMG 2D or 3D access
// (image_coordinates_lowering.hpp).
#include "gpu/recompiler/ir/lower/image_coordinates_lowering.hpp"

#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/resources/shader_resources.hpp"

namespace prosper::gpu::ir {

std::array<uint32_t, 3> lower_image_coordinates_3d(SpirvCompute& b, const ShaderResource* res,
                                                   bool is_load, const ImageAddressReader& read,
                                                   const ImageCoordinateNormalizer& normalized_spatial) {
    (void)b;
    // The moved text reads `vread(cvg(k))`; `read` performs both proof calls in that order.
    const auto cvg = [](uint32_t k) { return k; };
    const auto vread = [&](uint32_t slot) { return read(slot); };
    uint32_t cu = vread(cvg(0)), cv = vread(cvg(1)), cw = vread(cvg(2));
    if (!is_load) {
        cu = normalized_spatial(cu, res->width);
        cv = normalized_spatial(cv, res->height);
        cw = normalized_spatial(cw, res->depth);
    }
    return {cu, cv, cw};
}

std::array<uint32_t, 2> lower_image_coordinates_2d(SpirvCompute& b, const Rdna2Inst& in,
                                                   const ShaderResource* res, bool is_load,
                                                   const ImageAddressReader& read,
                                                   const ImageCoordinateNormalizer& normalized_spatial) {
    const auto cvg = [](uint32_t k) { return k; };
    const auto vread = [&](uint32_t slot) { return read(slot); };
    uint32_t cu, cv;
    if (in.mimg_a16) {
        const uint32_t packed = vread(cvg(0));
        cu = b.bfe_u(packed, b.uconst(0), b.uconst(16));
        cv = b.bfe_u(packed, b.uconst(16), b.uconst(16));
    } else {
        cu = vread(cvg(0));
        cv = vread(cvg(1));
    }
    if (!is_load) {
        cu = normalized_spatial(cu, res->width);
        cv = normalized_spatial(cv, res->height);
    }
    return {cu, cv};
}

}   // namespace prosper::gpu::ir
