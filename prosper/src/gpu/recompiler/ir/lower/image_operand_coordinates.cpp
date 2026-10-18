// Normalization body mechanically extracted from emit_alu with extract_case_routes.py.
#include "gpu/recompiler/ir/lower/image_operand_coordinates_lowering.hpp"
#include "gpu/recompiler/ir/ops/image_operand_coordinates.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
uint32_t LegacyImageSpatialNormalizer::operator()(uint32_t coordinate, uint32_t extent) const {
    if (!normalize_sampler_coordinates) return coordinate;
    if (!extent) {
        ok = false;
        return b.uconst(0);
    }
    return b.fbin(Op_FMul, coordinate, b.uconst(fbits(1.0f / static_cast<float>(extent))));
}
uint32_t ImageOperandCoordinateNormalizer::operator()(uint32_t coordinate, uint32_t extent) const {
    if (normalize_sampler_coordinates && !extent) ok = false;
    return emit_image_operand_coordinate(b, normalize_sampler_coordinates, extent, coordinate);
}
} // namespace prosper::gpu::ir
