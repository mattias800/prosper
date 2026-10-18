// Layer expression mechanically extracted with extract_case_routes.py.
#include "gpu/recompiler/ir/lower/image_gather_layer_lowering.hpp"
#include "gpu/recompiler/ir/ops/image_coordinates.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
uint32_t lower_image_gather_layer(SpirvCompute&, bool float_array_gather, bool is_gather_lz_o,
                                  const ImageAddressSlotReader& read) {
    const auto cvg = [](uint32_t slot) { return slot; };
    const auto& vread = read;
    const uint32_t layer = float_array_gather ? vread(cvg(is_gather_lz_o ? 3u : 2u)) : 0u;
    return layer;
}
} // namespace prosper::gpu::ir
