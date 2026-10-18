// Gather layer words use the shared identity definition; an absent layer keeps its id sentinel.
#include "gpu/recompiler/ir/lower/image_gather_layer_lowering.hpp"
#include "gpu/recompiler/ir/ops/image_coordinates.hpp"
#include "gpu/recompiler/ir/ops/image_operand_coordinates.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
uint32_t lower_image_gather_layer(SpirvCompute& b, bool float_array_gather, bool is_gather_lz_o,
                                  const ImageAddressSlotReader& read) {
    if (!float_array_gather) return 0u;
    return emit_image_operand_coordinate(b, false, 0, read(is_gather_lz_o ? 3u : 2u));
}
} // namespace prosper::gpu::ir
