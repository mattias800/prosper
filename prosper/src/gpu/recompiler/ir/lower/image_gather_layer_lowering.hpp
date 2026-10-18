// Gather layer bookkeeping: the reader retains the original MIMG slot/proof operation.
#pragma once
#include "gpu/recompiler/ir/ops/image_coordinates.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
uint32_t lower_image_gather_layer(SpirvCompute& b, bool float_array_gather, bool is_gather_lz_o,
                                  const ImageAddressSlotReader& read);
} // namespace prosper::gpu::ir
