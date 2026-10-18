// DPP16's bounded, FI=0 row-XOR SRC0 value; admission and subgroup contracts stay in lowering.
#pragma once

#include "gpu/recompiler/ir/builder.hpp"
#include <cstdint>

namespace prosper::gpu {
struct SpirvCompute;
}
namespace prosper::gpu::ir {
// AMD 70648, DPP16 ROW_XMASK: XOR the lane's low four bits. ROW_ROR:8 is XOR 8.
// An inactive source supplies zero for the admitted FI=0, BOUND_CTRL=1 forms.
Value dpp_row_xor_source(Builder& ir, Value source, Value active, uint32_t stride);
uint32_t emit_dpp_row_xor_source(SpirvCompute& b, uint32_t source, uint32_t active,
                                 uint32_t stride);
} // namespace prosper::gpu::ir
