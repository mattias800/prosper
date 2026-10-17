// The bounded e64 alias shares the existing half-to-float value definition.
#pragma once
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
const ConversionDef* find_vector_half_float_exact_e64_op(Rdna2Format fmt, uint32_t opcode);
} // namespace prosper::gpu::ir
