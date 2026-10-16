// Canonical e64 aliases of the existing half-to-integer conversion definitions.
#pragma once
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <span>
#include <cstdint>
namespace prosper::gpu::ir {
std::span<const ConversionDef> vector_half_integer_e64_defs();
const ConversionDef* find_vector_half_integer_e64_op(Rdna2Format fmt, uint32_t opcode);
} // namespace prosper::gpu::ir
