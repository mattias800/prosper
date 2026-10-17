// These aliases reuse integer conversion values; their constant-only admission stays in lower/.
#pragma once
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>

namespace prosper::gpu::ir {
std::span<const ConversionDef> vector_exact_integer_float_e64_defs();
const ConversionDef* find_vector_exact_integer_float_e64_op(Rdna2Format fmt, uint32_t opcode);
} // namespace prosper::gpu::ir
