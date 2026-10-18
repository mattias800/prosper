#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"

namespace prosper::gpu::ir {
std::span<const ConversionDef> vector_integer_float_rte_e64_defs();
const ConversionDef* find_vector_integer_float_rte_e64_op(Rdna2Format fmt, uint32_t opcode);
} // namespace prosper::gpu::ir
