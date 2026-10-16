// Integer-result e64 conversions reuse the existing saturating VOP1 definitions.
#pragma once
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
namespace prosper::gpu::ir {
std::span<const ConversionDef> vector_float_integer_e64_defs();
const ConversionDef* find_vector_float_integer_e64_op(Rdna2Format fmt, uint32_t opcode);
} // namespace prosper::gpu::ir
