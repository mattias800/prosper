// Canonical e64 subword conversions share the existing VOP1 arithmetic definitions.
#pragma once
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
std::span<const ConversionDef> vector_subword_to_float_e64_defs();
const ConversionDef* find_vector_subword_to_float_e64_op(Rdna2Format fmt, uint32_t opcode);
} // namespace prosper::gpu::ir
