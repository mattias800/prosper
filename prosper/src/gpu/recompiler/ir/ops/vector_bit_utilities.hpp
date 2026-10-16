// Ordinary numeric U32 data bit utilities; source facts come from instruction effects.
#pragma once
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
std::span<const TernaryIntegerDef> vector_bit_utilities_defs();
const TernaryIntegerDef* find_vector_bit_utilities_op(Rdna2Format fmt, uint32_t opcode);
}   // namespace prosper::gpu::ir
