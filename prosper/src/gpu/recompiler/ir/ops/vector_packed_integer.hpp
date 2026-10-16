// Inputs are zero-extended selected halves after the existing wrapper; output is one masked half.
// Packed source wrappers remain in lowering, with no new guest fold domain.
#pragma once
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
std::span<const TernaryIntegerDef> vector_packed_integer_defs();
const TernaryIntegerDef* find_vector_packed_integer_op(Rdna2Format, uint32_t);
}   // namespace prosper::gpu::ir
