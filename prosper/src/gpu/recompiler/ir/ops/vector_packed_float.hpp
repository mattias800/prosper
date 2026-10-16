// Existing packed binary half operations: one binary32 operation per expanded half.
#pragma once
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
std::span<const TernaryIntegerDef> vector_packed_float_defs();
const TernaryIntegerDef* find_vector_packed_float_op(Rdna2Format fmt, uint32_t opcode);
}   // namespace prosper::gpu::ir
