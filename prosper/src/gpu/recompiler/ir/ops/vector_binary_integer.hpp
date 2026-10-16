// Canonical e64 aliases of the existing VOP2 integer/bitwise data definitions.
#pragma once
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
std::span<const VectorOpDef> vector_binary_integer_defs();
const VectorOpDef* find_vector_binary_integer_op(Rdna2Format fmt, uint32_t opcode);
} // namespace prosper::gpu::ir
