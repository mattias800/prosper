// Plain VOP1 word operations; the caller owns source transforms and the EXEC-predicated write.
#pragma once
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
std::span<const VectorOpDef> vector_integer_unary_defs();
const VectorOpDef* find_vector_integer_unary_op(Rdna2Format fmt, uint32_t opcode);
}   // namespace prosper::gpu::ir
