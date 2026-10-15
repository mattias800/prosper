// Existing VOP1 F32 unary definitions, shared by eval and emission.
#pragma once
#include <cstdint>
#include "gpu/recompiler/rdna2_decode.hpp"
#include <span>
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
namespace prosper::gpu::ir {
std::span<const VectorOpDef> vector_float_unary_defs();
const VectorOpDef* find_vector_float_unary_op(Rdna2Format fmt, uint32_t opcode);
}   // namespace prosper::gpu::ir
