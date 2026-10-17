// VOP3 e64 aliases of the existing one-word F32 rounding definitions.
#pragma once
#include <cstdint>
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include <span>
namespace prosper::gpu::ir {
std::span<const VectorOpDef> vector_round_float_defs();
const VectorOpDef* find_vector_round_float_op(Rdna2Format fmt, uint32_t opcode);
}   // namespace prosper::gpu::ir
