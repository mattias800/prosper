// The existing VOP3 32-bit low/high multiply subgroup; no new encoding admission.
#pragma once
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
std::span<const VectorOpDef> vector_multiply_defs();
const VectorOpDef* find_vector_multiply_op(Rdna2Format fmt, uint32_t opcode);
}   // namespace prosper::gpu::ir
