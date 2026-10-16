// Four canonical e64 integer data aliases. MOV remains outside this admission.
#pragma once
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
std::span<const VectorOpDef> vector_unary_integer_e64_defs();
const VectorOpDef* find_vector_unary_integer_e64_op(Rdna2Format fmt, uint32_t opcode);
} // namespace prosper::gpu::ir
