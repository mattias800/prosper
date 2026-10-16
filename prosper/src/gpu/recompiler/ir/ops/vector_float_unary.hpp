// Existing VOP1 F32 unary definitions, shared by eval and emission.
#pragma once
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include <cstdint>
#include "gpu/recompiler/rdna2_decode.hpp"
#include <span>
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
namespace prosper::gpu::ir {
Value build_vector_fract(Builder& ir, const VectorOpIn& x);
Value build_vector_trunc(Builder& ir, const VectorOpIn& x);
Value build_vector_ceil(Builder& ir, const VectorOpIn& x);
Value build_vector_roundeven(Builder& ir, const VectorOpIn& x);
Value build_vector_floor(Builder& ir, const VectorOpIn& x);
std::span<const VectorOpDef> vector_float_unary_defs();
const VectorOpDef* find_vector_float_unary_op(Rdna2Format fmt, uint32_t opcode);
}   // namespace prosper::gpu::ir
