// Plain VOP1 word operations; the caller owns source transforms and the EXEC-predicated write.
#pragma once
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
Value build_integer_unary_not_b32(Builder& ir, const VectorOpIn& in);
Value build_integer_unary_bfrev_b32(Builder& ir, const VectorOpIn& in);
Value build_integer_unary_ffbh_u32(Builder& ir, const VectorOpIn& in);
Value build_integer_unary_ffbl_b32(Builder& ir, const VectorOpIn& in);
std::span<const VectorOpDef> vector_integer_unary_defs();
const VectorOpDef* find_vector_integer_unary_op(Rdna2Format fmt, uint32_t opcode);
}   // namespace prosper::gpu::ir
