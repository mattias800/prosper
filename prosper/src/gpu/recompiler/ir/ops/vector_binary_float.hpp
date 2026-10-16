// Existing VOP3 binary F32 forms only. Lazy word inputs preserve original source-call positions.
#pragma once
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
// Shared lane arithmetic; operand wrappers remain with each format's lowering.
Value build_float_add(Builder& ir, const TernaryInputs& x);
Value build_float_mul(Builder& ir, const TernaryInputs& x);
Value build_float_nmin(Builder& ir, const TernaryInputs& x);
Value build_float_nmax(Builder& ir, const TernaryInputs& x);
std::span<const TernaryIntegerDef> vector_binary_float_defs();
const TernaryIntegerDef* find_vector_binary_float_op(Rdna2Format fmt, uint32_t opcode);
}   // namespace prosper::gpu::ir
