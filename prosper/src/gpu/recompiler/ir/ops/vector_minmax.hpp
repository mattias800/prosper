// The six existing VOP3 signed/unsigned triple min/max/median operations.
#pragma once
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
std::span<const TernaryIntegerDef> vector_minmax_defs();
const TernaryIntegerDef* find_vector_minmax_op(Rdna2Format fmt, uint32_t opcode);
}   // namespace prosper::gpu::ir
