// Existing signed/unsigned normalized pack definitions over lazy raw-word inputs.
#pragma once
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
std::span<const TernaryIntegerDef> vector_pack_norm_defs();
const TernaryIntegerDef* find_vector_pack_norm_op(Rdna2Format fmt, uint32_t opcode);
}   // namespace prosper::gpu::ir
