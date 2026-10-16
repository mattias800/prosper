// Existing byte permutation definition; selectors and byte data stay in the same lane.
#pragma once
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
std::span<const TernaryIntegerDef> vector_permute_defs();
const TernaryIntegerDef* find_vector_permute_op(Rdna2Format fmt, uint32_t opcode);
}   // namespace prosper::gpu::ir
