// Existing word alignment operations; source and result bits stay local to their lane.
#pragma once
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
std::span<const TernaryIntegerDef> vector_align_defs();
const TernaryIntegerDef* find_vector_align_op(Rdna2Format fmt, uint32_t opcode);
}   // namespace prosper::gpu::ir
