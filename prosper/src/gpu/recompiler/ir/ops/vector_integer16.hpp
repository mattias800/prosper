// Selected, zero-extended U16 sources; results are zero-extended U16 words.
// Half selection, destination merging, modifiers and EXEC remain with the lowering/caller.
#pragma once
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
std::span<const TernaryIntegerDef> vector_integer16_defs();
const TernaryIntegerDef* find_vector_integer16_op(Rdna2Format fmt, uint32_t opcode);
} // namespace prosper::gpu::ir
