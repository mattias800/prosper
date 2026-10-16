// Existing VOP3 DX9 multiply; modifiers/publication stay with the lowering and root.
#pragma once
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
std::span<const TernaryIntegerDef> vector_legacy_mul_defs();
const TernaryIntegerDef* find_vector_legacy_mul_op(Rdna2Format fmt, uint32_t opcode);
}   // namespace prosper::gpu::ir
