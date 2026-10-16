// Existing VOP1 f16 unary operations. S0 is already unpacked into binary32; D is raw f16
// in the low word. Source selection, destination placement and predicate writes stay in lowering.
#pragma once
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
std::span<const VectorOpDef> vector_half_unary_defs();
const VectorOpDef* find_vector_half_unary_op(Rdna2Format fmt, uint32_t opcode);
}   // namespace prosper::gpu::ir
