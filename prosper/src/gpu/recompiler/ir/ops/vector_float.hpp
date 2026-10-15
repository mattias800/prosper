// Binary32 per-lane operation definitions. No wave state, source modifiers or folder widening.
#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
namespace prosper::gpu::ir {
std::span<const VectorOpDef> vector_float_defs();
const VectorOpDef* find_vector_float_op(Rdna2Format fmt, uint32_t opcode);
} // namespace prosper::gpu::ir
