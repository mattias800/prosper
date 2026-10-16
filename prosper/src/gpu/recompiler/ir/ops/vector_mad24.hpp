// Existing unsigned 24-bit multiply-add only; no signed or new encoding admissions.
#pragma once
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
std::span<const TernaryIntegerDef> vector_mad24_defs();
const TernaryIntegerDef* find_vector_mad24_op(Rdna2Format fmt, uint32_t opcode);
}   // namespace prosper::gpu::ir
