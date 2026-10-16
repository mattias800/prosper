// Selected, zero-extended U16 inputs; a whole U32 bit pack, without floating conversion.
#pragma once
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
std::span<const TernaryIntegerDef> vector_pack16_defs();
const TernaryIntegerDef* find_vector_pack16_op(Rdna2Format fmt, uint32_t opcode);
}   // namespace prosper::gpu::ir
