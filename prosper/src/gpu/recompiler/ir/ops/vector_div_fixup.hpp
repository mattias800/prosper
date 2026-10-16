// Existing division-fixup definition over lazy raw binary32 words; no guest fold is enabled.
#pragma once
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
std::span<const TernaryIntegerDef> vector_div_fixup_defs();
const TernaryIntegerDef* find_vector_div_fixup_op(Rdna2Format fmt, uint32_t opcode);
}   // namespace prosper::gpu::ir
