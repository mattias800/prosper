// Existing VOP3 RTZ conversion from two raw binary32 words to a full packed dword.
#pragma once
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
std::span<const TernaryIntegerDef> vector_rtz_pack_defs();
const TernaryIntegerDef* find_vector_rtz_pack_op(Rdna2Format fmt, uint32_t opcode);
}   // namespace prosper::gpu::ir
