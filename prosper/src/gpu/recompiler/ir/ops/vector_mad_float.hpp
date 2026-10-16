// Raw binary32 words run through the shared ternary word machinery; no guest folder is widened.
#pragma once
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <array>
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
using MadFloatDef = TernaryIntegerDef;
std::span<const MadFloatDef> vector_mad_float_defs();
const MadFloatDef* find_vector_mad_float_op(Rdna2Format fmt, uint32_t opcode);
Known eval_mad_float_op(const MadFloatDef& def, const std::array<Known, 3>& sources);
uint32_t emit_mad_float_op(SpirvCompute& b, const MadFloatDef& def,
                           const TernarySourceReader& read);
}   // namespace prosper::gpu::ir
