// Binary32 bits use the shared lazy ternary word runner; no guest folder domain is expanded.
#pragma once
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <array>
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
using MinmaxFloatDef = TernaryIntegerDef;
std::span<const MinmaxFloatDef> vector_minmax_float_defs();
const MinmaxFloatDef* find_vector_minmax_float_op(Rdna2Format fmt, uint32_t opcode);
Known eval_minmax_float_op(const MinmaxFloatDef& def, const std::array<Known, 3>& sources);
uint32_t emit_minmax_float_op(SpirvCompute& b, const MinmaxFloatDef& def,
                              const TernarySourceReader& read);
}   // namespace prosper::gpu::ir
