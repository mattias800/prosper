// Existing cubemap ID and signed major-axis definitions; no new guest fold domain.
#pragma once
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <span>
#include <cstdint>
namespace prosper::gpu::ir {
std::span<const TernaryIntegerDef> vector_cube_axis_defs();
const TernaryIntegerDef* find_vector_cube_axis_op(Rdna2Format, uint32_t);
}   // namespace prosper::gpu::ir
