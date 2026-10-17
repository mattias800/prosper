// Four existing F32 fused forms share one ternary operation; no encoding or fold is added.
#pragma once
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include <array>
#include <cstdint>
namespace prosper::gpu::ir {
Value build_fused_mad_float(Builder& ir, const TernaryInputs& x);
Known eval_fused_mad_float(const std::array<Known, 3>& sources);
uint32_t emit_fused_mad_float(SpirvCompute& b, const TernarySourceReader& read);
uint32_t emit_fused_mad_values(SpirvCompute& b, uint32_t a, uint32_t c, uint32_t addend);
} // namespace prosper::gpu::ir
