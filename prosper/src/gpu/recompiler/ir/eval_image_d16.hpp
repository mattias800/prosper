// Exact-only evaluation for MIMG D16; the generic PackHalfLo RTE reference stays separate.
#pragma once
#include "gpu/recompiler/ir/eval.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
Known image_half_exact(uint32_t bits);
Known image_half_pair_exact(uint32_t low, uint32_t high);
} // namespace prosper::gpu::ir
