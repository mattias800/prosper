// Numeric reference profiles for loaded-register store fields; no guest folder registration.
#pragma once
#include "gpu/recompiler/ir/eval.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
Known evaluate_pack_normalized_field(uint32_t bits, uint32_t mode);
Known evaluate_pack_unsigned_float(uint32_t bits, uint32_t mantissa);
}   // namespace prosper::gpu::ir
