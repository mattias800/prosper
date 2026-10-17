// Bit-level input flushing; instruction-specific MODE selection belongs to the lowering.
#pragma once
#include "gpu/recompiler/ir/builder.hpp"
namespace prosper::gpu::ir {
Value build_flush_f32_input(Builder& ir, Value bits);
Value build_preserve_f32_zero(Builder& ir, Value input, Value result);
} // namespace prosper::gpu::ir
