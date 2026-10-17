// AMD 70648 10.4.4 and 12.13.1: DS gathers without LDS access. Source EXEC supplies zero;
// destination EXEC, register publication and subgroup contracts remain the lowering's work.
#pragma once

#include "gpu/recompiler/ir/builder.hpp"

#include <cstdint>
#include <functional>

namespace prosper::gpu {
struct SpirvCompute;
}

namespace prosper::gpu::ir {
Value ds_bpermute_b32(Builder& ir, Value address, Value source, Value active, Value byte_offset);

// A swizzle computes the peer before reading its source word, then reads EXEC after the first
// shuffle. Deferred inputs retain those positions, including a missing-source zero constant.
struct DsSwizzleInputs {
    std::function<Value()> source;
    std::function<Value()> active;
};
Value ds_swizzle_quad_b32(Builder& ir, uint32_t control, const DsSwizzleInputs& inputs);
Value ds_swizzle_mask_b32(Builder& ir, uint32_t control, const DsSwizzleInputs& inputs);

uint32_t emit_ds_bpermute_b32(SpirvCompute& b, uint32_t address, uint32_t source, uint32_t active,
                              uint32_t byte_offset);
uint32_t emit_ds_swizzle_b32(SpirvCompute& b, uint32_t control,
                             const std::function<uint32_t()>& source, uint32_t active);
} // namespace prosper::gpu::ir
