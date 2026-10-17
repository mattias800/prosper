// Pure buffer component calculations. Memory access and architectural writes are caller-owned.
#pragma once

#include "gpu/recompiler/ir/ir.hpp"

#include <array>
#include <cstdint>

namespace prosper::gpu {
struct SpirvCompute;
}
namespace prosper::gpu::ir {
class Builder;

enum class BufferFieldKind : uint8_t { Unsigned, Signed, Half, Unorm, Snorm, Ufloat };
struct BufferFieldShape {
    BufferFieldKind kind;
    uint32_t offset = 0, width = 0;
    float norm = 0;
};
struct BufferShiftValues {
    Value shift, joined;
};
struct BufferLocationValues {
    Value index, shift;
};

Value buffer_field(Builder& ir, Value word, const BufferFieldShape& shape);
BufferShiftValues buffer_raw_shift(Builder& ir, Value byte_address, Value word);
Value buffer_low_shift(Builder& ir, Value word, Value shift);
Value buffer_high_join(Builder& ir, Value low, Value high, Value shift);
BufferLocationValues buffer_component_location(Builder& ir, Value address, uint32_t component,
                                               uint32_t component_bytes);
Value buffer_index_offset(Builder& ir, Value index, uint32_t dwords);

uint32_t emit_buffer_field(SpirvCompute& b, uint32_t word, const BufferFieldShape& shape);
std::array<uint32_t, 2> emit_buffer_raw_shift(SpirvCompute& b, uint32_t address, uint32_t word);
uint32_t emit_buffer_low_shift(SpirvCompute& b, uint32_t word, uint32_t shift);
uint32_t emit_buffer_high_join(SpirvCompute& b, uint32_t low, uint32_t high, uint32_t shift);
std::array<uint32_t, 2> emit_buffer_component_location(SpirvCompute& b, uint32_t address,
                                                       uint32_t component, uint32_t bytes);
uint32_t emit_buffer_index_offset(SpirvCompute& b, uint32_t index, uint32_t dwords);

} // namespace prosper::gpu::ir
