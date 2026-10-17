// Pure loaded-word conversions and component arithmetic. The caller owns every memory
// access, format/routing decision, proof, old destination and predicated publication.
#pragma once

#include <cstdint>

namespace prosper::gpu {
struct SpirvCompute;
}
namespace prosper::gpu::ir {

struct BufferShiftedWord {
    uint32_t shift, joined;
};
struct BufferComponentLocation {
    uint32_t index, shift;
};
struct BufferStaticLocation {
    uint32_t index, bit_offset;
};

BufferShiftedWord lower_buffer_raw_shift(SpirvCompute& b, uint32_t addr, uint32_t dw0);
uint32_t lower_buffer_raw_join(SpirvCompute& b, uint32_t joined, uint32_t dw1, uint32_t shift);
uint32_t lower_buffer_raw_field(SpirvCompute& b, uint32_t joined, uint32_t raw_bits,
                                bool raw_signed);
uint32_t lower_buffer_packed_field(SpirvCompute& b, uint32_t dw, uint32_t sk, bool packed_10_11_11,
                                   bool is_uint, bool is_sint, bool is_snorm);
BufferComponentLocation lower_buffer_half_location(SpirvCompute& b, uint32_t addr, uint32_t sk);
uint32_t lower_buffer_half_join(SpirvCompute& b, uint32_t joined, uint32_t dw1, uint32_t shift);
uint32_t lower_buffer_half_field(SpirvCompute& b, uint32_t joined);
BufferComponentLocation lower_buffer_dynamic_location(SpirvCompute& b, uint32_t addr, uint32_t sk,
                                                      uint32_t comp_bytes);
uint32_t lower_buffer_dynamic_join(SpirvCompute& b, uint32_t joined, uint32_t dw1, uint32_t shift);
uint32_t lower_buffer_dynamic_field(SpirvCompute& b, uint32_t joined, uint32_t comp_bytes,
                                    bool dyn_norm, bool is_snorm, float norm, bool is_sint);
BufferStaticLocation lower_buffer_static_location(SpirvCompute& b, uint32_t idx, uint32_t sk,
                                                  uint32_t comp_bytes);
uint32_t lower_buffer_static_field(SpirvCompute& b, uint32_t dw, uint32_t boff, uint32_t comp_bytes,
                                   bool is_half, bool is_uint, bool is_sint, bool is_snorm,
                                   float norm);

uint32_t lower_buffer_low_shift(SpirvCompute& b, uint32_t word, uint32_t shift);
uint32_t lower_buffer_next_dword(SpirvCompute& b, uint32_t index);

} // namespace prosper::gpu::ir
