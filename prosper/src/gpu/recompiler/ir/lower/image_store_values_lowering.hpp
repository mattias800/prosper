// Pure storage-image values; resource policy and architectural writes remain caller-owned.
#pragma once
#include <cstdint>
namespace prosper::gpu {
struct SpirvCompute;
namespace ir {
uint32_t lower_image_store_d16_word(SpirvCompute& builder, uint32_t packed, uint32_t half,
                                    bool unsigned_data);
void lower_image_storage_load_selects(SpirvCompute& b, const uint32_t out[4], const int channel[4],
                                      const uint32_t constant[4], uint32_t routed_out[4]);
void lower_image_storage_store_selects(SpirvCompute& b, const uint32_t vals[4], const int source[4],
                                       uint32_t routed[4]);
} // namespace ir
} // namespace prosper::gpu
