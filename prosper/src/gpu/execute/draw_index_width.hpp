#pragma once
#include <cstdint>

namespace prosper::gpu {

bool index_buffer_is_unannounced_32bit(const uint16_t* p16, const uint32_t* p32, uint32_t n);
bool index_buffer_is_unannounced_32bit_high(const uint16_t* p16, const uint32_t* p32, uint32_t n,
                                            uint32_t vertex_upper_bound);

} // namespace prosper::gpu
