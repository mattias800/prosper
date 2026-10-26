#pragma once
#include <cstdint>

namespace prosper::gpu {

bool index_buffer_is_unannounced_32bit(const uint16_t* p16, const uint32_t* p32, uint32_t n);
bool index_buffer_is_unannounced_32bit_high(const uint16_t* p16, const uint32_t* p32, uint32_t n,
                                            uint32_t vertex_upper_bound);
// #4637: a bounded 32-bit candidate whose indices may cross a 64 KiB window. At least three
// quarters of adjacent deltas must be below 0x8000; a large V# bound alone is insufficient.
// This heuristic still has ambiguous 16-bit patterns; see the implementation's KNOWN LIMIT.
bool index_buffer_is_unannounced_32bit_in_pool(const uint32_t* p32, uint32_t n,
                                               uint32_t vertex_upper_bound);

} // namespace prosper::gpu
