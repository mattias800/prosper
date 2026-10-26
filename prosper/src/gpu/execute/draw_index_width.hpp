#pragma once
#include <cstdint>

namespace prosper::gpu {

bool index_buffer_is_unannounced_32bit(const uint16_t* p16, const uint32_t* p32, uint32_t n);
bool index_buffer_is_unannounced_32bit_high(const uint16_t* p16, const uint32_t* p32, uint32_t n,
                                            uint32_t vertex_upper_bound);
// #4637: an unannounced 32-bit list into a vertex pool LARGER than 16-bit indices can address,
// whose indices cross a 64 KiB window inside the draw. Every one of the `n` 32-bit readings must
// address a vertex the pool holds; see draw_index_width.cpp for what that does and does not rule out.
bool index_buffer_is_unannounced_32bit_in_pool(const uint32_t* p32, uint32_t n,
                                               uint32_t vertex_upper_bound);

} // namespace prosper::gpu
