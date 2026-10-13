// Phase-specific guest-CFG witness bounds. Observation is refused for guest GDS users.
#pragma once
#include <cstddef>
#include <cstdint>

namespace prosper::gpu {
uint32_t emitted_loop_trip_bound(uint64_t program_address, uint32_t phase, uint32_t start_pc,
                                 uint32_t end_pc, const uint32_t* code, size_t dwords);
} // namespace prosper::gpu
