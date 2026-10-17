// Resolved FLAT window arithmetic; memory access and proof remain with the caller.
#pragma once
#include <array>
#include <cstdint>
namespace prosper::gpu {
struct SpirvCompute;
struct FlatAccessInfo;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
uint32_t lower_flat_window_byte_offset(SpirvCompute& b, uint32_t addr_lo, uint32_t base_lo);
std::array<uint32_t, 2> lower_flat_window_component(SpirvCompute& b, uint32_t byte0,
                                                    const FlatAccessInfo& access, uint32_t c);
}   // namespace prosper::gpu::ir
