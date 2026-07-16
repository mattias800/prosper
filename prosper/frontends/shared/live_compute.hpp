#pragma once
#include "gpu/gpu_execute.hpp"

#include <cstdint>
#include <vector>

namespace prosper::frontend {

// Pack one raw float32 channel to UNORM8 using the storage-image conversion contract. Kept public
// so the optimized scalar conversion can be checked directly against the previous lround path.
uint8_t storage_pack_unorm8(uint32_t float_bits);

// Execute already-realized compute items synchronously. Exposed for the production-backend test.
bool execute_live_compute_items(const std::vector<prosper::gpu::ComputeItem>& items);

// Register the synchronous Vulkan compute backend used by AGC submit processing.
void register_live_compute();

} // namespace prosper::frontend
