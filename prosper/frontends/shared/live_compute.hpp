#pragma once
#include "gpu/gpu_execute.hpp"

#include <vector>

namespace prosper::frontend {

// Execute already-realized compute items synchronously. Exposed for the production-backend test.
bool execute_live_compute_items(const std::vector<prosper::gpu::ComputeItem>& items);

// Register the synchronous Vulkan compute backend used by AGC submit processing.
void register_live_compute();

} // namespace prosper::frontend
