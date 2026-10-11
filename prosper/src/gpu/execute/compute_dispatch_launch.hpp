#pragma once

#include "gpu/pm4/command_processor.hpp"
#include "gpu/recompiler/compute_dispatch_launch.hpp"

namespace prosper::gpu {
// Read only the retained packet and its immutable producing register snapshot. Missing registers
// and unresolved indirect dimensions remain independently unknown; normalized defaults grant no
// original guest EXEC authority.
ComputeDispatchLaunchState compute_dispatch_launch_state(const GpuState::Dispatch&);
} // namespace prosper::gpu
