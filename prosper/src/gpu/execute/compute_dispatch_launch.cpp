#include "gpu/execute/compute_dispatch_launch.hpp"

#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/compute_dispatch_launch.hpp"

#include <array>
#include <cstddef>

namespace prosper::gpu {
ComputeDispatchLaunchState compute_dispatch_launch_state(const GpuState::Dispatch& d) {
    ComputeDispatchLaunchState out;
    out.initiator = d.modifier;
    out.known = ComputeDispatchLaunchState::kInitiator;
    // Ordered indirect resolution clears `indirect` but retains its argument address. Do not
    // authenticate the copied argument counts as original direct packet dimensions in this leaf.
    if (!d.indirect && !d.indirect_args_addr) {
        out.dimensions = {d.threads_x, d.threads_y, d.threads_z};
        out.known |= ComputeDispatchLaunchState::kDimX | ComputeDispatchLaunchState::kDimY |
                     ComputeDispatchLaunchState::kDimZ;
    }
    if (!d.state) return out;
    constexpr std::array<uint32_t, 3> registers{prosper::agc::Pm4::COMPUTE_NUM_THREAD_X,
                                                prosper::agc::Pm4::COMPUTE_NUM_THREAD_Y,
                                                prosper::agc::Pm4::COMPUTE_NUM_THREAD_Z};
    for (size_t axis = 0; axis < registers.size(); ++axis) {
        const auto found = d.state->sh.find(registers[axis]);
        if (found == d.state->sh.end()) continue;
        out.num_thread[axis] = found->second;
        out.known |= ComputeDispatchLaunchState::kNumThreadX << axis;
    }
    return out;
}
}   // namespace prosper::gpu
