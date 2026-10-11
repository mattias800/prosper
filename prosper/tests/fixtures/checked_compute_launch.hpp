// Project-owned direct group-count launches. Tests declare real raw packet/register words;
// normalized geometry and a required host subgroup width are not entry authority.
#pragma once
#include "gpu/recompiler/compute_dispatch_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>

namespace prosper::test {
inline void complete_compute_launch(prosper::gpu::ComputeShaderConfig& c, uint32_t groups_x = 1,
                                    uint32_t groups_y = 1, uint32_t groups_z = 1) {
    c.threads_x = c.local_x * groups_x;
    c.threads_y = c.local_y * groups_y;
    c.threads_z = c.local_z * groups_z;
    c.dispatch_launch.known = prosper::gpu::ComputeDispatchLaunchState::kAll;
    c.dispatch_launch.initiator = 1u | (c.wave_size == 32 ? 1u << 15u : 0u);
    c.dispatch_launch.num_thread = {c.local_x, c.local_y, c.local_z};
    c.dispatch_launch.dimensions = {groups_x, groups_y, groups_z};
}
inline void bind_checked_compute_launch(prosper::gpu::SpirvCompute& b,
                                        const prosper::gpu::ComputeShaderConfig& c) {
    // Keep the exact tests buildable against literal original shipping production for red controls.
    // The original compiler has no retained builder profile and must fail the new negative cases.
#if __has_include("gpu/recompiler/rdna2_native_launch.hpp")
    b.guest_compute_launch = prosper::gpu::compute_dispatch_launch_profile(c);
#else
    (void)b;
    (void)c;
#endif
}
}   // namespace prosper::test
