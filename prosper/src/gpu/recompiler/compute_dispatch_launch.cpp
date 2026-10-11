#include "gpu/recompiler/compute_dispatch_launch.hpp"

#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <cstddef>
#include <limits>

namespace prosper::gpu {
bool compute_dispatch_has_complete_groups(const ComputeShaderConfig& c) {
    // CONFIDENCE: HIGH. AMD's COMPUTE_DISPATCH_INITIATOR.PARTIAL_TG_EN is independent of
    // USE_THREAD_DIMENSIONS. With partial TG disabled, hardware ignores NUM_THREAD_PARTIAL;
    // normalized dimensions alone cannot establish this control or its producing register words.
    const auto& launch = c.dispatch_launch;
    constexpr uint64_t shader_enable = 1u, wave32 = 1u << 15u;
    // Ordinary shader-enable, FORCE_START_AT_000, ordered append/order, cache invalidations and
    // wave width. In particular RESTORE is not an original guest entry. Tunnel/reserved/unknown
    // controls and partial/thread-dimension modes are unproved, including every upper 32-bit flag.
    constexpr uint64_t ordinary_controls = 0x8c5du;
    if (launch.known != ComputeDispatchLaunchState::kAll || launch.initiator > UINT32_MAX ||
        !(launch.initiator & shader_enable) || (launch.initiator & ~ordinary_controls) ||
        c.exact_thread_extent || (c.wave_size != 32 && c.wave_size != 64) ||
        c.wave_size != ((launch.initiator & wave32) ? 32u : 64u))
        return false;
    const std::array<uint32_t, 3> local{c.local_x, c.local_y, c.local_z};
    const std::array<uint32_t, 3> threads{c.threads_x, c.threads_y, c.threads_z};
    for (size_t axis = 0; axis < local.size(); ++axis) {
        const uint32_t full = launch.num_thread[axis] & 0xffffu;
        const uint64_t extent = uint64_t{launch.dimensions[axis]} * full;
        if (!full || !launch.dimensions[axis] || full != local[axis] || extent > UINT32_MAX ||
            extent != threads[axis])
            return false;
    }
    const uint64_t xy = uint64_t{c.local_x} * c.local_y;
    return xy <= std::numeric_limits<uint32_t>::max() / c.local_z;
}
}   // namespace prosper::gpu
