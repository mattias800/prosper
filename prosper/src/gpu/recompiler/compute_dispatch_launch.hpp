#pragma once

#include <array>
#include <cstdint>

namespace prosper::gpu {

// Observed dispatch words, not an all-lanes-active assertion. Each presence bit names one real
// producing word. Legacy/offline callers have no words; an unresolved indirect dispatch has no
// dimensions. Consumers must independently validate the hardware fields and normalized config.
struct ComputeDispatchLaunchState {
    static constexpr uint32_t kInitiator = 1u;
    static constexpr uint32_t kNumThreadX = 2u, kNumThreadY = 4u, kNumThreadZ = 8u;
    static constexpr uint32_t kDimX = 16u, kDimY = 32u, kDimZ = 64u;
    static constexpr uint32_t kAll = 127u;

    uint32_t known = 0;
    uint64_t initiator = 0;
    std::array<uint32_t, 3> num_thread{};
    std::array<uint32_t, 3> dimensions{};

    bool operator==(const ComputeDispatchLaunchState&) const = default;
};

struct ComputeShaderConfig;
// Prove that the observed packet launches complete local groups. This says nothing about a
// complete logical guest wave (a local group can contain 65 workers), subsequent guest EXEC, or
// current host participation. Unknown/partial/thread-dimension launches remain outside this scope.
bool compute_dispatch_has_complete_groups(const ComputeShaderConfig&);
} // namespace prosper::gpu
