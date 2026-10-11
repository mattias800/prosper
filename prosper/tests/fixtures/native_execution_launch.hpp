// Owned native compute fixtures declare the raw one-group launch that supplies physical
// Wave64 participation. A required host subgroup width alone cannot supply that authority.
#pragma once
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

namespace prosper::test {
inline gpu::ComputeShaderConfig native_execution_launch() {
    gpu::ComputeShaderConfig config;
    config.local_x = config.wave_size = config.native_subgroup_size = 64;
    complete_compute_launch(config);
    return config;
}
} // namespace prosper::test
