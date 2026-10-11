// Project-owned native Wave64 structural tests share the same declared launch fixture.
#pragma once
#include "gpu/recompiler/rdna2_to_spirv.hpp"

namespace prosper::test::native_structural {
using prosper::gpu::ComputeShaderConfig;
inline prosper::gpu::ComputeShaderConfig
wave64_config(const prosper::gpu::ComputeShaderConfig& parent) {
    ComputeShaderConfig wave64_compute_config = parent;
    wave64_compute_config.wave_size = 64;
    wave64_compute_config.native_subgroup_size = 64;
    wave64_compute_config.local_x = 64;
    return wave64_compute_config;
}
}   // namespace prosper::test::native_structural
