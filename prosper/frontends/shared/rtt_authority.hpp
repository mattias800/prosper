#pragma once

namespace prosper::frontend {

// The live RTT cache has two independently useful representations. A CPU snapshot with no valid
// persistent image is authoritative CPU data. When the persistent image is valid, any CPU snapshot
// is a mirror produced by an ordered readback (or the matching CPU result of the same render pass),
// so sampling the image remains safe. Guest writes erase the cache entry and CPU-only publications
// explicitly clear gpu_valid before an importer can observe them.
enum class LiveRttAuthority { none, cpu, gpu, mirrored };

constexpr LiveRttAuthority live_rtt_authority(bool gpu_valid, bool has_cpu_snapshot) {
    if (gpu_valid) return has_cpu_snapshot ? LiveRttAuthority::mirrored
                                           : LiveRttAuthority::gpu;
    return has_cpu_snapshot ? LiveRttAuthority::cpu : LiveRttAuthority::none;
}

constexpr bool live_rtt_gpu_importable(bool gpu_valid, bool has_cpu_snapshot) {
    const LiveRttAuthority authority = live_rtt_authority(gpu_valid, has_cpu_snapshot);
    return authority == LiveRttAuthority::gpu || authority == LiveRttAuthority::mirrored;
}

} // namespace prosper::frontend
