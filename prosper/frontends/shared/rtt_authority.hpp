#pragma once

#include <cstdint>

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

// A mirrored guest write may re-authorize only the exact renderer image that actually received the
// same completed GPU result. Address overlap or an equal byte footprint is insufficient: reused
// allocations can expose different shapes/formats at one base address.
constexpr bool live_rtt_mirror_identity_matches(
    uint64_t target_addr, uint32_t target_width, uint32_t target_height, uint32_t target_format,
    uint64_t mirror_addr, uint32_t mirror_width, uint32_t mirror_height, uint32_t mirror_format) {
    return target_addr && target_width && target_height &&
        target_addr == mirror_addr && target_width == mirror_width &&
        target_height == mirror_height && target_format == mirror_format;
}

constexpr bool live_rtt_compute_mirror_eligible(
    bool exact_import_seed, bool imported_transfer_dst, bool disabled) {
    return exact_import_seed && imported_transfer_dst && !disabled;
}

} // namespace prosper::frontend
