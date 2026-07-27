#pragma once

#include <cstdint>
#include <limits>

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

// Guest GPU writes can replace a renderer-owned target through either its ordinary color plane or
// the separate DCC control surface described by a sampled-image descriptor. Color-plane writes
// discard the whole cached identity. A metadata write keeps the identity/extent so a self-contained
// fast-clear code can be materialized, but makes every retained pixel representation stale.
enum class LiveRttGuestWriteEffect { none, color_plane, dcc_metadata };

constexpr uint64_t live_rtt_range_end(uint64_t address, uint64_t bytes) {
    return bytes > std::numeric_limits<uint64_t>::max() - address
        ? std::numeric_limits<uint64_t>::max() : address + bytes;
}

constexpr bool live_rtt_ranges_overlap(uint64_t lhs_address, uint64_t lhs_bytes,
                                       uint64_t rhs_address, uint64_t rhs_bytes) {
    if (!lhs_address || !lhs_bytes || !rhs_address || !rhs_bytes) return false;
    return lhs_address < live_rtt_range_end(rhs_address, rhs_bytes) &&
           rhs_address < live_rtt_range_end(lhs_address, lhs_bytes);
}

constexpr LiveRttGuestWriteEffect live_rtt_guest_write_effect(
    uint64_t target_address, uint64_t target_bytes,
    uint64_t metadata_address, uint64_t metadata_bytes,
    uint64_t write_address, uint64_t write_bytes) {
    if (live_rtt_ranges_overlap(target_address, target_bytes, write_address, write_bytes))
        return LiveRttGuestWriteEffect::color_plane;
    if (live_rtt_ranges_overlap(metadata_address, metadata_bytes, write_address, write_bytes))
        return LiveRttGuestWriteEffect::dcc_metadata;
    return LiveRttGuestWriteEffect::none;
}

} // namespace prosper::frontend
