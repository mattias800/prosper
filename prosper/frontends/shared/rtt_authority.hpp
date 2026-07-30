#pragma once

#include <cstddef>
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

// Compute must override guest backing only while the renderer actually owns a current pixel
// representation. A cache entry can survive with identity/metadata alone after an invalidation;
// treating that stale shell as authoritative makes a later dispatch reject valid guest bytes.
constexpr bool live_rtt_compute_authoritative(bool gpu_valid, bool has_cpu_snapshot) {
    return live_rtt_authority(gpu_valid, has_cpu_snapshot) != LiveRttAuthority::none;
}

constexpr bool live_rtt_cpu_snapshot_matches(uint32_t width, uint32_t height,
                                             uint32_t bytes_per_pixel,
                                             size_t snapshot_bytes) {
    if (!width || !height || !bytes_per_pixel) return false;
    const uint64_t pixels = static_cast<uint64_t>(width) * height;
    if (pixels > std::numeric_limits<uint64_t>::max() / bytes_per_pixel) return false;
    return pixels * bytes_per_pixel == snapshot_bytes;
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
