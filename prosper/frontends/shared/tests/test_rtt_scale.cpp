// Unit test for rtt_integer_upscale_factor (compute-side renderer-owned-RTT scale detection).
// The factor is used to tell a PROSPER_RENDER_SCALE downscale (exact, equal-axis integer multiple ->
// upscale the cached snapshot) from a genuine view alias at a reused base (non-integer / unequal ->
// fall back to guest backing). Getting this wrong either blanks compute composites under scale>1 or
// scales a stale, wrong-view snapshot, so the boundary is pinned here.

#include "shared/rtt/rtt_scale.hpp"
#include "shared/rtt/rtt_authority.hpp"

#include <cstdio>

using prosper::frontend::rtt_integer_upscale_factor;
using prosper::frontend::rtt_direct_import_compatible;
using prosper::frontend::rtt_gpu_seed_import_extent_compatible;
using prosper::frontend::rtt_sampled_extent_compatible;
using prosper::frontend::rtt_scaled_axis;
using prosper::frontend::rtt_scaled_extent_compatible;
using prosper::frontend::LiveRttAuthority;
using prosper::frontend::live_rtt_authority;
using prosper::frontend::live_rtt_compute_authoritative;
using prosper::frontend::live_rtt_cpu_snapshot_matches;
using prosper::frontend::live_rtt_gpu_importable;
using prosper::frontend::live_rtt_uniform_uses_cpu_diagnostic_path;
using prosper::frontend::LiveRttGuestWriteEffect;
using prosper::frontend::live_rtt_guest_write_effect;
using prosper::frontend::live_rtt_color_footprint_bytes;
using prosper::frontend::live_rtt_complete_guest_overwrite;
using prosper::frontend::live_rtt_ranges_overlap;
using prosper::frontend::live_rtt_unpublished_volume_blocks_sample;
using prosper::frontend::live_rtt_compute_mirror_eligible;
using prosper::frontend::live_rtt_mirror_identity_matches;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

int main() {
    // PROSPER_RENDER_SCALE downscales: exact, equal on both axes -> the scale factor.
    CHECK(rtt_integer_upscale_factor(1920, 1080, 480, 270) == 4);   // scale=4 (Blue Prince)
    CHECK(rtt_integer_upscale_factor(1920, 1080, 960, 540) == 2);   // scale=2
    CHECK(rtt_integer_upscale_factor(3840, 2160, 1920, 1080) == 2);

    // Same size: no upscale.
    CHECK(rtt_integer_upscale_factor(1920, 1080, 1920, 1080) == 0);

    // Genuine view alias (Astro Bot 960x540 base reused for a 1216x684 view): non-integer -> 0.
    CHECK(rtt_integer_upscale_factor(1216, 684, 960, 540) == 0);

    // Unequal per-axis integer ratios (2x wide, 3x tall) are not a uniform downscale -> 0.
    CHECK(rtt_integer_upscale_factor(1920, 1620, 960, 540) == 0);

    // Non-divisible dimensions -> 0.
    CHECK(rtt_integer_upscale_factor(1000, 1080, 480, 270) == 0);

    // Smaller/degenerate requests never upscale.
    CHECK(rtt_integer_upscale_factor(480, 270, 1920, 1080) == 0);
    CHECK(rtt_integer_upscale_factor(1920, 1080, 0, 270) == 0);
    CHECK(rtt_integer_upscale_factor(0, 0, 0, 0) == 0);

    CHECK(rtt_sampled_extent_compatible(1920, 1080, 1920, 1080, 1, false));
    CHECK(rtt_sampled_extent_compatible(1920, 1080, 960, 540, 2, true));
    CHECK(!rtt_sampled_extent_compatible(1920, 1080, 960, 540, 2, false));
    CHECK(!rtt_sampled_extent_compatible(1920, 1080, 960, 540, 1, true));
    CHECK(!rtt_sampled_extent_compatible(1216, 684, 960, 540, 2, true));
    CHECK(!rtt_sampled_extent_compatible(1920, 1620, 960, 540, 2, true));
    // Aliased target reuse with incompatible extent (#3436):
    CHECK(!rtt_sampled_extent_compatible(1920, 1080, 504, 204, 1, false));
    CHECK(!rtt_sampled_extent_compatible(1920, 1080, 504, 204, 1, true));
    CHECK(!rtt_sampled_extent_compatible(1024, 1024, 504, 204, 1, false));
    CHECK(!rtt_sampled_extent_compatible(512, 256, 132, 36, 1, false));
    CHECK(rtt_direct_import_compatible(false, 1920, 1080, 1920, 1080, 1, false));
    CHECK(rtt_direct_import_compatible(false, 1920, 1080, 960, 540, 2, true));
    CHECK(!rtt_direct_import_compatible(false, 1920, 1080, 960, 540, 2, false));
    CHECK(!rtt_direct_import_compatible(true, 1920, 1080, 1920, 1080, 1, true));
    CHECK(rtt_gpu_seed_import_extent_compatible(1920, 1080, 1920, 1080));
    CHECK(!rtt_gpu_seed_import_extent_compatible(1920, 1080, 960, 540));
    CHECK(!rtt_gpu_seed_import_extent_compatible(0, 1080, 0, 1080));

    // Pass-local targets use nearest-integer division, so non-divisible native dimensions are still
    // valid renderer-owned images. This is common in Astro Bot's dynamic-resolution post chain.
    CHECK(rtt_scaled_axis(1216, 3) == 405);
    CHECK(rtt_scaled_axis(684, 3) == 228);
    CHECK(rtt_scaled_axis(1, 3) == 1);
    CHECK(rtt_scaled_extent_compatible(1216, 684, 405, 228, 3));
    CHECK(rtt_sampled_extent_compatible(1216, 684, 405, 228, 3, true));
    CHECK(!rtt_sampled_extent_compatible(1216, 684, 405, 228, 3, false));
    CHECK(!rtt_sampled_extent_compatible(1216, 684, 406, 228, 3, true));
    CHECK(!rtt_sampled_extent_compatible(1216, 684, 960, 540, 3, true));

    // A snapshot produced by an ordered readback mirrors a still-valid persistent GPU image; it
    // must not force a full GPU->CPU->GPU round trip on the next compute consumer. Conversely, a
    // CPU-only publication has gpu_valid=false and must never expose the stale image.
    CHECK(live_rtt_authority(true, false) == LiveRttAuthority::gpu);
    CHECK(live_rtt_authority(true, true) == LiveRttAuthority::mirrored);
    CHECK(live_rtt_authority(false, true) == LiveRttAuthority::cpu);
    CHECK(live_rtt_authority(false, false) == LiveRttAuthority::none);
    CHECK(live_rtt_gpu_importable(true, false));
    CHECK(live_rtt_gpu_importable(true, true));
    CHECK(!live_rtt_gpu_importable(false, true));
    CHECK(!live_rtt_gpu_importable(false, false));
    CHECK(live_rtt_compute_authoritative(true, false));
    CHECK(live_rtt_compute_authoritative(true, true));
    CHECK(live_rtt_compute_authoritative(false, true));
    CHECK(!live_rtt_compute_authoritative(false, false));
    CHECK(live_rtt_uniform_uses_cpu_diagnostic_path(true, true));
    CHECK(!live_rtt_uniform_uses_cpu_diagnostic_path(true, false));
    CHECK(!live_rtt_uniform_uses_cpu_diagnostic_path(false, true));
    CHECK(live_rtt_cpu_snapshot_matches(1920, 1080, 8, 1920ull * 1080 * 8));
    CHECK(!live_rtt_cpu_snapshot_matches(1920, 1080, 8, 1920ull * 1080 * 4));
    CHECK(!live_rtt_cpu_snapshot_matches(0, 1080, 8, 0));
    CHECK(!live_rtt_cpu_snapshot_matches(1920, 1080, 0, 0));
    CHECK(live_rtt_mirror_identity_matches(
        0x100000, 3840, 2160, 97, 0x100000, 3840, 2160, 97));
    CHECK(!live_rtt_mirror_identity_matches(
        0x100000, 3840, 2160, 97, 0x100000, 1920, 4320, 97));
    CHECK(!live_rtt_mirror_identity_matches(
        0x100000, 3840, 2160, 97, 0x100000, 3840, 2160, 37));
    CHECK(!live_rtt_mirror_identity_matches(
        0x100000, 3840, 2160, 97, 0x100800, 3840, 2160, 97));
    CHECK(live_rtt_compute_mirror_eligible(true, true, false));
    CHECK(!live_rtt_compute_mirror_eligible(false, true, false));
    CHECK(!live_rtt_compute_mirror_eligible(true, false, false));
    CHECK(!live_rtt_compute_mirror_eligible(true, true, true));

    // A compute fast-clear can touch only a target's DCC allocation. That must revoke both cached
    // pixel copies while preserving enough target identity to decode the clear. Ordinary color
    // writes retain the stronger erase-the-entry behavior, including if ranges overlap both planes.
    constexpr uint64_t target = 0x30f36d0000ull;
    constexpr uint64_t target_bytes = 3840ull * 2160ull * 4ull;
    constexpr uint64_t metadata = 0x304d780000ull;
    constexpr uint64_t metadata_bytes = 163840;
    CHECK(live_rtt_guest_write_effect(target, target_bytes, metadata, metadata_bytes,
                                      metadata, metadata_bytes) ==
          LiveRttGuestWriteEffect::dcc_metadata);
    CHECK(live_rtt_guest_write_effect(target, target_bytes, metadata, metadata_bytes,
                                      target + 4096, 64) ==
          LiveRttGuestWriteEffect::color_plane);
    CHECK(live_rtt_guest_write_effect(target, target_bytes, target + 1024, metadata_bytes,
                                      target + 2048, 64) ==
          LiveRttGuestWriteEffect::color_plane);
    CHECK(live_rtt_guest_write_effect(target, target_bytes, metadata, metadata_bytes,
                                      metadata + metadata_bytes, 1) ==
          LiveRttGuestWriteEffect::none);
    CHECK(live_rtt_guest_write_effect(target, target_bytes, metadata, metadata_bytes,
                                      0, metadata_bytes) ==
          LiveRttGuestWriteEffect::none);
    CHECK(live_rtt_guest_write_effect(
              std::numeric_limits<uint64_t>::max() - 31, 64,
              metadata, metadata_bytes,
              std::numeric_limits<uint64_t>::max() - 1, 1) ==
          LiveRttGuestWriteEffect::color_plane);
    // A 3D producer never wrote ordinary guest bytes. Ordered DMA must decline a read from its
    // last Z slice, or a range that starts just before the allocation and crosses into it.
    constexpr uint64_t volume_base = 0x70000000ull;
    constexpr uint64_t volume_bytes = live_rtt_color_footprint_bytes(64, 64, 4, 4);
    CHECK(volume_bytes == 65536u);
    CHECK(live_rtt_color_footprint_bytes(64, 64, 0, 4) == 16384u);
    // A later 2D pass at the same base can serve its 2D view, but cannot justify reading the
    // older volume's other slices from guest memory. A failed/partially invalidated 2D alias
    // cannot serve either shape until a proven allocation reset or new producer restores it.
    CHECK(!live_rtt_unpublished_volume_blocks_sample(volume_bytes, true, true, false));
    CHECK(live_rtt_unpublished_volume_blocks_sample(volume_bytes, true, false, true));
    CHECK(!live_rtt_unpublished_volume_blocks_sample(volume_bytes, false, false, true));
    CHECK(live_rtt_unpublished_volume_blocks_sample(volume_bytes, false, false, false));
    CHECK(!live_rtt_unpublished_volume_blocks_sample(0, true, false, false));
    CHECK(live_rtt_ranges_overlap(volume_base, volume_bytes,
                                  volume_base + 3u * 16384u, 4u));
    CHECK(live_rtt_ranges_overlap(volume_base, volume_bytes,
                                  volume_base - 1u, 2u));
    CHECK(!live_rtt_ranges_overlap(volume_base, volume_bytes,
                                   volume_base + volume_bytes, 1u));
    CHECK(!live_rtt_complete_guest_overwrite(
        volume_base, volume_bytes, volume_base + 3u * 16384u, 16384u));
    CHECK(live_rtt_complete_guest_overwrite(
        volume_base, volume_bytes, volume_base, volume_bytes));
    CHECK(!live_rtt_complete_guest_overwrite(
        volume_base, volume_bytes, volume_base, volume_bytes - 1u));
    CHECK(live_rtt_color_footprint_bytes(UINT32_MAX, UINT32_MAX, UINT32_MAX, 16) ==
          UINT64_MAX);

    if (failures == 0) std::printf("rtt_scale: OK\n");
    return failures == 0 ? 0 : 1;
}
