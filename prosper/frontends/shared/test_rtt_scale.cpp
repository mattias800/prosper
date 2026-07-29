// Unit test for rtt_integer_upscale_factor (compute-side renderer-owned-RTT scale detection).
// The factor is used to tell a PROSPER_RENDER_SCALE downscale (exact, equal-axis integer multiple ->
// upscale the cached snapshot) from a genuine view alias at a reused base (non-integer / unequal ->
// fall back to guest backing). Getting this wrong either blanks compute composites under scale>1 or
// scales a stale, wrong-view snapshot, so the boundary is pinned here.

#include "rtt_scale.hpp"
#include "rtt_authority.hpp"

#include <cstdio>

using prosper::frontend::rtt_integer_upscale_factor;
using prosper::frontend::rtt_direct_import_compatible;
using prosper::frontend::rtt_gpu_seed_import_extent_compatible;
using prosper::frontend::rtt_sampled_extent_compatible;
using prosper::frontend::rtt_scaled_axis;
using prosper::frontend::rtt_scaled_extent_compatible;
using prosper::frontend::LiveRttAuthority;
using prosper::frontend::live_rtt_authority;
using prosper::frontend::live_rtt_gpu_importable;

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

    if (failures == 0) std::printf("rtt_scale: OK\n");
    return failures == 0 ? 0 : 1;
}
