// Unit test for rtt_integer_upscale_factor (compute-side renderer-owned-RTT scale detection).
// The factor is used to tell a PROSPER_RENDER_SCALE downscale (exact, equal-axis integer multiple ->
// upscale the cached snapshot) from a genuine view alias at a reused base (non-integer / unequal ->
// fall back to guest backing). Getting this wrong either blanks compute composites under scale>1 or
// scales a stale, wrong-view snapshot, so the boundary is pinned here.

#include "rtt_scale.hpp"

#include <cstdio>

using prosper::frontend::rtt_integer_upscale_factor;

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

    if (failures == 0) std::printf("rtt_scale: OK\n");
    return failures == 0 ? 0 : 1;
}
