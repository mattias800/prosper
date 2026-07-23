#pragma once
#include <cstdint>

// Shared helper for the compute-side renderer-owned-RTT (live target) lookup.
//
// When PROSPER_RENDER_SCALE>1, the live renderer renders a color target at 1/scale of the guest's
// native resolution (e.g. 480x270 for a 1920x1080 target). A later COMPUTE op that samples that same
// target at the native guest resolution therefore finds the cached snapshot at an EXACT integer
// downscale. Previously any dimension mismatch made the compute path fall back to the (often zero)
// guest backing, so compute-composited scenes read black under RENDER_SCALE>1 (Blue Prince PPSA25009
// renders its whole frame via a compute composite; at scale=4 it was fully blank).
//
// The mismatch must be distinguished from a genuine VIEW ALIAS at the same base address (Astro Bot
// renders a 960x540 target then dispatches a 1216x684 view at the same base — a stale snapshot that is
// NOT this view and must fall back). A render-scale downscale is the ONLY case where the requested size
// is an exact, EQUAL-on-both-axes integer multiple of the cached size; an alias is a non-integer or
// unequal ratio. Returns that factor k (>=2) when src*k == dst on both axes, else 0.
namespace prosper::frontend {

constexpr uint32_t rtt_integer_upscale_factor(uint32_t dst_w, uint32_t dst_h,
                                              uint32_t src_w, uint32_t src_h) {
    if (!src_w || !src_h || dst_w <= src_w || dst_h <= src_h) return 0;
    if ((dst_w % src_w) != 0 || (dst_h % src_h) != 0) return 0;
    const uint32_t kx = dst_w / src_w, ky = dst_h / src_h;
    if (kx != ky || kx < 2) return 0;
    return kx;
}

} // namespace prosper::frontend
