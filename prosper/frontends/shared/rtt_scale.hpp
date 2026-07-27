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
// NOT this view and must fall back). The configured render scale plus the renderer's nearest-integer
// extent rule provide that proof: the cached axes must equal round(native/scale) independently. This
// also covers legitimate targets whose dimensions are not divisible by the scale (1216/3 -> 405),
// which an exact integer-ratio check incorrectly sent through CPU readback.
namespace prosper::frontend {

constexpr uint32_t rtt_integer_upscale_factor(uint32_t dst_w, uint32_t dst_h,
                                              uint32_t src_w, uint32_t src_h) {
    if (!src_w || !src_h || dst_w <= src_w || dst_h <= src_h) return 0;
    if ((dst_w % src_w) != 0 || (dst_h % src_h) != 0) return 0;
    const uint32_t kx = dst_w / src_w, ky = dst_h / src_h;
    if (kx != ky || kx < 2) return 0;
    return kx;
}

constexpr uint32_t rtt_scaled_axis(uint32_t native, uint32_t render_scale) {
    if (!native || render_scale <= 1) return native;
    // Match live_renderer.cpp's (native * scaled_present + present/2) / present when the
    // presentation extent is reduced by exactly render_scale. The 64-bit addition avoids overflow
    // for synthetic test extents even though real image dimensions are much smaller.
    const uint32_t scaled = static_cast<uint32_t>(
        (static_cast<uint64_t>(native) + render_scale / 2u) / render_scale);
    return scaled ? scaled : 1u;
}

constexpr bool rtt_scaled_extent_compatible(uint32_t requested_w, uint32_t requested_h,
                                            uint32_t cached_w, uint32_t cached_h,
                                            uint32_t render_scale) {
    return requested_w && requested_h && cached_w && cached_h && render_scale > 1 &&
           rtt_scaled_axis(requested_w, render_scale) == cached_w &&
           rtt_scaled_axis(requested_h, render_scale) == cached_h;
}

// A sampled descriptor can bind the renderer's reduced-resolution image directly: normalized
// sampling naturally maps the guest view onto the smaller image. Exact extents are always valid;
// a mismatch is valid only when it was produced by the configured render scale. Storage images and
// unnormalized texel-fetch contracts must keep using exact extents at their call sites.
constexpr bool rtt_sampled_extent_compatible(uint32_t requested_w, uint32_t requested_h,
                                             uint32_t cached_w, uint32_t cached_h,
                                             uint32_t render_scale) {
    if (requested_w == cached_w && requested_h == cached_h) return true;
    return rtt_scaled_extent_compatible(
        requested_w, requested_h, cached_w, cached_h, render_scale);
}

// A native-format storage image cannot be bound at a different extent because imageLoad/imageStore
// use integer texel coordinates. It can, however, stay entirely on the GPU through a temporary
// native-resolution image when the renderer owns an exact render-scale version: upscale before the
// dispatch and downscale the result afterwards. This deliberately has the same strict geometry gate
// as scaled sampled imports so an unrelated alias at the same guest address is never bridged.
constexpr bool rtt_storage_bridge_extent_compatible(uint32_t requested_w, uint32_t requested_h,
                                                    uint32_t cached_w, uint32_t cached_h,
                                                    uint32_t render_scale) {
    return (requested_w != cached_w || requested_h != cached_h) &&
           rtt_scaled_extent_compatible(
               requested_w, requested_h, cached_w, cached_h, render_scale);
}

} // namespace prosper::frontend
