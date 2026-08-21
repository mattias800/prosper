#pragma once
// fps_hud.hpp — WHAT the `--fps` counter says, separated from HOW it is drawn.
//
// `fps_overlay.{hpp,cpp}` owns ImGui, a render pass and swapchain framebuffers, none of which can
// run in the core test suite. The decision this file holds -- which numbers appear, in which order,
// and when the reader is warned that the presented rate is not the title's rate -- is the part that
// can be wrong in a way nobody notices, so it lives here, pure and asserted.
//
// The ordering is the load-bearing detail and it is not cosmetic: a reader glancing at a HUD takes
// the first number. If that number were the PRESENTED rate, the counter would read ~60 fps for a
// title whose picture is completely frozen, because the renderer re-publishes its retained frame
// through the ordinary publish path (gpu/present/present_frame_rate.hpp; instrument trap 90; the
// R-Type Delta regression #2783). So the distinct rate is always first, and the presented rate is
// always labelled.

#include "gpu/present/present_frame_rate.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace prosper::frontend {

// Re-base the rolling window? The HUD reports a ROLLING rate rather than a run average so a title
// that ran well for a minute and then collapsed shows the collapse rather than averaging it away.
inline bool fps_window_due(double window_start_seconds, double now_seconds, double window_seconds) {
    return now_seconds - window_start_seconds >= window_seconds;
}

// The HUD's lines, newest measurement first.
//
// `distinct_total` is the run's cumulative distinct-frame count, shown so the rate has a population
// beside it: "0.4 fps" from three frames is not the same claim as "0.4 fps" from four hundred.
inline std::vector<std::string> fps_hud_lines(const prosper::gpu::FrameRate& rate,
                                              uint32_t width, uint32_t height,
                                              uint64_t distinct_total) {
    char headline[96], detail[160];
    if (!rate.measured) {
        std::snprintf(headline, sizeof headline, "-- fps");
        std::snprintf(detail, sizeof detail, "no frames published yet");
        return {headline, detail};
    }
    std::snprintf(headline, sizeof headline, "%.1f fps", rate.distinct_fps);
    std::snprintf(detail, sizeof detail, "%.1f presented   %ux%u   %llu frames",
                  rate.presented_fps, width, height,
                  static_cast<unsigned long long>(distinct_total));
    std::vector<std::string> lines{headline, detail};
    // Stated in words, not left to be inferred from two numbers. The whole failure mode is a reader
    // taking the healthy-looking number, so the warning has to be prose rather than arithmetic.
    if (prosper::gpu::frame_rate_is_mostly_retained(rate))
        lines.emplace_back("RETAINED FRAME - not this title's rate");
    return lines;
}

} // namespace prosper::frontend
