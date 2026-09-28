#include "shared/diagnostics/capture_renderer_policy.hpp"

#include <cstdio>

using prosper::frontend::gpu_present_allowed_during_capture;
using prosper::frontend::live_color_targets_enabled;
using prosper::frontend::LiveColorTargetResidencyInputs;
using prosper::frontend::timeline_capture_allows_persistent_targets;

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) {
            std::fprintf(stderr, "FAIL: %s\n", what);
            ++failures;
        }
    };

    check(timeline_capture_allows_persistent_targets(false, false),
          "normal rendering retains persistent targets");
    check(timeline_capture_allows_persistent_targets(true, true),
          "phase-gated timeline capture retains persistent targets");
    check(!timeline_capture_allows_persistent_targets(true, false),
          "immediate timeline capture uses the readback path");

    const LiveColorTargetResidencyInputs normal{};
    check(live_color_targets_enabled(normal), "a normal run keeps GPU-resident targets");

    // #3895: the one-shot capture reads back on demand and must not change the run's render path.
    LiveColorTargetResidencyInputs capture = normal;
    capture.gpu_capture_requested = true;
    check(live_color_targets_enabled(capture),
          "PROSPER_GPU_CAPTURE keeps GPU-resident targets (#3895)");

    // Negative controls: every input that still selects the readback path does so, including when
    // combined with a capture, so the capture exemption above is not a blanket "always live".
    LiveColorTargetResidencyInputs c = capture;
    c.opted_out = true;
    check(!live_color_targets_enabled(c), "the explicit opt-out still wins under capture");
    c = capture; c.per_pass_pixel_diagnostic = true;
    check(!live_color_targets_enabled(c), "per-pass pixel diagnostics still read back");
    c = capture; c.timeline_capture_permits = false;
    check(!live_color_targets_enabled(c), "immediate timeline capture still reads back");
    c = capture; c.per_target = false;
    check(!live_color_targets_enabled(c), "single-target compositing has no live targets");
    c = capture; c.replay_export_rtt = true;
    check(!live_color_targets_enabled(c), "replay RTT export still reads back");
    c = capture; c.replay_rtt_seeds = true;
    check(!live_color_targets_enabled(c), "replay seeding without replay parity reads back");
    c.replay_live_targets = true;
    check(live_color_targets_enabled(c), "replay seeding with replay parity stays live");

    check(gpu_present_allowed_during_capture(true, false), "GPU present outside a capture");
    check(!gpu_present_allowed_during_capture(true, true),
          "a capture submit presents through the CPU path so it has an output oracle");
    check(!gpu_present_allowed_during_capture(false, false), "no GPU present when inactive");
    check(!gpu_present_allowed_during_capture(false, true), "no GPU present when inactive (capture)");
    return failures ? 1 : 0;
}
