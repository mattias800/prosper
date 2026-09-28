// capture_renderer_policy.hpp - renderer residency policy for detailed GPU capture.
#pragma once

namespace prosper::frontend {

// Immediate timeline capture needs authoritative CPU pixels throughout the run. A validated,
// nonzero AFTER_COMPUTE gate does not: it remains inert before the semantic phase and materializes
// referenced live targets through the capture RTT reader once armed.
constexpr bool timeline_capture_allows_persistent_targets(
    bool timeline_capture_requested, bool after_compute_gated) {
    return !timeline_capture_requested || after_compute_gated;
}

// Everything that decides, once at renderer start, whether intermediate colour targets stay
// GPU-resident (the production path) or are read back to the CPU after every pass.
struct LiveColorTargetResidencyInputs {
    bool per_target = true;                 // not PROSPER_RTT_SINGLE_TARGET
    bool opted_out = false;                 // PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS
    bool timeline_capture_permits = true;   // timeline_capture_allows_persistent_targets()
    // PROSPER_GPU_CAPTURE (the one-shot .prgcap). Deliberately NOT an input to the decision below:
    // everything a capsule needs is obtained on demand for the one selected submit -- referenced
    // targets through the capture RTT-seed reader, which materializes a GPU-resident target, and the
    // output oracle through gpu_present_allowed_during_capture() -- exactly as the interactive F9
    // capture has always done on this path. Before #3895 it forced readback from boot, so a
    // PROSPER_GPU_CAPTURE_AFTER run rendered through different code, at a different speed, for its
    // whole length, and a capture of a production-only defect showed a different defect (#3890).
    bool gpu_capture_requested = false;
    bool replay_export_rtt = false;         // PROSPER_GPU_REPLAY_EXPORT_RTT
    bool replay_rtt_seeds = false;          // PROSPER_GPU_REPLAY_RTT_SEEDS
    bool replay_live_targets = false;       // PROSPER_GPU_REPLAY_LIVE_TARGETS
    // Per-pass pixel diagnostics (PROSPER_DUMP_SAMPLED_RTT, PROSPER_DUMP_RTGROUPS[_RGBA],
    // PROSPER_DUMP_DRAWSTEPS, PROSPER_RESOURCE_HASH_DIM, PROSPER_TARGET_STEP_HASH_DIM,
    // PROSPER_RTTLOG). These read CPU pixels at every pass and still use the readback path.
    bool per_pass_pixel_diagnostic = false;
};

constexpr bool live_color_targets_enabled(const LiveColorTargetResidencyInputs& in) {
    return in.per_target && !in.opted_out && in.timeline_capture_permits &&
           !in.replay_export_rtt && (!in.replay_rtt_seeds || in.replay_live_targets) &&
           !in.per_pass_pixel_diagnostic;
}

// A submit whose one-shot capture is pending needs a CPU copy of its presented frame: the capsule
// records it as the output oracle (expected_output_hash) and an output-triggered candidate is
// accepted or rejected on it. prosper-app's GPU present hands the final image to the swapchain
// without a CPU copy, so for exactly those submits the renderer presents through the CPU path, which
// materializes the scanout target on demand. Every other submit keeps GPU present.
constexpr bool gpu_present_allowed_during_capture(bool gpu_present_active,
                                                  bool capture_needs_cpu_output) {
    return gpu_present_active && !capture_needs_cpu_output;
}

} // namespace prosper::frontend
