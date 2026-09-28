// capture_compute_policy.hpp - compute shader portability policy for detailed GPU capture.
#pragma once

#include <cstdint>

namespace prosper::gpu {

// Immediate timeline capture must compile device-independent storage-image paths from startup.
// A validated nonzero AFTER_COMPUTE request is inert until its semantic gate arms, at which point
// the compile key switches to the portable variant for the gate submit and all later capture work.
constexpr bool timeline_capture_requires_portable_compute(
    bool timeline_capture_requested, bool after_compute_gated, bool after_compute_armed) {
    return timeline_capture_requested && (!after_compute_gated || after_compute_armed);
}

// The one-shot PROSPER_GPU_CAPTURE stores portable compute modules too, but only the capture
// window needs them: from the moment its AFTER (invocation) and AFTER_MS (milliseconds since the
// first invocation past AFTER) gates are both met until its capsule has been written. Before #3895
// every compute of a capture run compiled the portable variant from boot, so a run with a late
// AFTER executed different shaders from a normal run for its whole length. The compile key carries
// native_storage_format_support, so a native module compiled before the window is never reused by a
// capture-bound dispatch. `legacy_readback` (PROSPER_GPU_CAPTURE_READBACK) restores the old
// whole-run behaviour as an A/B arm.
struct EnvCaptureWindow {
    bool requested = false;            // PROSPER_GPU_CAPTURE names a path
    bool legacy_readback = false;
    uint64_t invocations = 0;          // begin_requested_gpu_capture calls so far
    uint64_t after = 0;                // PROSPER_GPU_CAPTURE_AFTER
    bool clock_started = false;        // the AFTER_MS clock starts at the first call past AFTER
    uint64_t elapsed_ms = 0;
    uint64_t after_ms = 0;             // PROSPER_GPU_CAPTURE_AFTER_MS
    bool finished = false;             // the one-shot capsule has been written (or failed)
};

constexpr bool env_capture_requires_portable_compute(const EnvCaptureWindow& w) {
    if (!w.requested) return false;
    if (w.legacy_readback) return true;
    if (w.finished || w.invocations < w.after) return false;
    if (!w.after_ms) return true;
    return w.clock_started && w.elapsed_ms >= w.after_ms;
}

} // namespace prosper::gpu
