// capture_compute_policy.hpp - compute shader portability policy for detailed GPU capture.
#pragma once

namespace prosper::gpu {

// Immediate timeline capture must compile device-independent storage-image paths from startup.
// A validated nonzero AFTER_COMPUTE request is inert until its semantic gate arms, at which point
// the compile key switches to the portable variant for the gate submit and all later capture work.
constexpr bool timeline_capture_requires_portable_compute(
    bool timeline_capture_requested, bool after_compute_gated, bool after_compute_armed) {
    return timeline_capture_requested && (!after_compute_gated || after_compute_armed);
}

} // namespace prosper::gpu
