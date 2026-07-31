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

} // namespace prosper::frontend
