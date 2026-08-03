#pragma once

namespace prosper::frontend {

struct PerformanceTimingMode {
    bool measure = false;
    bool log = false;
};

// F8 may enable the clocks needed by its bounded structured artifact, but only the explicit
// PROSPER_RENDER_TIMING switch authorizes the existing high-volume periodic stderr summaries.
inline PerformanceTimingMode performance_timing_mode(bool environment_logging,
                                                      bool interactive_capture) {
    return {environment_logging || interactive_capture, environment_logging};
}

// A semantic submit may contain graphics spans separated by compute/DMA. Intermediate callbacks
// must retain the accumulated timing; only its final span consumes and clears that population.
inline bool should_reset_performance_timing_after_span(bool timing_enabled, bool final_span) {
    return timing_enabled && final_span;
}

template <typename Timing>
inline void reset_performance_timing_after_span(Timing& pending, bool timing_enabled,
                                                bool final_span) {
    if (should_reset_performance_timing_after_span(timing_enabled, final_span)) pending = {};
}

} // namespace prosper::frontend
