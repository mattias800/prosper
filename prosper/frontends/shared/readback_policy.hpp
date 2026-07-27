#pragma once

namespace prosper::frontend {

// Intermediate scanouts normally stay GPU-resident until the final presentation callback. The final
// scanout can do the same when the app consumes the renderer image directly; if that handoff fails,
// the completed persistent target is still available for the existing on-demand CPU fallback.
// A non-direct consumer later in the same submit is different: it needs authoritative bytes before
// the batch is submitted, so deferring there would make it observe the previous submit.
constexpr bool can_defer_scanout_readback(bool phase_allows_defer,
                                           bool final_gpu_present,
                                           bool defer_scanout,
                                           bool cpu_needed_same_batch) {
    return (phase_allows_defer || final_gpu_present) && defer_scanout &&
           !cpu_needed_same_batch;
}

} // namespace prosper::frontend
