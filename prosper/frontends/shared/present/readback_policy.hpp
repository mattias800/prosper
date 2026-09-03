#pragma once

#include <cstdint>

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

// Determines whether a color target slot wants a CPU readback.
// An unbound target (persistent_id == 0) never requests readback, avoiding tens of megabytes of
// staging copies for unallocated host memory. A bound target requests readback if it is
// non-persistent or explicitly flagged for readback.
constexpr bool is_color_target_readback_wanted(bool has_color_target,
                                               uint64_t persistent_id,
                                               bool persistent_color,
                                               bool target_readback) {
    if (!has_color_target) return true;
    return persistent_id != 0 && (!persistent_color || target_readback);
}

} // namespace prosper::frontend
