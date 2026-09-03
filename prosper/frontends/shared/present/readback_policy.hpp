#pragma once

#include <cstddef>
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
// When target_readback is true, readback was explicitly requested (by the caller or a split carrier).
// When target_readback is false, an unbound target (persistent_id == 0) never requests readback,
// avoiding tens of megabytes of staging copies for unallocated host memory. A bound target
// (persistent_id != 0) requests readback if it is non-persistent.
constexpr bool is_color_target_readback_wanted(bool has_color_target,
                                               uint64_t persistent_id,
                                               bool persistent_color,
                                               bool target_readback) {
    if (!has_color_target) return true;
    if (target_readback) return true;
    return persistent_id != 0 && !persistent_color;
}

// Calculates the required staging buffer size covering only the active readback slots (#3276).
// Sizing the allocation to max(offsets[slot] + bytes[slot]) over the selected slots captures
// the common case of unbound higher MRT slots without disturbing the absolute offsets.
template <size_t N, typename OffsetArray, typename BytesArray, typename WantedPred>
constexpr uint64_t compute_active_readback_bytes(size_t count,
                                                 const OffsetArray& offsets,
                                                 const BytesArray& bytes,
                                                 WantedPred&& is_wanted) {
    uint64_t max_extent = 0;
    const size_t limit = count < N ? count : N;
    for (size_t slot = 0; slot < limit; ++slot) {
        if (is_wanted(slot)) {
            const uint64_t extent = static_cast<uint64_t>(offsets[slot]) +
                                    static_cast<uint64_t>(bytes[slot]);
            if (extent > max_extent) max_extent = extent;
        }
    }
    return max_extent;
}

} // namespace prosper::frontend
