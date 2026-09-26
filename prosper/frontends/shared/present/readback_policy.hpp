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

// WHY a color target slot wants a CPU readback -- the reason, not just the verdict.
//
// Readback is the dominant cost inside a backend call (measured 2026-09-26 on The Messenger:
// 68.5% of `measured` on prosper-app, 78.5% under tools/screenshot) and it forces one of every two
// flushes, so it is also what makes the submit blocking. Deciding what to do about that needs to
// distinguish the three ways a slot arrives here, because they have nothing in common:
// a missing target is a caller-shape question, an explicit request is somebody downstream needing
// bytes, and a bound-but-non-persistent target is a residency question. A bare bool cannot say
// which, and the aggregate cost cannot be attributed without it.
enum class ColorReadbackReason : unsigned char {
    NotWanted = 0,        // the slot is GPU-resident and nothing needs its bytes on the CPU
    NoColorTarget,        // no color target object at all -- the caller passed none
    ExplicitRequest,      // the caller or a split carrier asked for it
    BoundNonPersistent,   // a bound target that is not persistent, so its pixels are not retained
};

// When target_readback is true, readback was explicitly requested (by the caller or a split carrier).
// When target_readback is false, an unbound target (persistent_id == 0) never requests readback,
// avoiding tens of megabytes of staging copies for unallocated host memory. A bound target
// (persistent_id != 0) requests readback if it is non-persistent.
constexpr ColorReadbackReason color_target_readback_reason(bool has_color_target,
                                                           uint64_t persistent_id,
                                                           bool persistent_color,
                                                           bool target_readback) {
    if (!has_color_target) return ColorReadbackReason::NoColorTarget;
    if (target_readback) return ColorReadbackReason::ExplicitRequest;
    return persistent_id != 0 && !persistent_color ? ColorReadbackReason::BoundNonPersistent
                                                   : ColorReadbackReason::NotWanted;
}

// The verdict, expressed in terms of the reason so the two can never disagree. This used to be the
// primary definition with the reasons implicit in its control flow; stating the reason first means
// a future edit to the policy cannot change what is counted without changing what is decided.
constexpr bool is_color_target_readback_wanted(bool has_color_target,
                                               uint64_t persistent_id,
                                               bool persistent_color,
                                               bool target_readback) {
    return color_target_readback_reason(has_color_target, persistent_id, persistent_color,
                                        target_readback) != ColorReadbackReason::NotWanted;
}

constexpr const char* color_readback_reason_name(ColorReadbackReason reason) {
    switch (reason) {
        case ColorReadbackReason::NotWanted: return "not-wanted";
        case ColorReadbackReason::NoColorTarget: return "no-color-target";
        case ColorReadbackReason::ExplicitRequest: return "explicit-request";
        case ColorReadbackReason::BoundNonPersistent: return "bound-non-persistent";
    }
    return "unknown";
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
