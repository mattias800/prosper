#pragma once

namespace prosper::frontend {

// Outcome of one present_frame attempt.
enum class PresentAttempt {
    presented,    // the rendered frame reached the swapchain
    skipped,      // no swapchain image available within the bounded acquire (occluded/minimized) — retry
    out_of_date,  // swapchain is stale/lost/unusable — recreate it before the next present
    failed,       // device/synchronization recovery failed — stop instead of waiting forever
};

// A staged GPU readback is only a snapshot of a proposed frame until the swapchain accepts it.
// Keep pending screenshots and authored/automatic snaps for a later attempt on every other result.
// Own the callback invocation here so a test can prove a rejected attempt never consumes one.
template <typename Capture>
constexpr bool dispatch_presented_capture(PresentAttempt attempt, bool staged_pixels_ready,
                                          Capture&& capture) {
    if (attempt != PresentAttempt::presented || !staged_pixels_ready) return false;
    capture();
    return true;
}

} // namespace prosper::frontend
