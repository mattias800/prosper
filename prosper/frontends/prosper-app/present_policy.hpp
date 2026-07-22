#pragma once

#include <vulkan/vulkan_core.h>

// Present policy for prosper-app (#1182). The app owns its OWN Vulkan device/queue, separate from the
// core live renderer's headless device (main.cpp: "two Vulkan contexts by design"). The guest↔app frame
// handoff is a lock-free CPU shared_ptr copy (present_acquire_rendered_frame), so the guest is NOT
// back-pressured by the app present at the Vulkan level. The one place the app can hurt the guest — and
// itself — is a BLOCKING swapchain acquire: `vkAcquireNextImageKHR(..., UINT64_MAX, ...)` blocks the app
// main thread indefinitely when the compositor releases no image (window occluded/minimized). That
// freezes visible output, stops SDL event/close handling, and (on the shared physical GPU) can stall the
// guest. The fix is to bound the acquire and treat a timeout as a benign SKIP, not an error: the main
// loop stays responsive and the guest keeps running. This header holds the pure result-classification so
// it is unit-testable without a device (Vulkan headers only, no loader).

namespace prosper::frontend {

// Outcome of one present_frame attempt.
enum class PresentAttempt {
    presented,    // the rendered frame reached the swapchain
    skipped,      // no swapchain image available within the bounded acquire (occluded/minimized) — retry
    out_of_date,  // swapchain is stale/lost/unusable — recreate it before the next present
};

// What present_frame should do after a BOUNDED vkAcquireNextImageKHR.
enum class AcquireAction { proceed, skip, recreate };

// Classify vkAcquireNextImageKHR under a bounded (non-infinite) timeout.
//  - VK_SUCCESS / VK_SUBOPTIMAL_KHR: an image was acquired (SUBOPTIMAL still yields a usable image) —
//    proceed to blit + present. SUBOPTIMAL is handled at present time, matching the pre-#1182 behavior.
//  - VK_TIMEOUT / VK_NOT_READY: no image was released in time — the window is occluded/minimized. This
//    is the crux of #1182: skip this frame and keep the main loop responsive; do NOT treat it as an
//    error and do NOT recreate the swapchain.
//  - everything else (OUT_OF_DATE, DEVICE_LOST, SURFACE_LOST, ...): the swapchain must be rebuilt.
constexpr AcquireAction classify_acquire(VkResult r) {
    switch (r) {
    case VK_SUCCESS:
    case VK_SUBOPTIMAL_KHR:
        return AcquireAction::proceed;
    case VK_TIMEOUT:
    case VK_NOT_READY:
        return AcquireAction::skip;
    default:
        return AcquireAction::recreate;
    }
}

// Classify vkQueuePresentKHR. Only VK_SUCCESS is a clean present; SUBOPTIMAL/OUT_OF_DATE (and any error)
// request a swapchain rebuild. A present call does not time out, so there is no skip outcome here. This
// preserves the pre-#1182 contract, where present_frame returned false (→ recreate) on anything but
// VK_SUCCESS.
constexpr PresentAttempt classify_present(VkResult r) {
    return r == VK_SUCCESS ? PresentAttempt::presented : PresentAttempt::out_of_date;
}

} // namespace prosper::frontend
