#pragma once

#include <cstdint>
#include <optional>

#include <vulkan/vulkan_core.h>

// Present policy for prosper-app (#1182). Real game boots normally share the renderer's Vulkan device;
// the private-device fallback hands frames across as a lock-free CPU shared_ptr copy. Neither path
// should let a BLOCKING swapchain acquire (`vkAcquireNextImageKHR(..., UINT64_MAX, ...)`) freeze event
// handling and stall the guest when the compositor releases no image (for example, while occluded or
// minimized). Bound the acquire and treat a timeout as a benign SKIP, not an error. This header holds
// the pure result classification so it is unit-testable without a device (Vulkan headers only).

namespace prosper::frontend {

// Real game boots prefer the renderer's shared-device scanout path: it avoids a full-frame GPU->CPU
// readback followed by a CPU->GPU upload. Adoption still validates all Vulkan prerequisites and falls
// back to the private-device path on failure. `PROSPER_APP_GPU_PRESENT=0` is the explicit recovery/A-B
// switch; test-pattern and no-game runs have no renderer-owned scanout to adopt.
constexpr bool request_gpu_present(const char* setting, bool test_pattern, bool has_game) {
    const bool explicitly_disabled = setting && setting[0] == '0' && setting[1] == '\0';
    return has_game && !test_pattern && !explicitly_disabled;
}

// F8's rendered-frame population is the CPU frame handoff count, not the number of successful
// swapchain presents. Shared-device GPU present deliberately skips that handoff. Until that producer
// exposes its own coherent completion counter, report the population as unavailable; substituting the
// app's host-present count would manufacture equality between two different clocks.
constexpr std::optional<uint64_t> rendered_frame_counter(bool gpu_present,
                                                         uint64_t cpu_frame_seq) {
    return gpu_present ? std::nullopt : std::optional<uint64_t>{cpu_frame_seq};
}

// Outcome of one present_frame attempt.
enum class PresentAttempt {
    presented,    // the rendered frame reached the swapchain
    skipped,      // no swapchain image available within the bounded acquire (occluded/minimized) — retry
    out_of_date,  // swapchain is stale/lost/unusable — recreate it before the next present
    failed,       // device/synchronization recovery failed — stop instead of waiting forever
};

// What present_frame should do after a BOUNDED vkAcquireNextImageKHR.
enum class AcquireAction { proceed, skip, recreate, fail };

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
    case VK_ERROR_DEVICE_LOST:
        return AcquireAction::fail;
    default:
        return AcquireAction::recreate;
    }
}

// Classify vkQueuePresentKHR. Only VK_SUCCESS is a clean present; SUBOPTIMAL/OUT_OF_DATE (and any error)
// request a swapchain rebuild. A present call does not time out, so there is no skip outcome here. This
// preserves the pre-#1182 contract, where present_frame returned false (→ recreate) on anything but
// VK_SUCCESS.
constexpr PresentAttempt classify_present(VkResult r) {
    if (r == VK_SUCCESS) return PresentAttempt::presented;
    if (r == VK_ERROR_DEVICE_LOST) return PresentAttempt::failed;
    return PresentAttempt::out_of_date;
}

// A failed submit normally abandons the acquired image and rebuilds the swapchain/synchronization.
// Device loss is different: no Vulkan object recreation on that device can succeed, so retrying only
// floods logs and spins the host CPU. Stop the frontend cleanly on that terminal result.
constexpr PresentAttempt classify_submit_failure(VkResult r) {
    return r == VK_ERROR_DEVICE_LOST ? PresentAttempt::failed : PresentAttempt::out_of_date;
}

} // namespace prosper::frontend
