// Unit test for the prosper-app present policy (#1182): the classification that turns a bounded
// vkAcquireNextImageKHR / vkQueuePresentKHR result into an action. The pre-#1182 code waited on the
// acquire with UINT64_MAX and returned a bool (VK_SUCCESS → present, anything else → recreate). That
// made an occluded/minimized window (no image released) block the app main thread forever. The fix
// bounds the acquire and treats VK_TIMEOUT / VK_NOT_READY as a benign SKIP. These asserts pin that
// contract; the VK_TIMEOUT/VK_NOT_READY → skip cases FAIL against any implementation that maps a
// timeout to "recreate" or "present", which is exactly the behavior the fix introduces.

#include "present_policy.hpp"

#include <cstdio>

using namespace prosper::frontend;

static int failures = 0;
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);\
            ++failures;                                                        \
        }                                                                      \
    } while (0)

int main() {
    // GPU presentation is the normal game policy because adoption has a safe CPU fallback. Keep the
    // explicit zero override for driver diagnosis and never request it without a renderer-owned game.
    CHECK(request_gpu_present(nullptr, false, true));
    CHECK(request_gpu_present("1", false, true));
    CHECK(!request_gpu_present("0", false, true));
    CHECK(!request_gpu_present(nullptr, true, true));
    CHECK(!request_gpu_present(nullptr, false, false));

    // `present_frame_seq()` belongs only to the CPU handoff path. A direct-present app must not
    // serialize its flat CPU counter as a real zero-rate population, while the CPU path retains the
    // exact counter value (including a legitimate zero).
    CHECK(!rendered_frame_counter(true, 73).has_value());
    CHECK(rendered_frame_counter(false, 0) == 0);
    CHECK(rendered_frame_counter(false, 73) == 73);

    // Acquire: a usable image (SUCCESS or SUBOPTIMAL) → proceed to blit+present.
    CHECK(classify_acquire(VK_SUCCESS) == AcquireAction::proceed);
    CHECK(classify_acquire(VK_SUBOPTIMAL_KHR) == AcquireAction::proceed);

    // Acquire: the crux of #1182 — a bounded timeout on an occluded/minimized window is a SKIP,
    // never an error and never a swapchain rebuild.
    CHECK(classify_acquire(VK_TIMEOUT) == AcquireAction::skip);
    CHECK(classify_acquire(VK_NOT_READY) == AcquireAction::skip);

    // Acquire: stale surfaces rebuild, but a lost device is terminal and must not enter a rebuild loop.
    CHECK(classify_acquire(VK_ERROR_OUT_OF_DATE_KHR) == AcquireAction::recreate);
    CHECK(classify_acquire(VK_ERROR_DEVICE_LOST) == AcquireAction::fail);
    CHECK(classify_acquire(VK_ERROR_SURFACE_LOST_KHR) == AcquireAction::recreate);
    CHECK(classify_acquire(VK_ERROR_OUT_OF_HOST_MEMORY) == AcquireAction::recreate);

    // Present: only a clean VK_SUCCESS presents; SUBOPTIMAL/OUT_OF_DATE/errors request a rebuild,
    // preserving the pre-#1182 "return false on anything but VK_SUCCESS" contract.
    CHECK(classify_present(VK_SUCCESS) == PresentAttempt::presented);
    CHECK(classify_present(VK_SUBOPTIMAL_KHR) == PresentAttempt::out_of_date);
    CHECK(classify_present(VK_ERROR_OUT_OF_DATE_KHR) == PresentAttempt::out_of_date);
    CHECK(classify_present(VK_ERROR_DEVICE_LOST) == PresentAttempt::failed);

    // A recoverable submit failure replaces synchronization once; device loss stops immediately.
    CHECK(classify_submit_failure(VK_ERROR_OUT_OF_DATE_KHR) == PresentAttempt::out_of_date);
    CHECK(classify_submit_failure(VK_ERROR_DEVICE_LOST) == PresentAttempt::failed);

    if (failures == 0) std::printf("present_policy: OK\n");
    return failures == 0 ? 0 : 1;
}
