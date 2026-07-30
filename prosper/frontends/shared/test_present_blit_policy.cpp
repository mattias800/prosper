#include "present_blit_policy.hpp"

#include <cstdio>

using prosper::frontend::present_blit_wait_completed;
using prosper::frontend::present_blit_has_new_flip;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

int main() {
    CHECK(present_blit_wait_completed(VK_SUCCESS));

    // #1303: a timeout or device error must not publish a slot whose blit may still be in flight.
    CHECK(!present_blit_wait_completed(VK_TIMEOUT));
    CHECK(!present_blit_wait_completed(VK_ERROR_DEVICE_LOST));

    // Several graphics submits may build one guest frame. Only the first pre-flip image and a
    // newly flipped image are publishable; repeated submits at the same flip stay GPU-local.
    CHECK(present_blit_has_new_flip(UINT64_MAX, 0));
    CHECK(!present_blit_has_new_flip(0, 0));
    CHECK(present_blit_has_new_flip(0, 1));
    CHECK(!present_blit_has_new_flip(42, 42));
    CHECK(present_blit_has_new_flip(UINT64_MAX, UINT64_MAX));

    if (!failures) std::printf("present_blit_policy: OK\n");
    return failures ? 1 : 0;
}
