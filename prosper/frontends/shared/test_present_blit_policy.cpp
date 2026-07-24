#include "present_blit_policy.hpp"

#include <cstdio>

using prosper::frontend::present_blit_wait_completed;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

int main() {
    CHECK(present_blit_wait_completed(VK_SUCCESS));

    // #1303: a timeout or device error must not publish a slot whose blit may still be in flight.
    CHECK(!present_blit_wait_completed(VK_TIMEOUT));
    CHECK(!present_blit_wait_completed(VK_ERROR_DEVICE_LOST));

    if (!failures) std::printf("present_blit_policy: OK\n");
    return failures ? 1 : 0;
}
