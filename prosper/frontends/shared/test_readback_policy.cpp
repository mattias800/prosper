#include "readback_policy.hpp"

#include <cstdio>

using prosper::frontend::can_defer_intermediate_scanout_readback;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

int main() {
    CHECK(can_defer_intermediate_scanout_readback(true, true, false));

    // #1295: a same-submit 1D/cube/storage consumer cannot use the direct GPU target binding.
    // Deferring its producer's readback makes that consumer decode stale guest bytes.
    CHECK(!can_defer_intermediate_scanout_readback(true, true, true));
    CHECK(!can_defer_intermediate_scanout_readback(false, true, false));
    CHECK(!can_defer_intermediate_scanout_readback(true, false, false));

    if (!failures) std::printf("readback_policy: OK\n");
    return failures ? 1 : 0;
}
