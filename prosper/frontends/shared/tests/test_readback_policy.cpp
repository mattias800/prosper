#include "shared/present/readback_policy.hpp"

#include <cstdio>

using prosper::frontend::can_defer_scanout_readback;
using prosper::frontend::is_color_target_readback_wanted;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

int main() {
    CHECK(can_defer_scanout_readback(true, false, true, false));
    CHECK(can_defer_scanout_readback(false, true, true, false));

    // #1295: a same-submit 1D/cube/storage consumer cannot use the direct GPU target binding.
    // Deferring its producer's readback makes that consumer decode stale guest bytes.
    CHECK(!can_defer_scanout_readback(true, false, true, true));
    CHECK(!can_defer_scanout_readback(false, false, true, false));
    CHECK(!can_defer_scanout_readback(true, false, false, false));

    // Unbound color targets (persistent_id == 0) must never trigger CPU readbacks,
    // even when target_readback is flagged true.
    CHECK(!is_color_target_readback_wanted(true, 0, false, true));
    CHECK(!is_color_target_readback_wanted(true, 0, true, true));
    CHECK(!is_color_target_readback_wanted(true, 0, false, false));
    CHECK(!is_color_target_readback_wanted(true, 0, true, false));

    // Bound color targets (persistent_id != 0) trigger readback when non-persistent or flagged.
    CHECK(is_color_target_readback_wanted(true, 0x1234, false, false));
    CHECK(is_color_target_readback_wanted(true, 0x1234, true, true));
    CHECK(!is_color_target_readback_wanted(true, 0x1234, true, false));

    // When there is no color target struct at all, default readback behavior is preserved.
    CHECK(is_color_target_readback_wanted(false, 0, false, false));

    if (!failures) std::printf("readback_policy: OK\n");
    return failures ? 1 : 0;
}
