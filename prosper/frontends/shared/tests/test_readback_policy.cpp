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

    // Unbound color targets (persistent_id == 0) do not trigger CPU readbacks when unflagged,
    // avoiding reading back unallocated host memory when persistent_color is false.
    CHECK(!is_color_target_readback_wanted(true, 0, false, false));
    CHECK(!is_color_target_readback_wanted(true, 0, true, false));

    // Explicitly requested readbacks (e.g. split pass carry or caller request) are always honored.
    CHECK(is_color_target_readback_wanted(true, 0, false, true));
    CHECK(is_color_target_readback_wanted(true, 0, true, true));
    CHECK(is_color_target_readback_wanted(true, 0x1234, true, true));

    // Bound color targets (persistent_id != 0) trigger readback when non-persistent, and stay
    // GPU-resident when persistent.
    CHECK(is_color_target_readback_wanted(true, 0x1234, false, false));
    CHECK(!is_color_target_readback_wanted(true, 0x1234, true, false));

    // When there is no color target struct at all, default readback behavior is preserved.
    CHECK(is_color_target_readback_wanted(false, 0, false, false));

    // Active readback sizing (#3276): staging buffer is sized only to the maximum extent of
    // selected slots, while preserving absolute offsets.
    using prosper::frontend::compute_active_readback_bytes;
    const uint64_t offsets[4] = {0, 33u << 20, 66u << 20, 99u << 20};
    const uint64_t bytes[4] = {33u << 20, 33u << 20, 33u << 20, 33u << 20};

    // Slot 0 only selected (common case: unbound higher MRT slots): only 33 MB, not 132 MB.
    CHECK(compute_active_readback_bytes<4>(4, offsets, bytes, [](size_t s) { return s == 0; }) ==
          (33u << 20));

    // Slots 0 and 1 selected: 66 MB.
    CHECK(compute_active_readback_bytes<4>(4, offsets, bytes, [](size_t s) { return s <= 1; }) ==
          (66u << 20));

    // Slot 2 selected (with holes at slots 0 and 1): covers up to end of slot 2 (99 MB).
    CHECK(compute_active_readback_bytes<4>(4, offsets, bytes, [](size_t s) { return s == 2; }) ==
          (99u << 20));

    // All slots selected: full 132 MB.
    CHECK(compute_active_readback_bytes<4>(4, offsets, bytes, [](size_t) { return true; }) ==
          (132u << 20));

    // No slots selected: 0 bytes.
    CHECK(compute_active_readback_bytes<4>(4, offsets, bytes, [](size_t) { return false; }) == 0);

    if (!failures) std::printf("readback_policy: OK\n");
    return failures ? 1 : 0;
}
