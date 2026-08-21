// test_render_cadence — when is a requested sparse render cadence INERT? (#2837)
//
// PROSPER_RENDER_EVERY=N asks the AGC submit path for a sparse phase, and a retained DMA copy in a
// submit that also draws overrides it. The override is correct; being silent about it is not. A
// title where it fires on every submit gets no sparse phase at all while the run still reports
// `every=N`, so an accelerated route was never accelerated and nothing says so. The same silence
// covers a cadence that never reached the process — which is what actually happened in #2837, where
// a launcher re-exported `PROSPER_RENDER_EVERY=1` over a caller's 16 and the run was read as
// evidence about the title.
//
// The arm carrying the design decision is `partial override is NOT inert`: the obvious sloppy
// spelling is a ratio ("mostly overridden"), and a title that still skips occasionally IS being
// accelerated — warning there would train the reader to ignore the warning that matters. Loosen the
// `==` in render_cadence_is_inert to a `>=` ratio and that arm reddens while every other arm here
// still passes.

#include "hle/graphics/render_cadence.hpp"

#include <cstdio>

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else std::printf("  [ok] %s\n", m); } while (0)

using prosper::RenderCadenceCounters;
using prosper::render_cadence_is_inert;
using prosper::render_cadence_override_percent;

int main() {
    constexpr uint64_t kMin = 256;

    // No cadence requested -> "inert" is not a verdict the run can earn, however DMA behaves. The
    // counters here are deliberately the fully-inert shape, so this arm fails if the
    // `requested_cadence <= 1` guard is deleted; giving it {4096,0,0} would make it pass through the
    // min-samples guard instead and pin nothing.
    {
        const RenderCadenceCounters c{4096, 3840, 3840};
        CHECK(!render_cadence_is_inert(1, c, kMin),
              "every=1 requests no sparse phase, so it is never inert even when every skip is forced");
    }

    // The Plucky Squire shape: every skip the cadence asked for was rendered anyway.
    {
        const RenderCadenceCounters c{4096, 3840, 3840};
        CHECK(render_cadence_is_inert(16, c, kMin),
              "every=16 with all 3840 requested skips DMA-forced is inert");
        CHECK(render_cadence_override_percent(c) == 100.0,
              "a fully overridden cadence reports 100% of requested skips forced");
    }

    // MUTATION ARM: one skip really happened, so the run IS sparser than cadence 1 -> stay silent.
    {
        const RenderCadenceCounters c{4096, 3840, 3839};
        CHECK(!render_cadence_is_inert(16, c, kMin),
              "3839 of 3840 forced is NOT inert: one real skip means the cadence did something");
        CHECK(render_cadence_override_percent(c) > 99.9 &&
                  render_cadence_override_percent(c) < 100.0,
              "the percentage still reports the near-total override the verdict declines");
    }

    // A handful of early submits is a coincidence, not a rate.
    {
        const RenderCadenceCounters early{8, 7, 7};
        CHECK(!render_cadence_is_inert(16, early, kMin),
              "7 requested skips is below the minimum sample count -- no verdict yet");
        const RenderCadenceCounters at_threshold{300, kMin, kMin};
        CHECK(render_cadence_is_inert(16, at_threshold, kMin),
              "the verdict arrives exactly AT the minimum, not one sample above it");
    }

    // The cadence never came up at all -> no verdict, and no division by zero.
    {
        const RenderCadenceCounters c{4096, 0, 0};
        CHECK(!render_cadence_is_inert(16, c, kMin),
              "zero requested skips cannot be 'all overridden'");
        CHECK(render_cadence_override_percent(c) == 0.0,
              "override percent is 0 (not NaN) when no skip was ever requested");
    }

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
