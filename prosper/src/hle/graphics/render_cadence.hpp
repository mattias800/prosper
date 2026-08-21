// Render-cadence sampling policy: pure predicates, no getenv and no clock.
//
// PROSPER_RENDER_EVERY=N asks the AGC submit path for a sparse phase -- render one draw submit in
// every N -- so a long intro can be advanced cheaply.  Two separate things can quietly make the run
// that comes out of it a cadence-1 run wearing an `every=N` label:
//
//   1. `execute_submit_work`'s `ordered_dma_requires_render` overrides the cadence per submit, since
//      a retained DMA copy in a submit that also draws may read pixels an earlier draw in the same
//      submit produced.  That override is correct; a title where it fires on every submit simply
//      gets no sparse phase.
//   2. The variable never reached the process at all.  A launcher that exports its own
//      `PROSPER_RENDER_EVERY=1` after the caller set 16 is not hypothetical -- it is exactly how
//      this instrument came to be written (#2837), and it produced a confidently wrong "the
//      accelerator is inert on this title" conclusion before anything caught it.
//
// Neither case announces itself.  Throughput does not improve, the frames are cadence-1 frames, and
// every other line in the run still says `every=N`, so the natural reading is "this title is simply
// slow" -- a claim about the subject that came from the apparatus.  Reporting the cadence ACTUALLY
// in effect, together with how many of its requested skips were overridden, is the only thing in the
// run that separates the three.
//
// These live here rather than inline in hle_agc.cpp so the "is it inert?" rule is testable without a
// live title: the file that owns them supplies only getenv and the counters.

#pragma once

#include <cstdint>

namespace prosper {

// Counters accumulated over the draw submits seen so far.
struct RenderCadenceCounters {
    uint64_t draw_submits = 0;   // draw submits offered to the cadence
    uint64_t skips_wanted = 0;   // ... of those, the ones the cadence asked to skip
    uint64_t dma_forced = 0;     // ... of THOSE, the ones a retained DMA copy rendered anyway
};

// A cadence is inert when every single skip it asked for was overridden. Deliberately `==` and not a
// ratio: a title that skips even occasionally is being accelerated, just less than requested, and
// warning about it would train the reader to ignore the warning that matters. `min_skips_wanted`
// keeps a handful of early submits from reading as a rate.
inline bool render_cadence_is_inert(unsigned requested_cadence,
                                    const RenderCadenceCounters& counters,
                                    uint64_t min_skips_wanted) {
    if (requested_cadence <= 1) return false;          // nothing was requested, so nothing is inert
    if (counters.skips_wanted < min_skips_wanted) return false;
    return counters.dma_forced == counters.skips_wanted;
}

// Percentage of requested skips that a retained DMA copy rendered anyway; 0 when none were asked
// for. Reported rather than only the inert/not verdict, so a partially-overridden run is legible.
inline double render_cadence_override_percent(const RenderCadenceCounters& counters) {
    if (counters.skips_wanted == 0) return 0.0;
    return 100.0 * static_cast<double>(counters.dma_forced) /
           static_cast<double>(counters.skips_wanted);
}

}  // namespace prosper
