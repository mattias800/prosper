#pragma once
// Why a render pass STOPPED accepting draws — one named reason per pass boundary.
//
// WHY THIS EXISTS. A pass is where prosper pays its fixed per-pass costs: a descriptor-pool
// creation, the process-wide persistent-resource lock whose critical section is the whole of
// `render_draw_pass_rgba`, the pass's barriers, and on a flush a queue submit. Those costs are
// paid per PASS, so what they actually cost over a run is set by how many draws a pass holds.
//
// Measured 2026-09-25 with the draw-disposition census, on a 60 s and a 90 s headless route:
//
//     The Messenger   149,215 passes   702,445 draws   mean 4.71   49.5% single-draw
//     Astro Bot        15,259 passes    18,821 draws   mean 1.23   98.5% single-draw
//
// A title whose passes hold one draw pays every fixed per-pass cost once per draw. But the
// grouping loop has three ways to stop, and the measurement above cannot say which one is
// responsible -- and they have completely different fixes:
//
//   * TargetsChanged     the next draw renders somewhere else. IRREDUCIBLE: a different colour
//                        target really is a different Vulkan render pass. If this dominates, the
//                        answer is to make a pass CHEAPER, not longer.
//   * MrtResolveDiffers  the next draw disagrees about MRT resolve. Worth auditing: if the
//                        predicate is stricter than Vulkan requires, relaxing it lengthens passes
//                        for every title at once.
//   * DepthFeedback      the next draw samples the depth this pass writes. IRREDUCIBLE: a real
//                        read-after-write hazard.
//   * EndOfItems         the submit ran out of draws; nothing was rejected.
//
// So this census exists to choose between "lengthen the passes" and "make a pass cheaper" on
// evidence instead of on the aggregate. Guessing wrong here costs a rewrite: the two answers
// touch different code and neither is cheap.
//
// It reports the DISTRIBUTION, not just the counts -- mean draws per pass per reason -- because
// the reason that ends the most passes is not necessarily the reason holding the mean down. A
// reason that fires often on already-long passes is not a problem; one that fires on singletons
// is.
//
// One line at end of run via `register_exit_report` (NOT std::atexit, which every frontend here
// skips -- #3353). `PROSPER_NO_PASS_BREAK_CENSUS=1` silences it.

#include <cstdint>

namespace prosper::gpu {

enum class PassBreak : uint8_t {
    TargetsChanged = 0,   // same_targets() rejected the next draw
    MrtResolveDiffers,    // mrt_same_resolve_pass() rejected the next draw
    DepthFeedback,        // the next draw samples the depth array this pass writes
    EndOfItems,           // no draws left in this submit
    Count
};

const char* pass_break_name(PassBreak reason);

class PassBreakCensus {
public:
    // `draws` is how many draws the pass that just ended accepted, so a reason can be judged by
    // the pass length it produced rather than only by how often it fires.
    void note_break(PassBreak reason, uint64_t draws);
    // Returns false when nothing was counted; see diagnostics/exit_census.hpp.
    bool report_totals();

    uint64_t passes(PassBreak reason) const;
    uint64_t draws(PassBreak reason) const;

private:
    struct State;
    State& state() const;
};

PassBreakCensus& pass_break_census();

}  // namespace prosper::gpu
