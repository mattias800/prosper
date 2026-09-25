#pragma once
// Why a graphics draw the guest issued did not reach the GPU — on by default, one named reason
// per drop.
//
// THE PROBLEM THIS EXISTS FOR. A pass can drop a draw for six different reasons, and in the state
// this instrument was written against, the six were not distinguishable from outside:
//
//   | reason                          | what the run told you                                |
//   |---------------------------------|------------------------------------------------------|
//   | fragment needs Geometry         | one `[render]` line per draw (unbounded volume)      |
//   | mesh shape unsupported          | one `[mesh]` line per draw (unbounded volume)        |
//   | subgroup/wave64 unsupported     | only under the opt-in wave64 census                  |
//   | persistent GDS alloc failed     | ONE line per process, via `std::call_once`           |
//   | buffer resources not ready      | NOTHING                                              |
//   | shader rejected (no SPIR-V)     | NOTHING at the skip site                             |
//
// All six then incremented one shared `skipped_draw()` counter. So "the world is black" and "the
// world is black BECAUSE 146 draws wanted wave64 on a 32-wide device" were the same observation,
// and separating them meant guessing which opt-in census to arm. That guess is what turns a
// ten-minute question into a multi-day one on each new title, and — because an unexplained black
// frame invites experimenting until something works — it is also what biases the eventual fix
// toward a title-shaped workaround instead of the general mechanism the reason names.
//
// A classified reason points at a MECHANISM (wave64 fragment shaders; mesh-shader group limits;
// the recompiler's rejection set), and a mechanism fixed once serves every title that needs it.
// That is the whole design intent: this is a steering instrument, not only a reporting one.
//
// WHAT IT IS NOT. It does not observe *deliberate* declines — `PROSPER_SKIP_DRAW_PROGRAM` and
// friends are a debugging lever with their own reporting (`draw_program_skip.hpp`). This counts
// only draws prosper WANTED to issue and could not. The two populations must not be merged: one
// is a defect, the other is an experiment.
//
// SELF-VALIDATION, because a census that cannot detect its own invalidity is worth little
// (charter: prefer experiments that detect their own invalidity). `seen` is counted once at the
// top of the setup loop, before any skip path can divert a draw; `recorded` is counted
// independently at the Vulkan draw call. A pass therefore asserts
//
//     seen == recorded + sum(dropped)
//
// by two routes that share no counter. When they disagree the report says
// `UNACCOUNTED=<n>` rather than printing a clean total — which means a drop path was added
// without a reason, and the census names its own blind spot instead of hiding it.
//
// VOLUME. Default on, and quiet on a healthy pass: nothing prints when every seen draw is
// recorded. The FIRST occurrence of each reason prints immediately (a new failure mode is exactly
// what a reader needs at once), then that reason follows `diag_ratelimit.hpp`'s contract — first
// N, then powers of two, carrying the 1-based ordinal on every line so the last ordinal is a
// lower bound on the population and the line count is never mistaken for one. Budgeting is PER
// REASON so a high-volume reason cannot exhaust the log before a rare one gets its line.
//
// A pass that records ZERO draws while having seen some is always reported, regardless of the
// per-reason budget: that is a black pass, and it is the observation the instrument exists for.
//
// `PROSPER_DRAW_DISPOSITION_VERBOSE=1` prints every pass including healthy ones. Verifying this
// census on a new title starts there: a silent instrument and one that was never reached look
// identical from outside, so a quiet default may only be read as "no drops" once a verbose run on
// the same route has shown the reporting path is live.
//
// `PROSPER_NO_DRAW_DISPOSITION=1` silences the reporting (the counters keep running, so a
// programmatic reader still works). There is deliberately no variable that turns the census ON,
// because a diagnostic you must know to enable is one nobody enables.

#include <cstdint>

namespace prosper::gpu {

// The reasons a draw prosper wanted to issue did not reach the GPU. Order is the order the setup
// loop can reach them, so a reader comparing two runs sees reasons appear in a stable sequence.
enum class DrawDrop : uint8_t {
    GeometryCapability = 0,  // fragment program needs the Geometry capability; device lacks it
    MeshShape,               // mesh draw exceeds the device's mesh work-group limits
    SubgroupFeatures,        // required fragment subgroup features (incl. wave64) unavailable
    GdsAllocation,           // the persistent internal-GDS buffer could not be allocated
    BufferResources,         // a draw's buffer resources did not resolve
    ShaderRejected,          // the recompiler produced no SPIR-V for a required stage
    PipelineCreation,        // vkCreateGraphicsPipelines declined the draw's pipeline
    Count
};

// Stable kebab-case strings; they appear in logs that get grepped. Never reword casually.
const char* draw_drop_name(DrawDrop reason);

class DrawDispositionCensus {
public:
    // Counted once per draw at the top of the setup loop, before any skip path can divert it.
    void note_seen(uint64_t count = 1);
    // Counted at the Vulkan draw call, by a route sharing no counter with note_seen.
    void note_recorded(uint64_t count = 1);
    // Counted at each involuntary skip site, naming the mechanism that was missing.
    void note_dropped(DrawDrop reason);

    // Called once at the end of a pass. Prints nothing when the pass was healthy and quiet under
    // the per-reason budget; always prints a pass that recorded zero draws having seen some, and
    // always prints when the two independent routes disagree.
    void report_pass();

    // One process-lifetime summary line, printed at end of run when any draw was seen, via
    // `register_exit_report` -- NOT std::atexit, which every frontend here skips (#3353). This is a
    // COMPLEMENT to report_pass(), never a replacement: a run that ends in a device loss may never
    // reach it, which is exactly why the per-pass report is the primary and this is the
    // convenience. It also separates the two states the per-pass report cannot distinguish by its
    // silence -- "every pass was healthy" and "no pass ever ran" both print nothing per-pass.
    void report_totals();

    // Programmatic readers (tests, tools). Totals are process-lifetime, not per pass.
    uint64_t seen() const;
    uint64_t recorded() const;
    uint64_t dropped(DrawDrop reason) const;
    uint64_t dropped_total() const;

private:
    struct State;
    State& state() const;
};

DrawDispositionCensus& draw_disposition_census();

// Reports the pass on EVERY exit from the scope it is declared in -- normal return, early return,
// or exception. A pass that returns early otherwise leaves its counters standing, and they are
// then charged to the NEXT pass, which reads as an accounting gap in a pass that did nothing
// wrong. That is not hypothetical: the explicit end-of-function call this replaces produced
// `UNACCOUNTED=1` on an Astro Bot run, and the unaccounted draw belonged to a different pass
// entirely. A guard is used rather than auditing the 48 `return` statements in the enclosing
// function because the next `return` added would silently reintroduce the leak.
struct DrawDispositionPassScope {
    DrawDispositionPassScope() = default;
    DrawDispositionPassScope(const DrawDispositionPassScope&) = delete;
    DrawDispositionPassScope& operator=(const DrawDispositionPassScope&) = delete;
    ~DrawDispositionPassScope() { draw_disposition_census().report_pass(); }
};

}  // namespace prosper::gpu
