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
// pass's entry (DrawDispositionPassScope), before any refusal or skip path can divert a draw;
// `recorded` is counted
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

// The reasons a draw prosper wanted to issue did not reach the GPU. The first eight follow the
// order the setup loop can reach them; later ones are appended (never inserted), so a slot's
// meaning, its perf::DropReason mirror and every grepped name stay stable across versions.
// clang-format off: one reason per line, comments aligned; perf::DropReason mirrors this order
enum class DrawDrop : uint8_t {
    GeometryCapability = 0,  // fragment program needs the Geometry capability; device lacks it
    MeshShape,               // mesh draw exceeds the device's mesh work-group limits
    SubgroupFeatures,        // required fragment subgroup features (incl. wave64) unavailable
    GdsAllocation,           // the persistent internal-GDS buffer could not be allocated
    BufferResources,         // a draw's buffer resources did not resolve
    ShaderRejected,          // the recompiler produced no SPIR-V for a required stage
    PipelineCreation,        // vkCreateGraphicsPipelines declined the draw's pipeline
    TargetMemory,            // no memory type could hold a pass attachment (#3901): whole pass
    // Whole-pass refusals before the per-draw loop. Each was a silent drop until `seen` moved to
    // the pass entry (#4643): a `return` there left seen = recorded = dropped = 0 and the
    // conservation check balanced trivially. Appended, so earlier slots keep their indices.
    ResourceOrder,           // a draw's compact resource order is not a permutation
    ResourceContract,        // a texture plane span or storage-image numeric contract failed
    UnprovenSubmission,      // an earlier submission's completion was never proven
    DeviceUnavailable,       // the backend has no usable Vulkan device
    DetileDevice,            // a GPU detile program belongs to another device
    OwnedWave,               // an owned-graphics-wave draw could not be materialized
    NggExpansion,            // a merged-NGG draw could not be expanded into its run draws
    VolumeView,              // the volume colour target's view range is invalid
    VolumeNotPersistent,     // a volume target with persistent colour targets disabled
    VolumeFeedback,          // a volume pass samples the volume it writes (no snapshot yet)
    CommandPool,             // no command pool could be leased for the pass
    VolumeMultiTarget,       // RETIRED (#4643): multi-target volume passes now render
    VolumeSeeded,            // a single-target volume pass with a CPU seed (multi-target wins)
    VolumeTargetLimits,      // the device cannot hold the volume target or its layer count
    VolumeDepthStencil,      // a volume pass with a depth/stencil attachment
    VolumeBudget,            // a volume target could not be retained (budget or allocation)
    TargetCreation,          // a pass attachment's image or view could not be created or bound
    RenderPassCreation,      // vkCreateRenderPass failed
    FramebufferCreation,     // vkCreateFramebuffer failed
    PressureFlush,           // the cache-pressure flush before the pass failed to submit or wait
    NggSubgroup,             // a merged-NGG draw the backend could not run whole (#3135 P5)
    VolumeMixedTarget,       // colour slots one layered framebuffer cannot hold together (#4643)
    Count
};
// clang-format on

// Stable kebab-case strings; they appear in logs that get grepped. Never reword casually.
const char* draw_drop_name(DrawDrop reason);

class DrawDispositionCensus {
public:
    // Counted ONCE per pass, by DrawDispositionPassScope at the entry of the backend's pass
    // (render_draw_pass_rgba), for every draw handed to it -- before any refusal or skip can
    // divert one. Not called per draw in the setup loop: an early `return` before that loop then
    // left `seen` at zero and the pass balanced trivially, which is how #4643's volume draws were
    // lost for weeks with no alarm. Counting at entry makes every unnamed exit UNACCOUNTED.
    void note_seen(uint64_t count = 1);
    // Counted at the Vulkan draw call, by a route sharing no counter with note_seen.
    void note_recorded(uint64_t count = 1);
    // Counted at each involuntary skip site, naming the mechanism that was missing.
    void note_dropped(DrawDrop reason, uint64_t count = 1);
    // The pass replaced its draw list with a different number of draws (a merged-NGG draw
    // becomes its run draws). Re-bases `seen` from `from` to `to` so the pass is judged against
    // the draws it will actually try to record.
    void note_rebatched(uint64_t from, uint64_t to);
    // All five above count nothing on a thread inside perf::SuppressDrawDropCounting (an F9 or
    // menu capture, a diagnostic re-realization): those passes are not live execution and must
    // not raise the alarm being investigated.

    // Called once at the end of a pass, on the pass's own thread. Prints nothing when the pass was healthy and quiet under
    // the per-reason budget; always prints a pass that recorded zero draws having seen some, and
    // always prints when the two independent routes disagree.
    void report_pass();

    // One process-lifetime snapshot line, printed at end of run when any draw was seen, via
    // `register_exit_report` -- NOT std::atexit, which every frontend here skips (#3353). This is a
    // COMPLEMENT to report_pass(), never a replacement: a run that ends in a device loss may never
    // reach it, which is exactly why the per-pass report is the primary and this is the
    // convenience. It also separates the two states the per-pass report cannot distinguish by its
    // silence -- "every pass was healthy" and "no pass ever ran" both print nothing per-pass.
    // Loads are independent and no caller proves CPU quiescence. RUN SNAPSHOT therefore always
    // says quiescence=unverified, even when the numbers balance. A snapshot-delta may reflect an
    // in-flight pass or interleaved loads, not a completed-pass accounting error (#3973).
    // Completed-pass UNACCOUNTED reports and their performance-alarm counter remain authoritative.
    // Returns false when no draw was seen; see diagnostics/exit_census.hpp.
    bool report_totals();

    // Records one completed pass's wall time against how many draws it held. Bucketed by draw
    // count, because the question this answers is not "how long is a pass" but "how much of a
    // pass is FIXED" -- the cost a pass pays whether it holds one draw or fifty. The mean
    // duration of a one-draw pass is that fixed cost plus a single draw's variable cost, so the
    // bucket series is the measurement and no curve needs to be assumed. A regression would have
    // to assume linearity; buckets show the shape and let it be false.
    void note_pass_duration(uint64_t draws, uint64_t nanoseconds);

    // Programmatic readers (tests, tools). Totals are process-lifetime, not per pass.
    // The CALLING THREAD's current pass: per-pass figures are thread-local, because concurrent
    // passes overlap at the entry where `seen` is counted (draw_disposition.cpp, PassCounters).
    // The current pass's seen count, for the scope guard to read before report_pass() resets it.
    uint64_t pass_seen_for_scope() const;
    // The current pass's seen - recorded - dropped, clamped at zero: what a refusal names.
    uint64_t pass_unaccounted_for_scope() const;
    uint64_t seen() const;
    // Passes recorded by note_pass_duration (refused passes are excluded; see the scope).
    uint64_t timed_passes() const;
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
//
// It is also where `seen` is counted: the constructor takes the number of draws handed to the
// pass. A refusal site names its cause with `refuse(reason)` (or `return scope.refuse(reason,
// out);`), and the scope then drops every draw still unaccounted at exit under that reason. An
// exit that names nothing leaves the gap standing, so it is reported as UNACCOUNTED.
struct DrawDispositionPassScope {
    explicit DrawDispositionPassScope(uint64_t draws);
    DrawDispositionPassScope(const DrawDispositionPassScope&) = delete;
    DrawDispositionPassScope& operator=(const DrawDispositionPassScope&) = delete;
    ~DrawDispositionPassScope();

    void refuse(DrawDrop reason) { refusal_ = reason; }
    template <typename Result>
    Result& refuse(DrawDrop reason, Result& result) {
        refuse(reason);
        return result;
    }
    // Some of the pass's draws dropped by a filter that lets the rest through.
    void drop(DrawDrop reason, uint64_t count) {
        draw_disposition_census().note_dropped(reason, count);
    }
    // The pass now holds `to` draws in place of `from` (DrawDispositionCensus::note_rebatched).
    void rebatch(uint64_t from, uint64_t to) { draw_disposition_census().note_rebatched(from, to); }

private:
    uint64_t start_ns_ = 0;
    DrawDrop refusal_ = DrawDrop::Count;
};

// A logical batch refused before it reached any pass (render_draws_rgba's whole-batch preflight):
// accounted as one pass that saw `draws` draws and dropped every one under `reason`.
void refuse_draw_pass(uint64_t draws, DrawDrop reason);

}  // namespace prosper::gpu
