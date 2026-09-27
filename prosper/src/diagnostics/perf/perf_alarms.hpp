#pragma once
// The `[perf-alarm]` engine (#3891): closes a rolling window of the frame ledger once per guest
// flip, runs the pure rules over it, and reports what fired.
//
// READ `[perf-alarm]` LINES FIRST when a title is slow: they name the cost, how big it is against
// the title's own frame budget, and the next instrument to reach for.
//
// Output, all default-on:
//   * stderr: `[perf-alarm] #<n> rule=<r> window=<s>s value=<v> <unit> threshold=<t> cost~<ms>
//     <detail> hint=<...>`. De-duplicated per rule on diag_ratelimit's contract: the first three
//     firings, then powers of two, each carrying its 1-based ordinal -- so the last ordinal is a
//     lower bound on how many windows fired, and the line count never is.
//   * PROSPER_PERF_ALARM_LOG=<path>: JSONL, flushed as written, so a run killed by SIGTERM -- which
//     skips the exit report -- still leaves its record. `"type":"alarm"` for every firing (no rate
//     limit) and `"type":"window"` for every window with the raw quantities the rules read, so a
//     quiet run still shows how close it came to each threshold.
//   * a rule is reported only once its condition has held for sustain_windows() consecutive
//     windows (two for performance rules, three for texture-reference-cost, more for
//     shader-compile, one for correctness), so a load spike does not read as a steady-state cost.
//   * correctness rules (dropped-draws, skipped-dispatches) carry a per-reason breakdown: the log
//     line's `reasons=` names the top sites, the JSONL alarm record has a `breakdown` object, every
//     JSONL window has `drop_reasons` / `dispatch_skips` / `host_copy_mib_by_site`, and the exit
//     summary adds the breakdown summed over every window the rule fired in.
//   * at exit (register_exit_report): one summary line per rule that fired (windows fired, worst
//     value, when), or a line saying no rule fired in N evaluated windows. The second form exists so
//     "nothing fired" cannot be confused with "the engine never ran": the latter prints nothing.
//     Either form also lists any rule that had NO DATA in every window (rule_has_data), so a rule
//     that could not have fired is not counted as quiet.
//
// Knobs: PROSPER_NO_PERF_ALARMS=1 (off, and the ledger hooks skip their clocks);
// PROSPER_PERF_ALARM_WINDOW_MS (default 5000); PROSPER_PERF_ALARM_THRESHOLD_PCT (default 100; lower
// it to make every non-correctness rule more sensitive for an investigation).

#include "diagnostics/perf/perf_alarm_rules.hpp"
#include "diagnostics/perf/perf_ledger.hpp"

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace prosper::diagnostics::perf {

struct EngineConfig {
    uint64_t window_ns = 5'000'000'000ull;
    RuleThresholds thresholds{};
    std::string jsonl_path;      // empty: no JSONL
    FILE* log = stderr;          // nullptr: no log lines (tests)
};

class AlarmEngine {
public:
    explicit AlarmEngine(EngineConfig config);
    ~AlarmEngine();
    AlarmEngine(const AlarmEngine&) = delete;
    AlarmEngine& operator=(const AlarmEngine&) = delete;

    // Running totals kept OUTSIDE the ledger by existing default-on censuses, read only when a
    // window closes (so they cost the hot path nothing extra). The process entry point fills it from
    // transfer_pressure; a test passes its own.
    struct ExternalTotals {
        uint64_t transfer_bytes[WindowSample::kTransferCount] = {};
    };

    // One guest flip at `now_ns`. Closes the window when it is due: takes the deltas of `ledger`
    // (and of `external`, when given) since the previous close, evaluates, reports. Returns what
    // fired (empty when not due).
    std::vector<AlarmFiring> on_flip(uint64_t now_ns, Ledger& ledger, uint32_t target_hz,
                                     const ExternalTotals* external = nullptr);

    // The most recent window's firings, as rule names with the time they fired: what the --fps
    // overlay marks. Rules that fired in the last window are "active".
    std::vector<const char*> active_rules() const;

    // Evaluate and report one already-built window. `t_seconds` is its end, relative to the
    // engine's first flip. Separate from on_flip so a test can drive it by hand. Returns what was
    // REPORTED: a rule whose condition has not yet held for sustain_windows() is not.
    std::vector<AlarmFiring> close_window(const WindowSample& w, double t_seconds);

    // Prints the exit summary to `out`. Returns false (printing nothing) when no window was ever
    // evaluated.
    bool write_summary(FILE* out) const;

    uint64_t windows_evaluated() const;
    uint64_t times_fired(const char* rule) const;

private:
    struct RuleState {
        uint32_t streak = 0;   // consecutive windows the condition has held, reported or not
        uint64_t windows_with_data = 0;   // windows that carried this rule's input (rule_has_data)
        uint64_t fired = 0;
        double worst_value = 0;
        double worst_t = 0;
        double first_t = 0;
        std::string worst_detail;
        // Summed breakdown (AlarmFiring::breakdown) over every window this rule was reported in.
        std::vector<std::pair<const char*, uint64_t>> breakdown_total;
    };
    RuleState& state_for(const char* rule);

    EngineConfig config_;
    mutable std::mutex mutex_;
    FILE* jsonl_ = nullptr;
    bool started_ = false;
    uint64_t origin_ns_ = 0;
    uint64_t window_start_ns_ = 0;
    uint64_t flips_in_window_ = 0;
    uint64_t windows_ = 0;
    uint64_t prev_cost_ns_[kCostCount] = {};
    uint64_t prev_cost_events_[kCostCount] = {};
    uint64_t prev_counters_[kCounterCount] = {};
    uint64_t prev_drop_reasons_[kDropReasonCount] = {};
    uint64_t prev_dispatch_skips_[kDispatchSkipCount] = {};
    uint64_t prev_transfer_bytes_[WindowSample::kTransferCount] = {};
    std::vector<const char*> active_;   // rules REPORTED in the most recent window
    std::vector<std::pair<const char*, RuleState>> rules_;
};

// ---- process-wide entry points (VideoOut) ----------------------------------------------------

// Called once per guest flip (sceVideoOutSubmitFlip and the GPU flip packet both reach it). A no-op
// under PROSPER_NO_PERF_ALARMS=1. Cheap between windows: one clock read and one compare.
void on_guest_flip();

// The guest's sceVideoOutSetFlipRate argument (0=60, 1=30, 2=20 Hz): the frame budget's source.
void set_guest_flip_rate(int32_t rate);

// For a HUD (prosper-app --fps): the rules reported in the most recently closed window, joined
// with ',' into `out` (NUL-terminated, truncated to `cap`). Returns how many rules are active; 0
// when none fired, the engine has not closed a window, or alarms are off. Lock-free and cheap
// enough to call once per presented frame: it reads a published snapshot, never the engine.
size_t active_alarm_summary(char* out, size_t cap);

}  // namespace prosper::diagnostics::perf
