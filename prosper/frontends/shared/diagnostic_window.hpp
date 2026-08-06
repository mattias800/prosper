// diagnostic_window.hpp — selecting a short renderer-callback census window by ORDINAL or by TIME.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace prosper::frontend {

// `PROSPER_DUMP_PERSISTENT=<N>` and `PROSPER_PASS_LOG=<N>` open a three-callback census window at
// renderer-callback ordinal N. That ordinal is a renderer-internal counter with no published rate,
// and the states worth censusing are named in seconds ("after the intro", "once the frame goes
// black"), so choosing N is guesswork: measured on Sonic Frontiers (#1968), the same 360 s route
// reaches ordinal 6,560 while its *submit* counter reaches 26,209 — a 4x gap that cost one full run
// when the ordinal was estimated from a capture's submit number. `PROSPER_GPU_CAPTURE_AFTER_MS`
// already solves exactly this for captures; this gives the two census switches the same form.
//
//     PROSPER_DUMP_PERSISTENT=26000     ordinal — unchanged for every reachable value
//     PROSPER_DUMP_PERSISTENT=ms:240000 the first callback at or after 240 s, and the next two
//
// One deliberate difference in the ordinal form, stated because "unchanged" should not be taken on
// trust: the old test was `at >= min && at < min + 3u`, whose addition wraps for the top three
// `uint64` values, so those windows could never fire. `contains()` subtracts instead and they do.
// Unreachable in practice — the counter is a per-callback ordinal — and the new answer is the
// intended one, but it is a difference and the test pins it.
//
// Time is measured from the first call to `contains()`, i.e. from the first renderer callback of the
// run. That is the same KIND of origin `PROSPER_GPU_CAPTURE_AFTER_MS` uses — the first armed check on
// its own path — but it is a separate lazily-started `steady_clock` static on a separate path, so a
// capture and a census aimed at the same millisecond agree only to within the gap between the two
// clocks' first calls. Close enough to aim both at one phase; not a shared timebase.
struct DiagnosticWindowSpec {
    bool by_time = false;      // `value` is milliseconds rather than an ordinal
    uint64_t value = 0;        // ordinal, or milliseconds since the first callback
    uint32_t span = 3;         // how many consecutive callbacks the window admits
};

// Parse one environment value. A bare number keeps `strtoull(text, nullptr, 0)`'s historical
// behaviour EXACTLY, including its treatment of junk as 0 — this switch has been aimed at ordinal 0
// by a typo before, and silently moving that to "never fires" would change what an existing recipe
// does. Only the `ms:` prefix is new; junk after it is likewise 0, i.e. "immediately".
inline DiagnosticWindowSpec parse_diagnostic_window(const char* text, uint32_t span = 3) {
    DiagnosticWindowSpec spec;
    spec.span = span ? span : 1u;
    if (!text) return spec;
    if (std::strncmp(text, "ms:", 3) == 0) {
        spec.by_time = true;
        spec.value = std::strtoull(text + 3, nullptr, 0);
        return spec;
    }
    spec.value = std::strtoull(text, nullptr, 0);
    return spec;
}

// The window itself. An ordinal spec is stateless. A time spec LATCHES: the first callback at or
// after the deadline fixes the window's start ordinal, and every later call answers against that
// fixed start. Without the latch the window would re-open on every subsequent callback and the
// census would run for the rest of the process — a three-submit diagnostic that reads back every
// persistent 4K colour target is not something to leave running.
//
// Not thread-safe, deliberately: both call sites live on the renderer's single present thread, next
// to the `dp_submit` / `warned` / `last_scanout_present` statics that already carry that contract.
class DiagnosticWindow {
public:
    DiagnosticWindow() = default;
    explicit DiagnosticWindow(DiagnosticWindowSpec spec) : spec_(spec) {}

    // `elapsed_ms` is ignored for an ordinal spec, so a caller with no clock may pass 0.
    bool contains(uint64_t ordinal, uint64_t elapsed_ms) {
        if (!spec_.by_time)
            return ordinal >= spec_.value && ordinal - spec_.value < spec_.span;
        if (!started_) {
            if (elapsed_ms < spec_.value) return false;
            started_ = true;
            start_ordinal_ = ordinal;
        }
        return ordinal >= start_ordinal_ && ordinal - start_ordinal_ < spec_.span;
    }

    const DiagnosticWindowSpec& spec() const { return spec_; }
    bool started() const { return started_ || !spec_.by_time; }

private:
    DiagnosticWindowSpec spec_{};
    bool started_ = false;
    uint64_t start_ordinal_ = 0;
};

} // namespace prosper::frontend
