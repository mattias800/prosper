// boot_phase_log.cpp — see boot_phase_log.hpp for why this exists.

#include "boot_phase_log.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace prosper::diagnostics {
namespace {

// The seven phases `boot_program()` records, in enumerator order. Kept here rather than reached for
// through `phase_name()` in `core/types.hpp`, which only exists in a `PROSPER_DIAGNOSTICS` build —
// this logger has to work in the DEFAULT build, which is the whole point of it.
//
// The two lists agreeing is pinned by a test rather than by a comment: `test_boot_phase_log`
// asserts this table against `BootPhase`'s enumerators, so adding a phase without a name here
// fails a check instead of silently printing "UNKNOWN" during the boot somebody is debugging.
const char* const kPhaseNames[] = {
    "PROCESS_START",
    "LINKING",
    "HLE_REGISTERED",
    "MODULES_MAPPED",
    "STUBS_INSTALLED",
    "GUEST_INITS_RUNNING",
    "BOOT_COMPLETE",
};
constexpr unsigned kPhaseCount = sizeof(kPhaseNames) / sizeof(kPhaseNames[0]);

bool enabled_once() {
    static const bool v = [] {
        const char* e = std::getenv("PROSPER_BOOTPHASE");
        // Present-and-not-"0" enables. `PROSPER_BOOTPHASE=0` reads as OFF because a reader who
        // writes that plainly means off, and an instrument that ignores its own disable is worse
        // than one that is missing.
        return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
    }();
    return v;
}

double elapsed_ms() {
    using clock = std::chrono::steady_clock;
    static const clock::time_point t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

} // namespace

const char* boot_phase_log_name(unsigned phase) {
    return phase < kPhaseCount ? kPhaseNames[phase] : "UNKNOWN";
}

bool boot_phase_logging_enabled() { return enabled_once(); }

void log_boot_phase(unsigned phase) {
    if (!enabled_once()) return;
    // Relative to the first phase logged, so consecutive lines read as durations. Flushed per line:
    // this instrument's main use is explaining a run that got KILLED, and a buffered tail is
    // exactly the part such a run loses.
    std::fprintf(stderr, "[bootphase] +%.1fms %s\n", elapsed_ms(), boot_phase_log_name(phase));
    std::fflush(stderr);
}

} // namespace prosper::diagnostics
