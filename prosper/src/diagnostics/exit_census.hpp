#pragma once
// One-line end-of-run census reports: registration and gating, in one place.
//
// Four censuses in this tree print a single summary line when the run ends, and each was written
// with its own copy of the same twelve lines. The formatting legitimately differs -- a per-reason
// distribution, a bucket series and a per-site table are not the same report -- but the two parts
// that are easy to get CATASTROPHICALLY wrong are identical every time:
//
//   1. WHICH end-of-run hook. `std::atexit` does not run here: every prosper frontend leaves
//      through `_exit()`/`_Exit()` (#3353, and `exit_reports.hpp` lists the exits by name). A
//      census registered with atexit is silent forever, and a missing end-of-run line is
//      indistinguishable from a measured zero -- so the failure reads as a finding. That mistake
//      was made once in this very series and caught only because a run that should have reported
//      something reported nothing.
//   2. THE ZERO GATE. A census that prints when it counted nothing is noise on every unrelated
//      run, and a noisy diagnostic gets switched off, which costs the one run where it mattered.
//
// So this owns registration and the opt-out variable, and hands formatting back to the caller.
// The report returns `false` to print nothing, which keeps the zero decision with the code that
// knows what zero means for that census.
//
// Deliberately NOT a shared counter type. `draw_disposition` carries per-pass state, an
// independent cross-check and a bucket series; `worker_spawn_census` is a per-site table. Forcing
// those into one container would grow the abstraction until every caller passed flags to opt out
// of it. The duplication that was worth removing is the part where a mistake is silent.

#include <functional>

namespace prosper::diagnostics {

// Registers `report` to run once at the end of the run, through the hook that actually survives
// how this process exits. `disable_env`, when set in the environment, suppresses the report
// entirely; pass nullptr for a census with no opt-out.
//
// `report` returns true if it printed. Returning false is the normal path for a run in which the
// census counted nothing, and costs nothing.
//
// Safe to call during static initialisation and from any thread. The report must not read objects
// with non-trivial destructors -- see exit_reports.hpp for why.
void register_census(const char* disable_env, std::function<bool()> report);

}  // namespace prosper::diagnostics
