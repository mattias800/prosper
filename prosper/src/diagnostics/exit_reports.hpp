// exit_reports.hpp -- end-of-run diagnostic reports that survive the ways a run actually ends.
//
// A report registered with std::atexit prints only when the process returns from main() or calls
// exit(). Almost no run here ends that way (#3353):
//
//   prosper-app          std::_Exit(exitCode)         skips atexit
//   tools/screenshot     _exit(verdict.exit_code)     skips atexit
//   tools/boot_trace     _Exit(42) on a host exception skips atexit
//   the guest's own exit  _Exit() in hle_kernel_time   skips atexit, on every frontend
//   worker-fault          _exit(90) in a signal handler skips atexit
//   SIGTERM / timeout     default action               skips atexit
//
// and a missing end-of-run line is indistinguishable from a zero. The worst case was the two
// reports that exist to explain an EMPTY capture (report_unfired_automatic_capture_gates and
// report_unfired_timeline_capture_selector): they could not reach the operator staring at an empty
// capture on any bounded run.
//
// So an end-of-run report registers HERE, and every deliberate exit path that is not in signal
// context calls flush_exit_reports() immediately before it terminates. A std::atexit fallback runs
// the same flush for an ordinary return from main(). Each registered report runs AT MOST ONCE,
// whichever path reaches it first.
//
// NOT covered, and stated so a missing line is read correctly:
//   * the worker-fault `_exit(90)` and other signal-handler exits -- a report calls fprintf and
//     takes locks, neither of which is async-signal-safe, so flushing there could deadlock the
//     exit it is meant to annotate;
//   * SIGTERM and SIGKILL. A report whose number must survive a killed run needs a periodic
//     report as well (live_compute.cpp's censuses do this).
#pragma once

namespace prosper::diagnostics {

using ExitReport = void (*)();

// Register `report` to run once at the end of the run. Safe to call from any thread and at any
// time, including during static initialisation. A report registered after a flush runs at the
// next one (or at the atexit fallback).
void register_exit_report(ExitReport report);

// Run every registered report that has not already run, in registration order, then fflush stdio.
// Call immediately before std::_Exit / _exit on any NON-signal exit path. Idempotent.
void flush_exit_reports();

}  // namespace prosper::diagnostics
