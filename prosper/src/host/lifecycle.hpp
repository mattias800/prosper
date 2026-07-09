// lifecycle.hpp — cooperative stop signal for a long-running guest session.
//
// The headless harness (boot_trace) drives the guest to a fixed frame budget. A frontend app that
// keeps the guest running as long as its window is open needs a way to ask the guest run-loop to
// wind down (e.g. on window-close). This is that signal — a single process-wide flag the run-loop
// polls at its own boundary; it does NOT interrupt guest code mid-instruction.
//
// It is deliberately tiny and dependency-free so it lives in prosper_core without pulling in any
// OS/windowing code. boot_trace is unaffected (it never sets the flag); only an opt-in run-loop
// that checks prosper_stop_requested() responds to it.
#pragma once

namespace prosper {

// Request the guest run-loop to stop at its next boundary. Idempotent; thread-safe (any thread).
void prosper_request_stop();

// True once a stop has been requested. Polled by the run-loop; thread-safe.
bool prosper_stop_requested();

// Clear the flag (tests / restarting a session in-process).
void prosper_reset_stop();

} // namespace prosper
