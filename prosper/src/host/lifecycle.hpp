// lifecycle.hpp — cooperative stop/pause signals for a long-running guest session.
//
// The headless harness (boot_trace) drives the guest to a fixed frame budget. A frontend app that
// keeps the guest running as long as its window is open needs a way to ask the guest run-loop to
// wind down (e.g. on window-close), or to wait at a safe boundary while the window keeps pumping
// events. These are process-wide signals; they do NOT interrupt guest code mid-instruction.
//
// It is deliberately tiny and dependency-free so it lives in prosper_core without pulling in any
// OS/windowing code. boot_trace is unaffected (it never sets either signal); only opt-in boundaries
// that check them respond.
#pragma once

namespace prosper {

// Request the guest run-loop to stop at its next boundary. Idempotent; thread-safe (any thread).
void prosper_request_stop();

// True once a stop has been requested. Polled by the run-loop; thread-safe.
bool prosper_stop_requested();

// Pause or resume boundary-aware guest work. Idempotent; thread-safe (any thread).
void prosper_set_paused(bool paused);

// True while the frontend has requested a pause. Thread-safe.
bool prosper_paused();

// Wait until pause is cleared or stop is requested. Returns false when released by stop, true when
// work may continue. Calling this while not paused returns immediately.
bool prosper_wait_while_paused();

// Clear both signals (tests / restarting a session in-process), releasing paused waiters.
void prosper_reset_stop();

} // namespace prosper
