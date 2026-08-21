// boot_phase_log.hpp — make the boot phases OBSERVABLE.
//
// `boot_program()` records seven phase transitions (PROCESS_START .. BOOT_COMPLETE). Until this
// file existed, nothing could ever print them: the recorder lives behind the compile-time
// `PROSPER_DIAGNOSTICS` option (OFF by default, so the calls inline to no-ops), and even in an
// enabled build `DiagnosticContext::enable()` was never called from anywhere in the tree and no
// subscriber was ever attached to the event bus. Seven instrumentation points, zero reachable
// output, in every build the project ships.
//
// That gap is expensive in one specific way. `boot_program()` does not merely load modules — it
// ends by calling `run_guest_inits()`, which executes the guest's own module initialisers. So a
// title can spend minutes, or forever, inside `boot_program()` running real guest code. A frontend
// that boots then samples (`tools/screenshot`, `tools/boot_trace`, `prosper-app`) prints nothing at
// all during that window, and `screenshot --timeout` cannot fire because its deadline is checked
// inside the sampling loop, which `boot_program()` has not yet returned to. The run therefore has
// to be stopped from outside, and an externally killed run looks exactly like a defect rather than
// like a run that was killed (instrument traps 197 and 213).
//
// This is deliberately a RUNTIME switch on a build-time-optional subsystem, and the asymmetry is
// the point: an instrument that needs a rebuild to answer "where did boot stop?" is unavailable
// exactly when it is needed, because the title that hangs is the one you are not already
// instrumented for.
//
//   PROSPER_BOOTPHASE=1   ->  [bootphase] +0.0ms PROCESS_START
//                             [bootphase] +812.4ms LINKING
//                             ...
//
// Timestamps are relative to the first phase logged, so the reader gets phase DURATIONS without
// arithmetic, and the last line printed names the phase the boot is currently inside. Output goes
// to stderr and is flushed per line: an unflushed buffer is lost when the run is killed, which is
// precisely the run this exists to explain.
//
// The phase argument is `unsigned` rather than `BootPhase` on purpose. `BootPhase` has two separate
// definitions selected by `PROSPER_DIAGNOSTICS` (the real one in `core/types.hpp`, the stub in
// `diagnostics.hpp`); taking the underlying value lets one implementation serve both builds with no
// ODR hazard and no dependence on which one is active.

#pragma once

namespace prosper::diagnostics {

// Print one boot-phase transition when PROSPER_BOOTPHASE is set; otherwise return immediately.
// Cheap enough to call unconditionally: the environment is read once and cached.
void log_boot_phase(unsigned phase);

// Name for a BootPhase's underlying value, or "UNKNOWN" when out of range. Exposed for tests.
const char* boot_phase_log_name(unsigned phase);

// True when PROSPER_BOOTPHASE selected logging. Exposed so a test can assert the gate itself
// rather than inferring it from the absence of output — a silent instrument and a disabled one
// look identical, which is the failure mode this whole file is about.
bool boot_phase_logging_enabled();

} // namespace prosper::diagnostics
