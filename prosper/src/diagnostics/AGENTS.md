# src/diagnostics — observing prosper's own startup

This folder holds the **observer-only** layer for prosper's boot: the seven `BootPhase`
transitions `boot_program()` records, the in-memory history of those events, and the JSON report
written from it. Nothing here participates in loading, linking, mapping or running the guest — if
code in this folder can change what the guest sees, it is in the wrong folder.

The split that matters:

- **`boot_phase_log.{hpp,cpp}` is reachable in the DEFAULT build.** One stderr line per phase,
  gated on the runtime environment variable `PROSPER_BOOTPHASE`. It exists because
  `boot_program()` ends by calling `run_guest_inits()` — real guest code — so a title can be
  inside `boot_program()` for minutes or forever, during which every frontend
  (`tools/screenshot`, `tools/boot_trace`, `prosper-app`) prints nothing and
  `screenshot --timeout` cannot fire, because that deadline lives in a sampling loop `boot_program()`
  has not returned to yet. The last `[bootphase]` line names the phase a stalled boot is inside.
- **`core/` and `storage/` are compiled out unless `PROSPER_DIAGNOSTICS` is defined** — that is what
  the source-glob filter in `CMakeLists.txt` excludes, and the filter is scoped to those two
  subfolders by name so a future file beside `boot_phase_log.cpp` is not silently dropped from the
  default build. They are the
  history, event bus and JSON writer. `diagnostics.hpp` is the single include and selects between
  the real types and stubs.

**The failure this folder already had once, so do not rebuild it:** three independent things each
made the phases unreachable, and any one of them was sufficient. `CMakeLists.txt` excluded the whole
of `src/diagnostics/` from `prosper_core` whenever `PROSPER_DIAGNOSTICS` was off, which is the
default; `DiagnosticContext::enable()` was never called from anywhere in the tree; and no subscriber
was ever attached to the bus. Seven instrumentation points existed and **no build the project
shipped could print any of them** — the instrument was unavailable in exactly the
situation it was written for, because a title that hangs is by definition one nobody had already
instrumented for. That is why the phase *line* is a runtime switch on a build-time-optional
subsystem: the asymmetry is deliberate, not an oversight.

So when adding anything here, ask which half it belongs to. A new *report* or aggregation is
`core/`/`storage/` work and may stay behind `PROSPER_DIAGNOSTICS`. A new signal that answers
"where did it stop?" must be reachable without a rebuild, or it will be missing on the day it
matters.

`diag_clock.hpp` is the second thing on the always-reachable side, and it is here rather than in a
subsystem because its whole purpose is to be shared. Diagnostics in different subsystems write to
one stderr stream from different threads, so their line order carries no ordering information —
comparing two of them by position in a log is unsound, and doing exactly that produced a wrong
published conclusion on #3142. `diag_now_us()` gives them a common monotonic reading; steady_clock
is process-wide, so independently taken values in unrelated translation units compare directly with
no anchor or registry. Anything that needs to be time-ordered against a diagnostic in another
subsystem should stamp it rather than grow a private clock.
