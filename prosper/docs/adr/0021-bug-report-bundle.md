---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0021: One command produces a complete, shareable bug report

## Context

A user reporting a problem today has to be told which switches to set and which logs to send, and
the developer reconstructs the rest: build, host, driver, configuration, which HLE calls were
unimplemented, which perf alarms fired, where the guest was when it faulted. The pieces exist --
the `[perf-alarm] summary`, the unimplemented-call census in `hle/dispatch`, `tools/guest_bt` for
guest backtraces, the fault handlers in `host/fault` -- but nothing gathers them, and a fatal fault
leaves whatever the terminal showed.

Proton's `PROTON_LOG=1` writes one complete log a user can attach; crash reporters in most desktop
applications do the same for faults.

## Decision

1. `prosper-app --report <dir>` (and the equivalent on a fatal guest or host fault) writes a bundle:
   prosper version and build, host OS, GPU and driver, the effective configuration (ADR 0017), the
   title id and `param.json` identity fields, the perf-alarm summary, the unimplemented NIDs called,
   the last N log lines, and the guest and host backtraces at a fault.
2. **The bundle contains no game bytes and no absolute host paths**: paths are written relative to
   the dump root or as placeholders, per the charter's publication rule.
3. The bundle format is versioned and has a reader in `tools/`, so a report can be triaged without
   reproducing the run.

Adds spec rule `OPS-1`.

## Consequences

User reports arrive complete and comparable, which matters most for hosts and drivers the developers
do not have. The fault path must write the bundle without relying on state the fault may have
corrupted, which is the hard part and needs its own tests.

## Alternatives considered

- Document what to collect: the status quo, and reports still arrive incomplete.
- Upload reports automatically: rejected, publishing is the user's decision.

## Approval

Requires the project owner's acceptance.
