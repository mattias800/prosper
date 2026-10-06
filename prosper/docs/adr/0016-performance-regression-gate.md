---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0016: Gate releases on the frame invariants, measured on the reference workloads

## Context

The frame invariants in `docs/spec/performance.md` have static CI proxies (the ratchet) and
always-on runtime alarms (`src/diagnostics/perf/`), but nothing fails when a release is slower than
the last one. Performance claims already follow a procedure: a same-binary A/B on reference
workloads (`.claude/skills/perf-change/`), compared with `tools/perf/compare_runs.py`. That
procedure is applied to a change someone chose to measure, not to every release.

Hosted CI cannot run it: it has no dumps and no GPU (`spec/performance.md` § Ruled out).

## Decision

1. The release process runs the reference workloads named in the perf-change skill, on the release
   candidate and on the previous release, same host, same frontend, present mode `immediate`, and
   records per workload: distinct frame rate, the `[perf-alarm] summary`, and the `gpu-sync-wait`,
   `shader-compile` and `host-copy-pressure` shares.
2. A regression beyond run-to-run spread on any workload, or a new perf alarm firing in steady
   state, is a release finding to fix or to accept in writing, as snapshot regressions are today.
3. The results are committed as a small dated table in a release note; no capture is committed.
4. Each restructure from ADRs 0009-0014 quotes this comparison in its PR.

Adds spec rule `PERF-G1`.

## Consequences

Performance becomes a tracked property of releases instead of a property of whichever change was
measured. The cost is release time on a GPU host. Measurement hygiene from the charter applies:
quote the harness and present mode, and drop samples around a capture.

## Alternatives considered

- A per-PR performance gate: rejected, CI cannot run titles, and a local per-PR gate on noisy fps
  would be red or ignored.
- Frame-time budgets per title as hard thresholds: rejected for now, the budgets differ by host and
  would encode one machine's numbers.

## Approval

Requires the project owner's acceptance, because it adds a release step.
