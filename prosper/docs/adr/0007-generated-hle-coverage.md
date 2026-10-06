---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0007: Generate HLE library coverage from the registry, never estimate it

## Context

How much of each Sony library prosper implements is answered today by reading code or by booting a
title. An unregistered import does not fail: the dispatcher logs it once and returns 0
(`src/hle/dispatch/dispatch.cpp`), which for most `SCE_OK` contracts tells the guest an operation
succeeded that never ran -- the false-success class (#2081). Two instruments already see this:
`tools/nid_census` answers it statically and exhaustively per title, from the loader's own parser
and the runtime registry (not a source grep), and the always-on `unimplemented-hle-calls` perf alarm
reports it for each run.

AnyPS5, the sibling PS5 project compared in the architecture review, publishes a generated per-library progress
report (`tools/progress.py`), counting a function done when its body no longer calls the stub
helper, and a shader instruction done when its decoder recognises it. Generating the number rather
than estimating it is the idea worth taking. Its definition of "done" is not: in prosper a handler
that returns a constant without the stub helper would count as done, and that is exactly the shape
the charter forbids ("do not ship shims that fake output").

## Decision

1. `tools/nid_census` grows a summary mode that reports, per library, imported NIDs that are
   registered, resolved by another module of the title, or falling to the return-0 default. It
   reads the registry and the dumps, so it runs where the dumps are, never in hosted CI.
2. The committed artefact is counts only -- per title and library, no function names or NIDs --
   because the dumps and the firmware symbol data are never committed. It is regenerated at
   release alongside the snapshot pass and diffed against the previous release; a rise in a
   title's return-0 count is a release finding.
3. Spec `HLE-2`. "Implemented" means registered; whether a registered handler is correct is what
   tests and the `implement-hle-function` procedure establish, and no count claims it.

## Consequences

The project gains a progress number that cannot drift from the code, and the false-success class
becomes visible per library instead of per boot log. The number is honest about its limit: it
measures reach, not correctness. Nothing about it can run in hosted CI, which is stated rather than
papered over with a source-grep proxy that would report table-driven registrations as missing.

## Alternatives considered

- A source grep for `HLE(...)` definitions in CI: rejected, `nid_census` documents that registration
  is not only literal call sites, so a grep reports false gaps.
- Counting stub-free bodies as done: rejected above.

## Approval

Requires the project owner's acceptance; it adds a release-time artefact and step.
