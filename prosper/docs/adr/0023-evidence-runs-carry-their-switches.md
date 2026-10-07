---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0023: An evidence run records its switches, and a run with behaviour switches is not evidence

## Context

The charter already holds the principle in prose: progression screenshots must come from direct
frontend captures, output from forced guest-state diagnostics is not acceptance evidence, and "a
frame that needs a non-default switch is not default-route evidence" (the charter's *ArcRunner*
row, where tracker #1817 holds the rung at 0 for exactly this reason). Nothing checks it. A
`tools/screenshot` manifest (`capture_manifest.hpp`) records the flip pacing in force but not the
other `PROSPER_*` switches that were set, and `tools/snapshot/snapshot.py` passes the caller's whole
environment to each run (`env = dict(os.environ)`) without recording it. So a reader cannot tell
from the artefact whether it qualifies, and a diagnostic left set in a shell silently becomes part
of a snapshot verdict.

The sibling project PortPS5 makes the rule mechanical: its results file records every debug key,
and a run with any debug key can never pass.

## Decision

1. The screenshot and snapshot harnesses write every `PROSPER_*` variable present in the run's
   environment into the manifest they already produce.
2. Using the switch classes from the registry (#4543): a run is marked **acceptance evidence** when
   every recorded switch is a `host-capability` or `diagnostic` switch. A diagnostic, by the
   charter's definition, observes and changes nothing the guest sees, so it does not disqualify a
   run; if one turns out to change guest-visible output, it was misclassified and is a selector.
   A `selector` marks the run **not evidence**, and an `unclassified` switch marks it **unverified**
   until classified.
   The charter's own recipe for reaching the frame loop sets `PROSPER_RENDER=1` and
   `PROSPER_GUEST_ARGS=-force-gfx-direct`. This ADR proposes both as `host-capability`: they choose
   the route every evidence run uses, not a deviation from it. If the owner classifies either
   differently, that choice must name how evidence runs qualify, or none will.
3. The snapshot `check` refuses to report a guarded title as passing from a run that is not
   acceptance evidence.
4. PR and tracker evidence quotes the manifest's classification next to the screenshot.

Adds spec rule `VER-4`.

## Consequences

"Which switches did this capture need?" is answered by the artefact instead of by the author's
memory, and rung claims built on a non-default route are caught by the tool that produced them. It
depends on #4543 landing and on classifying the switches the harnesses commonly set; until a switch
is classified, runs using it are honestly reported as unverified rather than silently accepted.

## Alternatives considered

- Keep the prose rule: it is correct and has not been enforced.
- Scrub all `PROSPER_*` from evidence runs: `tools/gpu_replay/regress.py` does this for replay, but a
  live run sometimes legitimately needs a host-capability switch.

## Approval

Requires the project owner's acceptance and #4543.
