---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0025: Title-derived data stays on the user's machine, with a stated rule for test fixtures

## Context

The charter forbids committing game content: dumps are gitignored and "game content is never
redistributed". Several proposals extend that to data *derived* from a title -- replay capsules
(ADR 0005: they carry game imagery), pipeline-cache keys (ADR 0014), fuzz inputs (ADR 0019),
bug-report bundles (ADR 0021) -- but each states it for itself, and the general rule is nowhere.

The tree does not follow a single rule today. `main` commits title-derived shader programs as test
fixtures:

- `prosper/tests/fixtures/gta5_nullable_output_fixture.hpp`, whose own comment reads "Complete
  consumed production programs captured from routed GTA V PPSA04263";
- `prosper/tests/data/messenger_scene_vs.bin` and `messenger_scene_ps.bin`, loaded by
  `tests/gpu/recompiler/test_messenger_recompile_guard.cpp` (#321).

These make recompiler regressions testable without a dump, which is valuable. They are also bytes
derived from a title's own programs, so a rule written as "nothing derived from a title is ever
committed" would be violated on the day it landed.

## Decision

Proposed, for the owner to settle explicitly:

1. Nothing derived from a title's content is uploaded or published outside the repository, and
   none of the following is committed: dumps, captures and capsules, replay corpora, pipeline-cache
   files, fuzz inputs taken from dumps, bug-report bundles carrying game bytes. What a PR may commit
   is prosper's own data about a run -- hashes, counts, timings, pipeline keys -- and screenshots
   under the charter's screenshot rules. Spec rule `LOCAL-1`.
2. **Shader programs as test fixtures** are the open question, and this ADR names the options
   rather than choosing:
   - (a) allowed, as today, when a fixture is the minimum needed to pin a recompiler behaviour,
     with a size bound per fixture and its source recorded beside it;
   - (b) allowed only as synthetic reductions: the program is reduced to the instructions that
     exercise the behaviour, so no complete production program is committed;
   - (c) moved out of tree into the local test corpus, with the tests skipping when it is absent.
   Whichever is chosen is written into `LOCAL-1` before it is accepted.

## Consequences

The rule that several ADRs lean on exists once, and the one place the tree already departs from a
strict reading is decided rather than discovered. Option (c) would move tests that run in CI today
into the local-only set; option (b) costs reduction work per fixture; option (a) keeps today's
practice with a bound.

## Alternatives considered

- Accept the strict rule as written: contradicted by the fixtures above.
- Leave each ADR to state its own local-only clause: the status quo, with no answer for fixtures.

## Approval

Requires the project owner's acceptance and a choice among 2(a)-(c).
