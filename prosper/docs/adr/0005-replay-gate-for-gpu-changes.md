---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0005: Gate GPU-path changes on recorded-frame replay before merge

## Context

Snapshots are a release-time inventory, not a merge gate (charter, *Verification*), so a renderer
regression is normally found weeks after the PR that caused it. A faster instrument already exists:
`tools/gpu_replay/regress.py` (#1258) replays a corpus of recorded capsules offline and compares
each rendered hash with a committed baseline, in seconds. It is optional today, and nothing records
whether it was run.

Two constraints decide where it can run. **The corpus cannot be published**: capsules carry game
imagery and are gitignored like the dumps (`tools/gpu_replay/README.md`), so hosted CI on this
public repository cannot hold it. **CI has no GPU**: the hosted jobs run lavapipe, with an 8-lane
fragment subgroup, so even a corpus it could hold would cover a smaller domain than RADV. And a
replay hash covers the translation path only -- recompiler, decode, state resolve, executor
ordering, detile -- not live residency (#1103).

The proposal this answers came from an architecture review comparing prosper with AnyPS5, a sibling
PS5 project. It suggested replay second in the refactor order, so the later restructures (submit
worker, resource identity, command representation, shader IR) land behind a safety net. That ordering is adopted here; the hosted-CI
form it assumed is not possible for the reasons above.

## Decision

1. A PR that changes `src/gpu/`, the Vulkan backend or `frontends/shared/live/` runs
   `regress.py check --strict` against the author's local corpus and quotes the result (corpus
   size, changed and failed counts, baseline commit) in its Verification section. An intended
   change re-records with `regress.py update` in the same PR and says why each hash moved.
2. The committed baseline (`regress-baseline.json`, basename -> hash) lives in the repository so
   two lanes compare against the same hashes; the capsules never do.
3. A self-hosted runner holding the corpus and a real GPU MAY run the same check on PRs later; it
   adds no redistribution, because the corpus stays on that machine.
4. Spec `VER-1` records the rule. It does not replace the release snapshot pass, which is the
   only check of live residency.

## Consequences

Translation regressions surface at review time instead of release time, and the order of the
remaining refactors changes: this lands right after the structural moves of ADRs 0003 and 0004 and
before the performance restructures. The cost is corpus upkeep: capsules go stale as replay
semantics change and need periodic re-recording, and the corpus differs between machines, so
`--strict` plus a quoted corpus size is what keeps a shrunken corpus from passing silently.

## Alternatives considered

- Hosted CI replay of real-title capsules: rejected, it would publish game imagery.
- Hosted CI replay of synthetic capsules built from the test fixtures: possible and worth doing for
  the recompiler, but it tests hand-built inputs, not what titles emit -- a same-source control,
  which the charter warns tests the discriminator rather than the domain.

## Approval

Requires the project owner's acceptance, because it adds a merge expectation for every GPU-path PR.
