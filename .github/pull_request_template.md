## Context

<!-- Root problem, failure scenario (inputs/state -> wrong behavior), and link the issue: Fixes #NN / Refs #NN. -->

## Higher Goal

<!-- What this unlocks or protects; which title/rung or subsystem it serves. -->

## Acceptance Criteria

- [ ]

## Out of Scope

<!-- Boundaries and deferred work; link existing issues or use the authorized issue workflow. -->

## Summary of Changes

<!-- Files, behavior contract, invariants, risks. Affected and deliberately unaffected behavior. -->

## Verification

<!-- Exact commands and results. Quote the ctest COUNT with the exit code (--no-tests=error). -->
- Tests added:
- Red without the fix (how confirmed):
- Commands run + results:
- Could not verify (no dump / no GPU):

## Checklist

- [ ] Behavior changes have a meaningful regression; nonbehavioral scope and execution limits are stated ([test rules](https://github.com/mattias800/prosper/blob/main/prosper/tests/AGENTS.md))
- [ ] New source/test files live under `prosper/`; repository metadata uses the contribution-shape allowlist; a new `prosper/src/**/*.cpp` touches `prosper/tests/`
- [ ] No Sony code/keys/firmware, no game data, no `*-app0` content, no absolute host paths
- [ ] No unconditional "owned" answers to entitlement/add-content queries
- [ ] Any falsified hypothesis recorded in the relevant `## Ruled out` section
- [ ] `git diff --check` clean; no `Claude-Session:` trailers on commits
- [ ] Screenshots (if progression) are direct frontend captures, with a `BLOG.md` entry
