## Context

<!-- Root problem, failure scenario (inputs/state -> wrong behavior), and link the issue: Fixes #NN / Refs #NN. -->

## Higher Goal

<!-- What this unlocks or protects; which title/rung or subsystem it serves. -->

## Acceptance Criteria

- [ ]

## Out of Scope

<!-- Boundaries and deferred work (file an issue for each deferred item). -->

## Summary of Changes

<!-- Files, behavior contract, invariants, risks. Affected and deliberately unaffected behavior. -->

## Verification

<!-- Exact commands and results. Quote the ctest COUNT with the exit code (--no-tests=error). -->
- Tests added:
- Red without the fix (how confirmed):
- Commands run + results:
- Could not verify (no dump / no GPU):

## Checklist

- [ ] A test that fails without the change is included, or the PR says why none can run ([prosper/tests/AGENTS.md](../prosper/tests/AGENTS.md))
- [ ] Every new file lives under `prosper/`; a new `prosper/src/**/*.cpp` touches `prosper/tests/`
- [ ] No Sony code/keys/firmware, no game data, no `*-app0` content, no absolute host paths
- [ ] No unconditional "owned" answers to entitlement/add-content queries
- [ ] Any falsified hypothesis recorded in the relevant `## Ruled out` section
- [ ] `git diff --check` clean; no `Claude-Session:` trailers on commits
- [ ] Screenshots (if progression) are direct frontend captures, with a `BLOG.md` entry
