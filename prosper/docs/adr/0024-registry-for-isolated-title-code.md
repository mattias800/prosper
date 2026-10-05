---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0024: Register isolated title-specific code, with its measurement and removal condition

## Context

The charter allows exactly two forms for behaviour one title needs: a general rule the evidence
supports, or "isolated code with its measurement (`src/gpu/recompiler/gta5/`)". The ratchet freezes
the second (`title-dir`, a line cap), but nothing records *why* each piece of isolated code exists,
what measurement justified it, or what would let it be deleted. Spec rules `TITLE-1` and `TITLE-2`
carry the policy; target-tree move 10 proposes generalising `gta5/` away.

The sibling project PortPS5 handles the same escape hatch with a registry: a per-title workaround
exists only as a key named after the **mechanism** it toggles (never the game), registered in code
and in a document that must match in both directions, each entry stating why no general fix exists
yet, its tracking issue and its removal condition, and an entry no gated title uses is deleted.

## Decision

1. `docs/architecture/ISOLATED_TITLE_CODE.md` lists every isolated title-specific unit in shared
   code: its path, the mechanism it implements in engine-neutral words, the measurement and issue
   that justify it, why no general rule exists yet, and the condition under which it is removed.
2. A check compares the registry with the tree's `title-dir` rows in both directions, so isolated
   code cannot exist unlisted and a listed entry cannot outlive its code.
3. An entry whose title no longer needs it -- shown by its removal condition -- is deleted together
   with its code; the registry only ever shrinks unless a new entry passes review with its
   measurement.

Adds spec rule `TITLE-3`.

## Consequences

The one sanctioned exception to `TITLE-1` becomes auditable: every piece of it has a reason, an
owner issue and an exit. Today that is one entry (`gta5/`); the cost is small, and the value is that
the next one cannot arrive without the same record.

## Alternatives considered

- Per-game configuration files toggling behaviour, as PortPS5 also has: rejected, it conflicts with
  `TITLE-1` by turning the exception into a supported feature.
- No registry, relying on the line cap: freezes size, not justification.

## Approval

Requires the project owner's acceptance.
