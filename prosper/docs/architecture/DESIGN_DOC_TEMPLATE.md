---
kind: process
status: current
---

# Subsystem design document template

Copy this when writing a new design document for a subsystem (a `gpu/` or `subsystems/` doc that
explains how something works and where it is going), and use its sections when restructuring an
existing one. A design document explains; the binding rules it relies on live in `docs/spec/` and
the decisions behind them in `docs/adr/`, and it cites both by ID rather than restating them.

Every section is required; write "none" rather than deleting one, so a reader can tell an empty
answer from a forgotten one. The shape follows PortPS5's subsystem specs, which use the same nine
sections for every subsystem.

```markdown
# <Subsystem>

## Scope
What this subsystem owns, and its boundary against its neighbours. Which spec layer it sits in
(`docs/spec/layers.md`).

## Current state
What exists today, measured, with the commit it was measured on. Paths and issue links, no prose
claims without a source.

## Decision
The direction chosen, citing the ADR (`ADR NNNN`) and spec rule IDs it serves. If no ADR exists yet
and the choice is architectural, write one instead of deciding here.

## Target design
The structure being built toward: components, data flow, ownership of state.

## Interfaces
The functions, types and data formats other layers depend on, and which of them are stable.

## Failure modes
How each failure is reported (`FAIL-1`), what the guest observes, and what a developer sees.

## Tests
What pins the behaviour: test names, the evidence each expectation rests on (`VER-3`), and what no
test can see.

## Milestones
The ordered steps, each small enough to land as one coherent PR, and what each unblocks.

## Open questions
What is not known yet, with `CONFIDENCE` where a claim rests on thin evidence.

## Ruled out
One line per falsified hypothesis: the hypothesis, the evidence that killed it, the issue or PR.
```
