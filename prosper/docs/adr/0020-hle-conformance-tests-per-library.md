---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0020: Organise HLE tests as per-library conformance suites with stated evidence

## Context

HLE behaviour is well tested, but organised by the defect that prompted each test. `tests/hle/` has
subfolders for some libraries (`audio/`, `kernel/`, `sync/`, ...) and still holds dozens of loose
`test_*.cpp` files beside them, so there is no per-library view of which return codes and edge
cases are pinned. The charter's evidence hierarchy ranks prosper's live traces first, then
published contracts, firmware symbol data and guest disassembly; a test rarely records which of
these its expectation came from.

Wine keeps a conformance suite per DLL (`dlls/<name>/tests`) that defines the API contract and runs
on real Windows as well as on Wine. prosper cannot run tests on a PS5 (it does nothing to the
console), so its equivalent of "ran on real hardware" is a recorded trace of the real guest.

## Decision

1. `tests/hle/<library>/` mirrors ADR 0015's declaration tables, one folder per library; loose
   files move in move-only commits with `tools/refactor/classify_tests.py`.
2. Each conformance test names its evidence in a comment: a trace (title, route, capture), a
   published contract, firmware symbol data, or guest disassembly, with `CONFIDENCE` when the
   evidence is a single secondary source.
3. ADR 0007's coverage report shows, per library, functions registered and functions with at least
   one conformance test of their documented return codes.

Adds spec rule `VER-3`.

## Consequences

"What does prosper promise for `sceXxx`?" has one place to look, and an untested registered function
is visible instead of discovered. The cost is the test move, which conflicts with lanes adding tests
in the same folders and so goes library by library.

## Alternatives considered

- Keep defect-organised tests and add an index: the index would rot; the folder cannot.
- Generate conformance tests from traces automatically: worth trying later, but a trace records one
  run's values, not the contract.

## Approval

Requires the project owner's acceptance.
