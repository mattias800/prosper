---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0015: Declare each library's HLE exports in one table per library

## Context

Handlers are registered imperatively, in registration functions spread over large files (for
example `#define R(str, fn) Hle::register_fn(nid_hash(str), ...)` in
`src/hle/kernel/hle_kernel.cpp`, and `hle_service.cpp`, which `REFACTOR_PLAN_2026_09.md` counts as
fifteen Sony libraries in one file). The NID is derived from the function name by `nid_hash`, so a
registration needs only the name. There is no single place that says which functions of a library
prosper implements, and finding a handler means grepping.

Wine declares every DLL's exports in a `.spec` file from which the build generates the export
table; an unimplemented export is declared as a stub that fails loudly. Structural reference only.

## Decision

1. Each reimplemented library gets one declaration table -- `src/hle/<library>/exports.inc` or an
   equivalent constexpr array -- listing the Sony function names it implements and their handlers.
   Registration iterates the tables.
2. The table holds names and handler symbols only. NIDs are derived by `nid_hash` at build or start
   time, so no firmware symbol data is committed (the 3.20 library dump stays a gitignored sibling).
3. Libraries move out of `hle_service.cpp` into `src/hle/<library>/` one at a time, each move a
   move-only commit with its table.
4. ADR 0007's coverage report reads the tables and the registry, so "implemented" has one source.

Adds spec rule `HLE-3`.

## Consequences

Finding a handler becomes opening its library's table; a duplicate or misspelt registration becomes
a build-time or startup error rather than a silent miss. The tables also make the per-library
surface reviewable in one diff. Registrations that are genuinely computed (loops, aliases) need a
table form that expresses them, and `nid_census` documents that such registrations exist.

## Alternatives considered

- Generate the tables from the firmware library dump: rejected, its contents may not be committed,
  and generating from it would also declare functions prosper does not implement.
- Keep imperative registration and only split files: smaller files, still no list of the surface.

## Approval

Requires the project owner's acceptance.
