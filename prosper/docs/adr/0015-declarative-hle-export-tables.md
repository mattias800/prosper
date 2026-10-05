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

The calling-convention half already exists and is generated, not hand-written: on Windows every
import stub is emitted by `abi::emit_sysv_to_ms_bridge` (#2955), with variadic calls on the generic
`guest_varargs` path (#3246; `src/host/abi/AGENTS.md`). Handlers registered through
`Hle::register_typed` -- those taking or returning a float or double -- carry a `CallSignature` from
`abi::signature_of`, which records only the argument count and which arguments are floating-point
(`src/host/abi/call_signature.hpp`). Every other handler is registered through `R()` /
`register_fn` with an `(HleFn)` cast, which erases its declaration, and takes the default integer
placement, which is the same under both conventions (`src/hle/dispatch/dispatch.hpp`).
Call tracing exists in pieces: `tools/hle_calls` histograms calls, return values (`--values`, #2075)
and the bytes behind a pointer argument (`--out-bytes N`), and `PROSPER_SVCLOG` logs at the
`svc_log` call sites. What does not exist is a generic, typed trace of every call's decoded
arguments, because for most handlers the registry keeps no argument types at all, and even a
`CallSignature` records only which arguments are floating-point, not what a pointer refers to or how
large it is.

Wine declares every DLL's exports in a `.spec` file from which the build generates the export
table, the calling-convention thunks and the `+relay` trace; an unimplemented export is declared as
a stub that fails loudly. Structural reference only.

## Decision

1. Each reimplemented library gets one declaration table -- `src/hle/<library>/exports.inc` or an
   equivalent constexpr array -- listing the Sony function names it implements and their handlers.
   Registration iterates the tables.
2. The table holds names and handler symbols only. NIDs are derived by `nid_hash` at build or start
   time, so no firmware symbol data is committed (the 3.20 library dump stays a gitignored sibling).
3. Libraries move out of `hle_service.cpp` into `src/hle/<library>/` one at a time, each move a
   move-only commit with its table.
4. ADR 0007's coverage report reads the tables and the registry, so "implemented" has one source.
5. A table entry may declare argument semantics the C++ type cannot carry: which arguments are
   guest pointers, what they point to, and their sizes (`GuestPtr`/`GuestSpan`, ADR 0018). The
   existing `emit_sysv_to_ms_bridge` keeps generating the Windows bridge; the table does not
   replace it. From the declared semantics, one relay trace logs every
   call with its decoded arguments and return code behind one logging channel (ADR 0017), the way
   Wine's `+relay` does from its `.spec` files; it supersedes the partial views `tools/hle_calls
   --values` / `--out-bytes` and `PROSPER_SVCLOG` give today.

Adds spec rule `HLE-3`.

## Consequences

Finding a handler becomes opening its library's table; a duplicate or misspelt registration becomes
a build-time or startup error rather than a silent miss. A typed trace of decoded arguments, which
today's tools approximate from registers and return values, comes for every declared function at no
per-function cost. The tables also make the per-library
surface reviewable in one diff. Registrations that are genuinely computed (loops, aliases) need a
table form that expresses them, and `nid_census` documents that such registrations exist.

## Alternatives considered

- Generate the tables from the firmware library dump: rejected, its contents may not be committed,
  and generating from it would also declare functions prosper does not implement.
- Keep imperative registration and only split files: smaller files, still no list of the surface.

## Approval

Requires the project owner's acceptance.
