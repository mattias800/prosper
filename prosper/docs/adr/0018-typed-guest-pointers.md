---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0018: Typed guest pointers at the HLE boundary

## Context

The guest runs in prosper's own address space, so an HLE handler can dereference a guest pointer
directly, and today handlers do: there is no guest-pointer type in `src/`. That is fast, and it
means a bad guest pointer, or a handler that reads past a guest structure, faults inside prosper at
a site unrelated to the call that caused it. Handler signatures also do not say which arguments are
guest memory, how large the referenced object is, or whether it is read or written.

RPCS3 passes guest memory to its HLE functions as `vm::ptr<T>`; Ryujinx and others use similar
typed views. The type documents the contract and gives one place to validate and trace accesses.
Structural reference only.

## Decision

1. `GuestPtr<T>` and `GuestSpan<T>` in `guest/memory` (ADR 0003): a guest address typed with what it
   points to, with explicit `read()`, `write()` and span access. In release builds they compile to a
   raw pointer access; in diagnostic builds an access is checked against the guest memory map and
   an invalid one reports the handler, argument and address before anything faults.
2. Handler arguments that are guest memory use these types. New handlers start with them;
   existing ones convert when touched, or when ADR 0015's tables give them a signature.
3. The relay trace (ADR 0015) prints a `GuestPtr` argument's contents for small types, which
   makes argument-level tracing useful rather than a list of addresses.

Adds spec rule `HLE-4`.

## Consequences

Guest memory contracts become visible in signatures, a class of host faults becomes a named report
at the call that caused it, and the relay trace can show what a call actually received. The release
cost is zero by construction; diagnostic builds pay a map lookup per access. `CONFIDENCE: MED` that
every access pattern fits the types: handlers that walk guest linked structures need span helpers.

## Alternatives considered

- Validate pointers inside each handler: repeated and inconsistent, which is the status quo.
- Always-on validation: a map lookup on hot HLE paths conflicts with `PERF-P4`.

## Approval

Requires the project owner's acceptance and ADR 0003 for the type's home.
