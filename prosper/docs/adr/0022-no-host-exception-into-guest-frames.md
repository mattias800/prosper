---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0022: No host C++ exception unwinds into guest frames

## Context

Guest code calls prosper's HLE handlers directly and runs on the same stack; many titles are native
C++ themselves (every Unity title through IL2CPP) and carry their own exception tables and
`catch (...)` blocks. A host exception thrown inside a handler, or inside code a handler reaches
(the GPU submit path, the recompiler), and not caught in prosper code reaches the HLE boundary with
guest frames above it on the stack. prosper registers no guest unwind information with the host
unwinder: there is no `__register_frame`, `RtlAddFunctionTable` or
`RtlInstallFunctionTableCallback` anywhere in `prosper/src`. So the expected outcome is not a guest
`catch` running: on Linux the unwinder cannot step through the guest frame and the process ends in
`std::terminate`; on Windows x64 a frame with no function-table entry is treated as a leaf, so the
unwind goes wrong in a way that depends on the stack. Either way the failure is not the visible,
logged one `FAIL-1` requires, and it is reported far from the call that caused it.
`CONFIDENCE: LOW` on the exact Windows behaviour: it has not been measured. A test that throws from
a handler called through a guest-ABI frame, on each host, would settle it.

The sibling project PortPS5 records the same hazard in its technical-debt ledger, with a per-module
count of host throws that can reach a guest-callable export, which is where this was noticed.

Measured on this branch with the ratchet rule below: **23 `throw` sites in 10 files** under
`src/{hle,gpu}` (comments and strings excluded), mostly capture codecs and recompiler choice
parsing. A count of sites, not of escapes: several are caught before they leave prosper code.

## Decision

1. Code in `src/{hle,loader,self,gpu}` reports failure through return codes and the logging abort
   path, not through exceptions that can leave the HLE call. Spec rule `HLE-5`.
2. The `host-throw` ratchet rule counts `throw` sites in those roots, one row per file, and may only
   fall. A throw that is caught before it leaves prosper code may stay, with a comment saying where
   it is caught, and its row raised in the PR that adds it.
3. Where an exception must be used internally (a third-party API that throws), the HLE entry that
   reaches it catches everything and converts it to the call's error return plus a log line.

## Consequences

A class of silent failure, the guest catching prosper's error, cannot grow. Removing the existing
sites is ordinary work on the files that hold them. The count is a proxy: it cannot tell a caught
throw from an escaping one, which is why the rule allows a documented, caught throw.

## Alternatives considered

- Catch-all guards at every HLE entry: one per handler, and they hide which call failed unless each
  logs; they are the fallback in point 3, not the rule.
- Compile host code with `-fno-exceptions`: the standard library and some dependencies still throw,
  and it would turn every remaining throw into an abort at build time across all layers at once.

## Approval

Requires the project owner's acceptance; it adds a twelfth ratchet rule.
