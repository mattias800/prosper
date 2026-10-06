---
kind: adr
status: accepted
date: 2026-10-04
---

# ADR 0002: Reach the OS through one interface per service, one backend file per OS

## Context

The reimplemented libraries call the same few OS services -- reserve and protect memory, wait on an
address, start a thread, read a clock, open a file -- inline, inside `#ifdef _WIN32` /
`#elif defined(__linux__)` arms. A platform arm then drifts into a reduced copy of the other arm's
logic; the Windows arm of `src/hle/memory/hle_kernel_mem.cpp` stubbing two APR handlers is the
recorded case (#2384). The full survey is `docs/architecture/HOST_PLATFORM_SEAM.md`.

## Decision

As stated in `HOST_PLATFORM_SEAM.md` § Decision: one interface per service under
`src/host/platform/`, one backend file per OS chosen by the build (no runtime dispatch, no virtual
interface); new code calls the seam; existing sites migrate when touched, each move a
behaviour-neutral commit of its own; platform arms hold primitives only. Spec rules `PLAT-1` and
`PLAT-2`.

## Consequences

Portable logic is written once, and a missing port fails to link instead of surfacing title by
title. Enforced by the `platform-ifdef` and `platform-stub` ratchet rules; a reduced copy not named
as a stub remains a review rule. The interface signatures are `CONFIDENCE: MED` and are settled by
the first migration of each service.

## Alternatives considered

- Runtime-dispatched virtual interfaces: rejected, there is exactly one host per binary.
- Keeping `#if` arms and testing both: rejected, the arms diverged in practice (#2384).

## Approval

The seam document landed in #4215 (2026-10-02). The decision became charter policy ("Platform arms
hold primitives only", `CLAUDE.md`) in #4367 on 2026-10-04, alongside the `platform-stub` ratchet
rule in #4366. Recorded here as a backfill; the date is that of the charter adoption.
