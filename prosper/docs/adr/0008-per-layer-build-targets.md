---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0008: Enforce the layer order in the build with one CMake target per layer

## Context

The core is one target: `file(GLOB_RECURSE PROSPER_SRC CONFIGURE_DEPENDS src/*.cpp)` builds every
layer into `prosper_core`, whose include root is all of `prosper/src`. Any file can therefore
include any header, and the layer order (ADR 0001, spec `LAY-1`) is held only by a text scan of
`#include` lines, which misses what it does not parse and cannot see a dependency introduced
through a macro or a forward declaration. LLVM, Chromium and most large C++ codebases enforce
layering in the build instead, where an edge the graph does not allow fails to compile.

## Decision

Spec `LAY-5`. Each layer becomes a static library target (`prosper_self`, `prosper_loader`, ...)
whose sources are that layer's folder and which links `PRIVATE` only to the layers below it, with
per-target include directories so a header from a layer not linked is not on the path. A layer is
split out only once its baselined `layer-include` inversions are zero; until then it stays in the
monolithic target and the ratchet holds it. `self` has no inversions and is first; `input` follows.
`gpu/recompiler` becomes its own target once ADR 0006 makes it pure.

## Consequences

An inversion becomes a compile error at the moment it is written, and the target graph is the
architecture diagram rather than a description of it. Link order and static-initialisation
registration need care: HLE handlers that register themselves from static initialisers can be
dropped by the linker from a static library nothing references, so registration must stay
reachable (an explicit registration call, or an object library). Each split is its own PR, built
on Linux, Windows MinGW and macOS before merge.

## Alternatives considered

- Keep the text ratchet as the only enforcement: it stays for layers not yet split, but it is a
  proxy and remains one.
- Split every layer at once: rejected, layers with inversions would not link.

## Approval

Requires the project owner's acceptance; it changes the build structure every platform job uses.
