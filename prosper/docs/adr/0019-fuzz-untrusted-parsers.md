---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0019: Fuzz every parser of untrusted bytes

## Context

prosper parses bytes it does not control on every launch: SELF/ELF images (`src/self/module.cpp`),
PM4 command streams (`src/gpu/pm4/command_processor.cpp`), RDNA2 shader binaries (the recompiler's
decoder), and its own capture files (`src/gpu/capture/serialize/capture_deserialize.cpp`, whose
`deserialize_gpu_capture` is a 1,425-line body per `REFACTOR_PLAN_2026_09.md`). CI runs ASan, UBSan
and TSan, but only over hand-written tests, and there are no fuzz targets in the repository. The
loader's refusal of truncated segments (#2631) is an example of a defect class -- a malformed input
silently mapped as zeros -- that fuzzing finds directly.

Mesa, browser engines and most mature parsers run coverage-guided fuzzing under sanitizers.

## Decision

1. A libFuzzer target per parser: SELF/ELF, PM4 decode, RDNA2 instruction decode, capture
   deserialisation. Each feeds bytes to the same entry point production uses and asserts only what
   the parser promises (no crash, no sanitizer report, refusal instead of a partial result).
2. Seed corpora are synthetic, built from the test fixtures and minimised. **No game bytes are
   committed or uploaded**, so the corpus never redistributes content.
3. CI builds the targets with clang and runs each briefly per PR as a smoke check; longer runs are
   scheduled, and every crash becomes a regression test before its fix merges.

Adds spec rule `VER-2`.

## Consequences

Parser defects that real dumps have not yet exercised are found before a title does. CI gains a
clang build of a few targets (the main jobs use gcc, which has no libFuzzer). Fuzzing a decoder
that is already a 1,000-line body will also make its splitting safer, because a fuzz corpus is a
behaviour check that does not depend on a title.

## Alternatives considered

- Rely on real dumps as the test corpus: they cover what titles do, not what malformed input does,
  and cannot be committed.
- AFL++ instead of libFuzzer: either works; libFuzzer integrates with the existing CMake and
  sanitizer setup with fewer moving parts.

## Approval

Requires the project owner's acceptance, because it adds a clang build to CI.
