---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0012: Lower RDNA2 to an SSA IR before SPIR-V, one instruction family at a time

## Context

The recompiler emits SPIR-V directly from decoded RDNA2 instructions. `emit_alu`
(`src/gpu/recompiler/rdna2_emit_alu.cpp`) is one function of several thousand lines with seven
top-level statements and hundreds of `case` labels (`docs/architecture/REFACTOR_PLAN_2026_09.md`
calls it SEAMLESS: no run of statements a tool can extract), and control flow is emitted as a state
machine (`emit_cfg_state_machine`). There is no stage at which a transform can be applied and tested
independently of emission.

The yuzu/Ryujinx and shadPS4 recompilers translate guest instructions into an SSA intermediate
representation, run optimisation and structurisation passes over it, and emit SPIR-V from the result,
with each guest instruction lowered by a small function. Structural references only.

## Decision

1. An SSA IR inside `gpu/recompiler`: RDNA2 decode -> IR -> passes (constant folding, dead code,
   structurisation) -> SPIR-V emission. The IR stays private to the recompiler.
2. Migration is per instruction family (SOP, VOP1/2/3, VOP3P, memory, ...). Each family moves when
   its instructions lower through the IR and the emitted SPIR-V for a recorded shader corpus is
   byte-identical before and after, or every difference is explained and execution-tested.
3. Each lowering is a function small enough to test alone, so `emit_alu` shrinks to a dispatch table.
4. Depends on ADR 0006: the recompiler must be a pure function first, so before/after comparison is
   a property of the input, not of the process environment.

Adds spec rule `GPU-3`.

## Consequences

The largest function in the codebase becomes a table of small lowerings, shader transforms gain a
place to live, and per-instruction tests stop going through the whole emitter. Recompilation runs
once per unique program (results are cached), so an IR's extra cost does not reach the frame. The
risk is a long period with two paths; the byte-identical gate is what keeps it from diverging.

## Alternatives considered

- Split `emit_alu` by opcode ranges into files without an IR: smaller files, same structure, and no
  place for passes.
- Rewrite the recompiler wholesale: rejected, too much verified behaviour to re-establish at once.

## Approval

Requires the project owner's acceptance and ADR 0006 first.
