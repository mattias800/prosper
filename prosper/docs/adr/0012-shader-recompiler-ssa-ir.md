---
kind: adr
status: accepted
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
5. **Small files and small functions are an acceptance criterion of every family, not later cleanup**
   (owner, 2026-10-15). Today `rdna2_emit_alu.cpp` is about 9,400 lines and `rdna2_emit_cfg.cpp`
   about 7,400. Files that size cannot be tested in pieces, and every edit recompiles one huge
   translation unit, which costs build and CI time. So:
   - New code lives under `gpu/recompiler/ir/`, split by role (IR core, lowering, passes, SPIR-V
     emission), with one file per instruction family or sub-family.
   - A file aims at 500 lines or fewer and must not pass 1,000. A function aims at 80 lines or fewer.
     The first family's PR adds a directory-scoped ratchet rule for the 1,000-line cap, so CI holds
     the line.
   - Each lowering has its own focused test file.
   - Each family PR deletes the cases it migrated from the old emitter, quotes the old files' line
     counts before and after, and lowers their ratchet rows with `--update`.
   - A mechanical split of an existing large file, done with `tools/refactor/`, is welcome along the
     way, in its own commit and kept apart from behaviour changes.

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

Accepted by the project owner on 2026-10-15. The first family is the scalar ALU (SOP1/SOP2/SOPC/SOPK), which also
resolves the duplicated folder and emitter semantics in #2514.
