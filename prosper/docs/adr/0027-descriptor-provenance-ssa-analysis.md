---
kind: adr
status: proposed
date: 2026-10-07
---

# ADR 0027: Track descriptor provenance with one SSA analysis, and resolve the rest at runtime

## Context

A shader's image, buffer and sampler accesses name their resource through T#/V#/S# words held in
SGPRs. To bind the right host resource, prosper must know where those words came from: a user-data
slot, a scalar load from a known table, or something it cannot prove. When it cannot, the draw or
dispatch is refused at recompile (fail-visible, `FAIL-1`).

**What the model is today.** Provenance is answered by a linear register-state fold
(`resolve_dynamic_fetch`, `src/gpu/execute/gpu_executor.cpp`) plus case-by-case control-flow proofs
added when a title needed one: `mapped_split_t8_reaches_use` (`src/gpu/execute/split_t8_proof.cpp`)
and the x16-load proof (`src/gpu/recompiler/smem_x16_descriptor_proof.cpp`). Each proof re-derives
reaching definitions over the decoded CFG in its own way, with its own write model and its own
limits. `docs/gpu/RESOURCE_BINDING.md` (§ Descriptor provenance) describes the contract they serve.

**Evidence (measured 2026-10-07, Windows / RTX 4070 SUPER, *Assassin's Creed IV Black Flag
Resynced* `PPSA28183`, a capture of the corrupted post-autosave frame, submit 3816):**

- **51 compute dispatches and 51 draws** in that one submit were refused at shader recompile.
- About **26 distinct compute programs** refused; **12** failed at a MIMG op with
  `[mimg-unresolved] ... srt_tag=NONE pc_res=null written=1` -- the descriptor SGPRs *were*
  written, by a scalar load whose provenance the fold could not prove.
- In **7 of the 8** programs decoded by hand, the descriptor came from one `s_load_dwordx16` that
  loads **two** T#s, in branchy code.

**Three independent gaps in the bolt-on proofs were found in one title in one day:**

1. The proof's `may_write` treated every SOP1 opcode `>= 0x20` -- which includes every
   `s_*_saveexec` -- as writing every SGPR, so the first structured branch erased all descriptor
   provenance (4 programs). Fixed by calling the shared `rdna2_escapes_decoded_effects()`; #4529 had
   already fixed the same drift in a different copy of the write model.
2. The proof was capped at 2048 dwords because it re-ran per dispatch (2 programs, 3720 and 2912
   dwords). Fixed with a per-program cache.
3. Two programs lose provenance at a CFG **join**. Still open.

Fixes 1 and 2 are #4712. The pattern is
the point: each gap is a place where one proof's private approximation of "which instruction last
wrote this SGPR on every path" disagreed with the ISA. Fixing them one at a time converges slowly
and each fix lives in only one of several copies.

**Reference designs (verification-only; described, not copied).**

- KytyPS5 lowers the shader to an SSA IR and tracks resources in a pass over it
  (`src/graphics/shader/recompiler/ir/passes/ResourceTracking.cpp`, ~2,440 lines). A descriptor is
  an SSA value, so a branch or `saveexec` cannot "kill" it: at a join it becomes a `Phi` whose
  operands are themselves traced. Where a descriptor is read from a table at a dynamic offset, the
  pass emits IR that computes a bounds-checked key at runtime and rewrites `Phi` edges onto that key
  (the indirect-descriptor projection, around lines 374-430), i.e. a static analysis with a runtime
  path for what it cannot fold.
- shadPS4 discovers sharps over its SSA IR (`src/shader_recompiler/ir/passes/resource_discover_pass.cpp`,
  `resource_patching_pass.cpp`) and flattens the indirect user-data tree into a buffer the shader
  reads (`flatten_extended_userdata_pass.cpp`). Its discovery asserts that a sharp's producer is not
  a `Phi` (`resource_discover_pass.cpp:161`, `:178`), so it shows the SSA framing without solving the
  join case this ADR targets.
- AnyPS5 runs decode -> CFG -> structurizer -> SSA IR -> constant folding -> resource tracking
  (`core/shader/recompiler/Recompiler.cpp`, `PrepareResourceProgram`) and handles the join case
  explicitly: non-invariant descriptor phis are split per predecessor
  (`Optimization/src/ResourceTracker.cpp`, `SplitDescriptorPhis`). It then specializes the IR against
  the descriptor words in guest memory, one compiled variant per snapshot -- the recompile churn the
  dynamic layer below is meant to avoid.

None of the three is evidence of PS5 behaviour; all are design references.

## Decision

Two layers, behind the existing provenance interface.

1. **Static layer: one SSA-form, phi-aware provenance analysis.** Over the decoded guest CFG,
   compute for every descriptor-consuming operand the set of possible producers of each SGPR word,
   in SSA form, with joins as phis. A phi whose arms all resolve to the same descriptor source (same
   table base and offset, or same user-data slot) resolves; a phi with distinct but each-provable
   arms resolves to a small *selected* set the recompiler can bind and select between. The write
   model is the one shared decoded-effects table (`rdna2_escapes_decoded_effects()` and its
   successors), never a per-proof copy.
2. **Computed once per program version** and cached on the decoded program, like `FoldControlPlan`
   and the split-T# per-program cache. No per-dispatch CFG walk and no size cap.
3. **One source of truth.** The recompiler (what to bind and how to select) and the executor (which
   guest resource each slot names for this dispatch) both consume this analysis's answer. They may
   not re-derive provenance independently -- the charter's "answer the same question the same way
   through every path" applied to the GPU layer.
4. **Dynamic layer: runtime lookup where the static answer is genuinely "unknown".** The shader
   reads the raw T#/V#/S# words at runtime and resolves them through a GPU-visible table, maintained
   by the emulator, keyed by guest descriptor identity (base address plus the format/dimension words
   that select a view), mapping to a bindless index of a host resource the emulator has already
   materialised (detiled, format-converted). A miss returns a defined fallback and is **counted and
   logged** with the program and pc; the dispatch is refused only when the table cannot be built for
   it at all. This is a fail-visible path, never a silent skip (`FAIL-1`). Two shapes refine it,
   and each is chosen per operand by what layer 1 proves about the descriptor's *source*:
   - **Guarded specialization (descriptor words vary rarely).** Compile the variant specialized to
     the descriptor observed, with a guard that compares the live T#/V#/S# words in the shader. A
     guard failure takes the generic runtime-lookup path in the same module; it does not recompile.
     After a small fixed number of distinct variants for one program, only the generic path is
     emitted. This is the inline-cache/deoptimization pattern of JIT compilers, and it avoids both
     per-snapshot recompile churn (AnyPS5's model) and a lookup on every access.
   - **Descriptor-table mirroring (the guest indexes a table: guest-side bindless).** When layer 1
     proves an operand is `table_base + 32 * index` read from an entry-rooted table, prosper keeps a
     host table mirroring that guest table index for index. The shader indexes the mirror with the
     guest's own index, so no hash lookup is needed. Guest writes to the table update the mirror;
     where the host cannot watch writes (Windows today), the table's bytes are compared at the
     submit that uses it. This follows how vkd3d-proton maps D3D12 descriptor heaps onto Vulkan.
     The generic keyed table remains for descriptors with no table structure.
5. **What bindless does not solve, stated so nobody treats it as the fix.** Descriptor indexing,
   bindless arrays and `VK_EXT_descriptor_buffer` replace *binding*, not *provenance*. A guest T# is
   not a host descriptor: its encoding differs, and the image it names must still be detiled and
   format-converted into a host resource before anything can sample it. So the emulator must still
   know, before the draw, *which guest images a shader may name* -- statically from layer 1, or from
   the runtime table's population in layer 2. Neither layer lets the shader point at raw guest memory.

Adds spec rule `GPU-4`.

## Consequences

**Easier.** A new branch shape, `saveexec` form or join stops being a new proof; it is either
handled by the analysis or it is a bug in one place with one test. The executor and recompiler can
no longer disagree about which resource a slot names. Programs refused today because a provable
descriptor crosses a join (gap 3) become bindable.

**Costs.**

- *Compile time:* an SSA construction and a provenance fixpoint per program version. Bounded by
  program size and paid once (cached), off the submit thread per ADR 0014. Not yet measured;
  the A/B below must report it.
- *Memory:* per-program SSA state and the cached answer; the runtime table in layer 2 (one entry per
  live guest descriptor identity) and its GPU-visible copy.
- *Migration:* the analysis must reproduce every answer the fold and both proofs give today before
  any proof is deleted.
- *Runtime (layer 2 only):* an extra table read and compare per access through an unresolved
  descriptor, and host work to keep the table populated. Only paid where layer 1 says "unknown".
  A guarded specialization costs one compare of the descriptor words per access on the hit path.
  A mirrored table costs one indexed host-descriptor read, plus keeping the mirror current (a
  write-watch or a per-submit compare of the table bytes).

**Risks.** A wrong static answer binds the wrong resource -- worse than a refusal, because it
renders. The analysis therefore answers only what it proves; anything else is "unknown" and goes to
layer 2 or fail-visible. Layer 2's population policy (which guest images enter the table) is the
hard part and is where a stale-content bug would hide; it should reuse ADR 0010's canonical resource
identity rather than inventing a second one. `CONFIDENCE: MED` that a runtime table keyed on
descriptor words is sufficient for all titles; the keying is an open question below.

**What stays.** The register fold and the existing proofs remain authoritative until the analysis
subsumes each of them under the A/B gate. Refusal stays fail-visible throughout.

**Relation to other ADRs.** ADR 0012 introduces an SSA IR for emission; this analysis should run on
that IR once it exists, but does not wait for it -- it can build SSA over the decoded CFG directly.
ADR 0006 (pure recompiler) makes the cached answer a property of the program bytes.

## Alternatives considered

- **Keep adding per-pattern proofs.** Three gaps in one title in one day, each a divergent copy of
  the write model or a resource cap, is the evidence against. Every new engine brings new shapes.
- **Dynamic-only (always read descriptor words at runtime).** Removes the static question but pays a
  lookup on every access, and the emulator still has to know which guest images to materialise --
  which is the static question again, now answered late and per frame. Kept as the fallback layer.
- **Descriptor heap / bindless only.** Solves binding-slot pressure, not provenance (Decision 5).

## Migration order

1. Build the SSA provenance analysis behind the current interface, cached per program version, with
   unit tests on hand-built CFGs: a two-arm `saveexec` diamond, a join of two provable T#s from one
   `s_load_dwordx16`, a loop back-edge, and a join of a provable and an unprovable arm (must stay
   "unknown").
2. Shadow mode: run it alongside the fold and proofs and log every disagreement, per program. Any
   case where the analysis claims a source the old path refuted is a bug to fix before going on.
3. A/B on the snapshot titles and on the `PPSA28183` capture: refused-dispatch and refused-draw
   counts, compile time, and replay output (ADR 0005). Switch the default only when no guarded
   title's output changes unexplained.
4. Delete the proofs one by one, each in its own PR with the A/B showing it is subsumed.
5. Layer 2 last, for the residue the analysis cannot prove, with a counter in the
   `[perf-alarm]` summary for runtime-table misses. Build order within it: descriptor-table
   mirroring first (it is exact and has the clearest model), then guarded specialization for the
   rare-variant case, then the generic keyed table for what remains. Each shape is A/B'd as above,
   with guard-failure and miss counts reported.

## Evidence still needed / open questions

- How many of the ~26 refused `PPSA28183` programs layer 1 alone recovers; the decoded 8 suggest
  most, but the 12 MIMG cases are the only ones classified.
- Runtime-table key: are base address plus format/dims words enough to identify a view, or do some
  titles alias one allocation with different swizzle/mip ranges that must key separately?
- Layer 2 population: materialise on first miss (one frame late, visible) or speculatively from
  memory that has ever been bound as an image?
- Whether VGPR-sourced (per-lane, non-uniform) descriptors occur in the corpus; they need
  `nonuniform` indexing in layer 2 and cannot be answered statically.
- Whether the analysis should move into ADR 0012's IR when that lands, or stay on the decoded CFG.

## Approval

Requires the project owner's acceptance. Acceptance unblocks step 1; deleting any existing proof
(step 4) additionally requires its A/B. Layer 2 is a separate decision point after step 4.
