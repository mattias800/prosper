---
kind: adr
status: proposed
date: 2026-10-18
---

# ADR 0038: Make one dataflow analysis the owner of register validity facts in the recompiler

## Context

ADR 0012 (accepted) moves the RDNA2 recompiler onto an SSA IR. `docs/architecture/IR_CONTROL_FLOW_PLAN.md`
orders the control-flow half of that move into twelve steps. Step 8 makes one analysis the single
owner of validity facts, and the plan says that needs its own ADR. This is that ADR.

Measured on `origin/main` at `d9d00433d` (2026-10-18). Line numbers refer to that commit.

**Where the plan stands.** Steps 1 to 6 have merged: the guest CFG and acyclic region tree (#5417),
if regions walked by `emit_structured` (#5426), the mechanical split of the top-level CFG helpers
(#5418), loop regions and the loop cursor (#5425, #5432), dispatcher blocks from `GuestCfg` (#5435)
and SSA state placement (#5437). They live in `src/gpu/recompiler/ir/cfg/`. Each step is guarded:
the emitter takes the new structure only where it equals the old one, so the corpus gate (ADR 0006)
reports `changes: none`. Step 7, EXEC as an IR value with static identity (`ExecValue`), has not
merged. No `ir/analysis/` directory exists yet.

### Two models carry the same facts

A *validity fact* says, for one register word at one program point, whether its lanes hold values
prosper can stand behind, in which lanes, and under which assumptions. Two models hold these facts
today.

- **The structured emitter records facts while it emits.** `emit_alu` updates
  `ScalarSourceMarks` (`rdna2_vector_word_marks.hpp:18`) as a side effect, and the joins
  (`join_vector_word_marks`, `merge_edge_marks`, `join_structured_saved_masks`, `join_lane_slots`)
  combine them at merges. A mark has eight fields: `placeholder`, `memory`, `hard`, `arbitrary`,
  `blanket_roots`, `outside_exec` (the owner), `scoped_complete_word` and `scoped_blanket_roots`.
  The owner is a **SPIR-V id**, as is `RegState::exec_subset_of`. Loop facts come from a
  single-pass optimistic check: assume at the header, emit the body, check at the back edge with
  `LoopBlanketCheck` (`rdna2_to_spirv_internal.hpp:3579`), and refuse when the check fails.
- **The dispatcher re-derives facts in a pre-pass.** `emit_cfg_state_machine`
  (`rdna2_emit_cfg.cpp:75-5126`, 5,052 lines) runs its own transfer functions over its blocks:
  the `advance_wave64_b64_masks` lambda (1202-1724, 523 lines), worklist solvers at 967 (entry M0),
  1727 (Wave64 MUST) and 2046 (lane-slot invalidation), plus `SpilledMaskHalves`,
  `SpillSlotDomains`, `ActiveVectorWords`, the Wave32 dispatch words and the mask-half aliases in
  their own files. `load_state` (2496-2752) then builds a fresh `RegState` at every case and does not
  copy the marks. `mark_relayed_words` (`rdna2_spilled_mask_halves.cpp:342`, called at 2641)
  re-creates four flags: `placeholder`, `hard`, `arbitrary` and `memory`. Blanket roots and scoped
  roots are not carried. The owner is re-anchored to the reloaded EXEC only for the words
  `ActiveVectorWords` proves (`restore_active_vector_words`, `rdna2_active_vector_words.cpp:123`,
  called at 2741 and 2744).

So the dispatcher's model is coarser by construction, and every fact must be implemented twice:
once as a mark transfer and once as a pre-pass transfer plus its reload.

### The bugs this split caused

- **A fact one model carries and the other drops or re-derives differently.** #4949 (preserve
  physical DATA and prove current predicate inputs); #5172 (EXEC confinement lost through four
  lane-wise relays); #5173, fixed by #5179 (dispatcher block entries drop vector-word EXEC
  confinement; seven *Kena* compute programs were blocked on this alone); #5175 and #5178 (reloaded
  spilled mask halves consumed as a mask); #5181 and #5184 (`ActiveVectorWords` loses `current`
  across an EXEC-or-zero select spilled through `v_writelane`); #5183 and #5188
  (`ds_read_addtid_b32` never completes an active word). #5174 with #5180, and #5185 with #5186, are
  the policy decisions below, and each exists twice: #5180 changed both `rdna2_vector_word_marks.cpp`
  and `rdna2_emit_cfg.cpp`, and the dispatcher's relay restates the arbitrary-propagation rule in
  its own words (`rdna2_spilled_mask_halves.cpp:224`). Of the numbers from #5172
  to #5188, 13 belong to this class. The other four do not: #5176 and #5177 are CI disk space, #5182
  is an image VGPR count and #5187 accepted ADR 0012.
- **A fact one model cannot state.** #5202's second gap is a narrowed complete write of an
  arbitrary source over a hard old word. The honest fact is "arbitrary inside the owner, hard
  outside it". `ScalarSourceMarks` has one definedness per word, so the owner is dropped and a
  later read is refused. The same lane-confinement fact refuses the GTA Wave64 VCC_LO dispatch
  fixture in `rdna2_to_spirv_exec` (#5039).
- **A write model stated by hand that disagrees with the shared table.** #5436 and #5439: the
  structured emitter's phi scan (`loop_written_regs`) charged one SGPR for 64-bit SOP2 and SOP1
  results that write two. Step 6's placement, which reads the instruction-effects table, found it.
  This class is not caused by the two fact models, but it shows the fix shape this ADR uses: state
  each instruction's effects once.

### How many validity analyses there are

#5146 says 37 of the recompiler's files are separate validity analyses. It does not give a
method, so this was re-counted two ways.

- **By file name.** The top-level files of `src/gpu/recompiler/` whose names match
  `marks|words|mask|proof|provenance|definedness|carry|subset|domain|alias|lifetime|arbitrary|facts|requirements|_data|exec_|observab|participation|equality|widening|narrowing|old_word|complement`:
  **51 files, 15,193 lines**. The filter is generous. It catches small data files
  (`fragment_entry_facts`, 23 lines) and misses analyses that are not files (the dispatcher's
  lambdas). So 37 is the right order of magnitude, and nothing below depends on the exact figure.
- **By write model.** **27 of the 51** call no function declared in
  `rdna2_instruction_effects.hpp` and none of the shared write helpers
  (`rdna2_escapes_decoded_effects`, `vector_write_completes_active_word`,
  `vector_read_completes_active_word`, `scalar_alu_source_words`, `scalar_write_width`). Each of
  those states instruction effects by hand. The other 24 call a shared helper for some instructions,
  and that alone does not show their whole transfer is table-driven.

### What the instruction-effects table covers

The plan's transfer is "driven by `rdna2_instruction_effects`". Today the table cannot drive a
complete transfer. `range_for` (`rdna2_instruction_effects.cpp:1346`) tabulates SOP2, SOP1, SOPK,
SOPC, SOPP, SMEM, VOP1, VOP2, VOPC, VOP3 and VOP3P. **DS, MUBUF, MTBUF, MIMG, FLAT, EXP and VINTRP
have no rows.** VALU rows have status `Valu`, whose scalar destination and implicit writes "are NOT
described and read as zero" (`rdna2_instruction_effects.hpp:41`). That covers the VOPC SDST, the
VOP3B carry-out and the `v_readlane` SGPR. The #5193 policy needs exactly the missing memory facts:
load versus store versus atomic, whether prosper reproduces the out-of-bounds result, and whether an
address reaches other lanes.

### Gate coverage

The local corpus has 1,003 distinct programs (manifest generated 2026-10-14). The number with at
least one recorded-input sidecar is **compute 363 of 518, fragment 247 of 305, vertex 141 of 180**.
Compute programs without a sidecar compare verdicts only, so 155 compute programs, 30%, are
verdict-only. Vertex replays mostly end `incomplete`. The dispatcher route, which steps 9 and 11
change most, is almost entirely compute: 366 of the 373 programs that try it.

## Decision

1. **One lattice.** Every state word has one fact per program point. The state words are each SGPR,
   VGPR, VCC, EXEC, M0, SCC, `v_writelane` spill slot and LDS addtid slot.

   ```cpp
   struct WordFact {
       Domain domain;            // Data, Mask, MaskHalf{lo|hi, instance}, M0Token, Ambiguous, Absent
       Validity inside;          // lanes inside `owner`
       Validity outside;         // lanes outside `owner`
       ExecValueId owner;        // kAll for a whole-wave fact
       bool loaded;              // the value came from memory (provenance, not validity)
       RootSet assumptions;      // loop-header assumptions the fact depends on
   };
   enum class Validity : uint8_t { Defined, Arbitrary, Hard };   // join = the later of the two
   ```

   `Validity` is a chain. **Defined** means prosper holds the hardware value. **Arbitrary** means
   hardware holds a value nothing constrains, and prosper holds one value hardware could have held
   and uses it consistently. **Hard** means hardware holds a real value that prosper does not know.
   A fact that holds only under loop assumptions is `Defined` or `Arbitrary` with a non-empty
   `assumptions` set. That is today's soft placeholder.

   This departs from the plan's sketch on one point. The plan lists `Memory` as a fourth
   definedness. In the code, `memory` is orthogonal to validity: "MEMORY remains a separate
   classification, never proof that the address was supplied"
   (`rdna2_vector_word_marks.cpp:331`). So it is a separate field.

   `owner` is an `ExecValue` (plan step 7), never a SPIR-V id. So **step 7 is a prerequisite**, and
   the analysis cannot start before it.

2. **One transfer function, driven by the instruction-effects table.** Each instruction's reads,
   writes, widths, lane flow, EXEC effect and memory behaviour come from
   `rdna2_instruction_effects`. The transfer adds the validity rules on top of those effects and
   does not restate them. First the table is extended to the untabulated formats, with a coverage
   test that fails on a decodable opcode with no row (#5146 step 4). Those formats are DS, MUBUF,
   MTBUF, MIMG, FLAT, EXP and VINTRP. The VALU scalar destinations and implicit writes are added too.
   **An instruction the table cannot describe makes every word it may write Hard.** It never makes
   one Defined. The transfer replaces `advance_wave64_b64_masks`, the Wave32 dispatch-word
   transfer, the `SpilledMaskHalves`, `SpillSlotDomains` and `ActiveVectorWords` transfers, the
   mask-half alias analysis and the `record_*` side effects in `emit_alu`.

3. **A real loop fixpoint.** A loop header's facts are the join of the preheader and back-edge
   facts, iterated until stable. The lattice has finite height and the assumption roots are bounded
   by the loop headers, so the iteration terminates. `LoopBlanketCheck`'s per-root voiding stays:
   an assumption that fails voids only the reads that relied on it. The solver iterates instead of
   refusing after one pass.

4. **Solved once per compiled region, before emission. Consumers query and do not record.** The
   analysis produces `facts_at_entry(block)` and `facts_at(block, inst)`, with joins at merges and
   an edge transfer for the `s_cbranch_execz` skip edge (`confine_execz_skip_marks`'s rule). The
   solver stores facts at block entries. A query inside a block replays the block's transfers, so
   memory stays bounded for regions of up to 248 blocks (the largest dispatcher region in the
   plan's census). The `emit_alu` admission checks read `facts_at`. These include
   `fresh_mask_sources_defined`, `vector_word_defined_in_exec`, `active_vector_sources_defined` and
   the mask-domain reads. `load_state` materializes `facts_at_entry` and stops filtering through
   parallel vectors. Building the analysis calls nothing on `SpirvCompute`, so it cannot renumber
   SPIR-V ids. The analysis is a pure function of the program and the request (GPU-1). The same
   facts serve the structured route and the dispatcher route, so no route can have a fact that the
   other route lacks.

5. **The validity policy the analysis must preserve.** Each rule below is an existing decision.
   The analysis encodes it once, and its refused twins stay refused.
   - **Never-written words that hardware does not load are Arbitrary** (#5174, implemented by
     #5180). This applies only when the stage's explicit launch inventory (`rdna2_launch_vgprs`)
     says hardware does not load the word. With an unknown or partial launch profile, an absent word
     stays Hard.
   - **Hardness taints and arbitrariness propagates** (#5185, implemented by #5186). A lane-wise
     operation's result lane is the join of the lanes it reads. A cross-lane operation (DPP,
     `v_readlane`, permute, `ds_swizzle`, ...) joins over every lane it can read, so one hard lane
     makes every result lane Hard. Arbitrary joined with Defined is Arbitrary. Arbitrary joined with
     Arbitrary is Arbitrary. Anything joined with Hard is Hard.
   - **Data loaded at arbitrary addresses is Arbitrary** (#5193). This applies to reads only, and
     every condition must hold. The descriptor and every scalar address input are not Hard. The
     form is a bounds-checked, side-effect-free load or sample whose out-of-bounds result prosper
     reproduces; stores, atomics, GDS and forms whose result depends on other lanes' addresses
     (`ds_bpermute`, append/consume) are Hard. The rule applies per lane: a lane with a defined
     address keeps its ordinary loaded-data status.
   - A read is admitted when its lanes are Defined, or Arbitrary where the consumer allows arbitrary
     input (fresh-mask sources and #5193 addresses). Anything else refuses visibly (FAIL-1).

6. **Spec.** Adds rule `GPU-6`: a register word's validity has one analysis, queried by every
   route. It does not change `GPU-4`. Descriptor provenance (ADR 0027) stays a separate analysis,
   and the plan keeps it out of scope. The two analyses may later share the CFG and the effects
   table.

## Migration contract

The plan's steps, with the gates this ADR makes binding.

| Step | What | Gate |
| --- | --- | --- |
| 7 | `ExecValue` identities and static subset/complement relations | as in the plan; prerequisite |
| 8a | Extend the effects table (Decision 2) | coverage test. Byte-identical corpus gate, because existing callers now get answers where they got `Unknown`; every changed outcome is explained |
| 8b | `ir/analysis/`: lattice, transfer, fixpoint. **Shadow only: nothing consumes it** | census (below); corpus gate `changes: none` |
| 9 | The dispatcher consumes `facts_at_entry`; its pre-pass is deleted | no open contradiction; flip groups witnessed |
| 10 | The structured path consumes `facts_at`; `record_*`, `join_*` and `LoopBlanketCheck`'s single pass go. The two-sided mark (#5202 gap 2) arrives here | as step 9, plus the parked tests |

**The census (step 8b).** At every point where an existing consumer consults a fact, the census
records `(program, pc, word, old verdict, new verdict, reason)`. It runs in `shader_inspect` over the
corpus, and as a ctest over the committed `tests/data/*.hex` fixtures. Each consult point is one of
three:

- **equal**: same verdict.
- **refines**: the new analysis admits where the old model refuses, **and** the census names the
  fact the old model lost from a closed list of reasons. The list starts as: owner dropped at a
  reload, relay coarsened to four flags, single-pass loop check, and two-sided owner needed. Adding
  a reason is a reviewed change.
- **contradicts**: the new analysis refuses where the old model admits, or admits with no listed
  reason. **Any contradiction blocks step 9.** It is resolved by attribution, never by tuning the
  census. Either the new analysis is incomplete (fix it), or the old model admits wrongly. In that
  case the old model is fixed or the read is refused on `main` first, with its own issue, because
  that is a live wrong-acceptance.

The census also reports how many programs and consult points it reached, per stage, split between
live inputs and synthetic inputs. A program that refuses early under synthetic inputs never reaches
its later consult points, and the report must say so rather than count it as equal.

**Verdict flips (steps 9 and 10).** These steps knowingly change verdicts where the old model lost a
fact. Each PR groups its corpus diff by census reason. **Each flip group needs an execution
witness**: a test that runs production `emit_body` output and checks the guest-visible result
against a reference derived from the ISA. The closed SPIR-V interpreter harness from #5436/#5439
(`tests/gpu/recompiler/test_if_merge_scalar_pair_writes.cpp`) is the model. The test also needs a
GPU-execution arm where lane counts matter, since CI's lavapipe runs 8-lane fragment subgroups.
Every newly accepted module passes `spirv-val`. A refused-to-accepted flip with no witness for its
group does not land. An accepted-to-refused flip is a contradiction the census missed, and it is
explained before the step merges. Each step also gets a *Kena* title-screen burst (#5026).

**Mutation arms.** Each fact consumer moved to the analysis gets an arm that removes the fact's
source, such as the defining write or the narrowing that set the owner, and asserts that the read
now refuses. The refused twins of #5180, #5186 and #5193 stay red under the new analysis. Those are
a hard launch input, a hard accumulator, a relay chain ending at a hard input, a cross-lane pull of
a hard lane, a hard descriptor, a store or atomic at an arbitrary address, a cross-lane LDS form and
a read with a hard address lane. A further arm makes the transfer treat an untabulated row as
Defined and must turn the census red.

**Corpus-gate coverage (the plan's open question 4). Recommended: do not wait for general corpus
growth; rely on execution witnesses, with a targeted harvest of the programs that flip.**

- Step 8b is not limited by sidecars. The census compares facts, not SPIR-V, so it covers the 155
  verdict-only compute programs and the `incomplete` vertex replays as far as their consult points
  are reached.
- Steps 9 and 10 change verdicts. For every program that flips, the PR records whether it has a
  sidecar. Where a locally available title can produce one, a short run with the refused-shader
  dump harvests it before the PR merges. The rest are listed as verdict-only, and their group's
  execution witness must cover their shape.
- General corpus growth (more compute sidecars, complete vertex inputs) is welcome but does not
  gate these steps. It is unbounded. A witness per flip group is bounded, and it tests execution,
  which a byte comparison never does.

**While two models coexist** (from step 8b until step 10 merges), a change to validity semantics
lands in both models in the same PR. The fixture census is the ctest that reddens when it does not.
If the owner adds the #5202 inside/outside field to `ScalarSourceMarks` before step 10 (the plan's
open question 2), step 8b reproduces it, and the census treats it like any other fact.

## Consequences

**Easier.** A fact cannot exist on one route and be missing on the other. That class produced 13 of
the numbers from #5172 to #5188. The dispatcher stops being a second fact model. A narrowed write
can state "arbitrary inside, hard outside" (#5202). Loops stop refusing after one optimistic pass.
A new instruction's effects come from one table row instead of from many hand-written cases.
Steps 9 and 10 can delete the `advance_wave64_b64_masks` lambda (523 lines) and the three worklist
solvers in `emit_cfg_state_machine`. Most of `rdna2_spilled_mask_halves` (611 lines),
`rdna2_spill_slot_domain` (229), `rdna2_wave32_dispatch_words` (532), `rdna2_active_vector_words`
(489) and `rdna2_mask_half_alias` (224) can go too, as can the transfer half of
`rdna2_vector_word_marks` (1,061). That is about 3,100 lines in six files. Not every line in them is
a transfer, so read that as an upper bound.

**Risks, and what mitigates each.**

- **A wrong admission is silent.** The unsafe failure is a fact that admits wrongly: the shader is
  accepted, reads a value hardware would not have, and nothing reports it. No test sees such a fact
  in a program outside the fixtures and the corpus. The mitigations are each named above:
  - untabulated effects are Hard;
  - contradictions block;
  - a refinement must name a reason from a closed list;
  - each moved consumer has a mutation arm;
  - the policy twins stay red;
  - each flip group has an execution witness.

  The analysis answers only what it proves. Anything else refuses visibly. `CONFIDENCE: MED` that
  the closed reason list is enough to separate real refinements from wrong admissions. The census
  will show how often a refinement needs a new reason.
- **A long period with two models.** Steps 8b to 10 run both models side by side. The cost is a
  second implementation of every semantics change for that period. The fixture census is what stops
  the two from drifting apart unnoticed.
- **Compile time.** One fixpoint per compiled region. Recompiles are cached per program, so the
  cost does not reach the frame, but the census must report the time per program before step 9.
  Not measured yet.
- **File size.** New code lives under `src/gpu/recompiler/ir/analysis/`, which the existing
  `dir-file-cap` ratchet already holds to 1,000 lines a file (SIZE-1, ADR 0012 decision 5). The aim
  is 500 or fewer lines per file and 80 or fewer per function, one file per role:
  - the lattice and its join;
  - the scalar, VALU, memory and LDS transfers, one file each;
  - the edge transfers;
  - the solver;
  - the query;
  - the census.

  The 523-line Wave64 transfer lambda does not move whole. It is re-expressed per family. *Proposal,
  not measured:* the transfers will be the largest files. A family whose transfer passes 500 lines
  is split by sub-family, as the IR lowerings are.

## Alternatives considered

- **Extend `ScalarSourceMarks` in place** (#5202's suggested fix, the plan's open question 2). Adding
  an inside/outside split unblocks `indirect_pointer_descriptor_range` sooner. But the dispatcher
  still rebuilds four flags in `mark_relayed_words`, so the new field becomes one more fact that the
  pre-pass must re-derive and `load_state` must restore. That is the #5173 shape again. It is
  acceptable as a stopgap if the owner wants #5202 earlier. It is rejected as the architecture.
- **Keep two models but share the transfer function.** This removes write-model drift (the
  #5436/#5439 class). It leaves representation drift: the dispatcher's four-flag lattice, owners
  keyed by SPIR-V id that a reload empties, and the single-pass loop check. Those, not the write
  model, caused #5172-#5188. Its useful half is kept as step 8a.
- **Keep fixing bugs one at a time.** Thirteen issues and PRs in one run of consecutive numbers,
  each implemented in both models, is the evidence against.
- **Make lane masks a type and let type checking do the proofs** (the wording of #5146 step 5).
  Validity depends on the path and on the EXEC owner at the read, which a value's type cannot
  express without a dataflow fact. Typed masks help the mask-domain part (`Domain`). They do not
  replace the analysis.

## Approval

Only the project owner can accept this ADR. Acceptance unblocks plan step 8b and adds `GPU-6` as
policy. Step 8a (the effects-table extension) is #5146 step 4 and needs no acceptance. Step 7 is
covered by ADR 0012. Steps 9 and 10 additionally need a census with no open contradiction.
