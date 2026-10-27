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
- **Prove less: carry the uncertainty in the shader instead** (added 2026-10-23, not yet decided).
  See the next section.

### The prove-less alternative (2026-10-23)

This section records an alternative considered after the decision above was written. It may shrink
the decision, so it comes before acceptance.

**What prosper would change.** This alternative would retain prosper's wave-vote branches
(subgroup any/all over the per-lane predicate) and `select(exec, new, old)` VGPR writes. It proposes
two policy changes, subject to the proof obligations and measurements below:

1. **An unsupplied word could read as a value.** Initializing every register to zero would give
   unsupplied words a value. Prosper calls such a word a *placeholder* and refuses any read it cannot
   prove is confined to lanes that are never observed (`fresh_mask_sources_defined`,
   `vector_word_defined_in_exec`, the loop-carried fabricated check).
2. **An SGPR's mask representation could be tracked at runtime.** Each SGPR would carry an
   SSA-tracked "holds a mask" bit next to its value. A mask read would select between the per-lane
   bit and the bit decoded from the integer at the lane index. Constant propagation might remove
   some selects; their remaining cost needs measurement. Prosper proves the domain statically
   (`Domain`, the mask-half and spill-slot domain analyses) and refuses when the proof fails.

**What those two policies cost prosper today.**

- *Kena*, `prosper-app` on `main` at `4204ef669`, default launch, 130 s: four refused programs.
  Three are policy 1:
  - fragment `0x500aed0000`, `fabricated-fresh-mask-source`, which accounts for **all 329** dropped
    draws in the run;
  - compute `0x50071e0000`, `fabricated-fresh-mask-source`;
  - compute `0x500a540000`, loop-carried `v0` fabricated on the back edge.

  The two compute programs account for the 659 `shader-recompile` skipped dispatches. The fourth
  program (vertex `0x5009400000`, `mbcnt-cross-lane`) is unrelated. So the zero-dropped-shaders goal
  for this title is, today, a question about policy 1.
- **Corpus**, 4,000 entries (about 1,003 distinct programs), excluding refusals caused by the
  synthetic resource table. 1,755 entries are refused for a recompiler reason:
  - **597** `fabricated-fresh-mask-source` (519 fragment, 78 compute) and **45** loop-carried
    fabricated: policy 1;
  - **523** `structured-saved-mask-domain-missing-edge` (522 fragment): policy 2.

  That is **1,165 of 1,755 (66%)**. These are entries, not distinct programs. `CONFIDENCE: MED` that
  the synthetic-table filter separates every artefact.

**Why prosper refuses, and where that is right.** Admitting policy 1 blindly gives up exactly the
failure ADR 0038 exists to prevent: a placeholder that stands for a real input prosper failed to
supply (a launch VGPR missing from the launch inventory, a spill prosper did not track) would be
read as zero, silently. For a word the hardware does not load, the value really is arbitrary, and
reading zero is what #5174/#5180 already allow. So the honest version of policy 1 is the existing
rule applied everywhere: **a placeholder is Arbitrary when prosper can name why hardware holds no
meaningful value there, and Hard only when it is a real input prosper does not model.** The Measurements below classify a dated current cohort.

Policy 2 does not touch validity at all. It replaces proofs about representation with a runtime bit
and a select. Its cost is one extra boolean per SGPR in the SSA state and a select per mask read
that constant propagation cannot fold. It needs the integer form of a mask to exist, which a native
64-lane subgroup gives; the Wave64-on-narrower-host routes of ADR 0028 would need checking.

**What it would change in this ADR.**

- If policy 2 holds up, `WordFact::domain` and the mask-half and spill-slot domain analyses are not
  needed. That is the largest part of steps 9 and 10.
- If most policy-1 placeholders turn out to be Arbitrary, the admission checks that refuse them
  become census lines (a logged count) instead of refusals. The analysis still has a job, because it
  is where a placeholder's origin gets named. But its result decides less, and a wrong answer costs
  a diagnostic instead of a dropped shader.
- One model still beats two, so Decision 1 to 4 survive either way. What shrinks is the lattice and
  what consumes it.

**Owner direction (2026-10-26, #5497).** The honest version of policy 1 is approved:
only placeholders rooted entirely in explicitly unloaded words may become Arbitrary, with an
always-on diagnostic. Missing launch inputs, untracked spills and unknown launch profiles remain
Hard. Policy 2 is out of scope. The first PR adds origins and measurements without changing
admission; the admission PR follows only after the measurements merge. This direction does not
accept the proposed unified-analysis migration above.

#### Measurements (2026-10-27)

Diagnostics-only branch based on `0f6b8dd3ecd0`, including #5511, #5512, #5515 and #5526.
No game run. This is a conservative union of possible original-CFG roots,
not dynamic read counts or evidence that every path executes. Merge and loop nodes are traced
through; they are not origin categories. Supplied native Input publications and genuine resource
bindings terminate their respective dependency edges.

All 4,000 current corpus entries were surveyed: 3,567 use recorded-input replay and 433 fall
back to the stage probe. The effective policy-1 subset is **572 entries: 564 fresh-mask and eight
loop-carried refusals**. The earlier 642 above is a historical measurement, not a current filter.
No entry in this subset lacks an origin. Counts are entries, not distinct guest programs.
Of these, 505 use recorded replay and 67 use a stage probe. A raw stage probe lacks the producing
launch/resources: its origins describe the supplied probe configuration, not demonstrated original
hardware inputs. Of the 24 unloaded-only slices, 23 use replay and one uses a stage probe.

| Origin | Entries containing it | Distinct roots |
| --- | ---: | ---: |
| `unloaded` | 40 | 1373 |
| `launch-input` | 473 | 478 |
| `spill` | 0 | 0 |
| `unknown-launch` | 71 | 573 |
| `lost-definition` | 4 | 4 |
| `unmodelled-transfer` | 5 | 5 |
| `trace-budget` | 0 | 0 |
| `unreachable` | 0 | 0 |

**24 entries have only `unloaded` roots**; 548 contain another origin. Entry counts in the
table overlap when a read has mixed roots. A root is a distinct origin/word/lane/site tuple
within one entry. `lost-definition` means the compiler retained a fabricated mark but the slice
found only supplied values/constants, so it has no demonstrated unloaded root.
`unmodelled-transfer` names an uninventoried image address, implicit LDS/GDS dependency,
an unmatched loop edge or an endpoint outside the physical register file.
Trace exhaustion and missing instruction endpoints have separate names and occurred zero times.
None of these diagnostic gaps is relabelled `unloaded`.

An explicit partial launch word remains `launch-input`; words outside a partial or unknown
inventory remain `unknown-launch`. The compute request does not retain SCRATCH_EN, so the possible
scratch-offset SGPR following its known user/system prefix remains unknown as specified by
[RDNA2 ISA §3.12.2](https://docs.amd.com/api/khub/documents/Et~wpu9g~Ffl7d9q0QZ~Og/content).
Special-register initialization is also not inferred from that ordinary SGPR prefix.

All supplied game dump directories were checked: 27 binaries, 24 sidecars, no additional binaries.
Both builds replayed every sidecar; all nine accepted modules validate and match byte for byte.
The table covers every program, including unrelated refusals and missing sidecars. Origin
numbers count distinct roots in that program; a dash is not an unloaded classification.

| Game / program | Replay result | Root origins |
| --- | --- | --- |
| GTA `cs_4042f47600` | Accepted | — |
| GTA `cs_4042f49a00` | Unresolved operand, PC 16 | — |
| GTA `cs_405b540600` | Accepted | — |
| GTA `cs_405b552200` | VCC exit, header 53 | — |
| GTA `cs_405b56dc00` | No sidecar; not replayed | — |
| GTA `cs_405b573a00` | No sidecar; not replayed | — |
| GTA `cs_405b574e00` | No sidecar; not replayed | — |
| GTA `ps_405a9f3400` | Accepted | — |
| GTA `ps_405a9f6e00` | Accepted | — |
| GTA `ps_405a9fc400` | Accepted | — |
| GTA `ps_405ac25e00` | Fresh-mask, PC 324 | `unloaded` 36 |
| GTA `ps_405ac2ba00` | Fresh-mask, PC 324 | `unloaded` 36 |
| GTA `ps_405b1b1600` | MIMG address, PC 350 | — |
| GTA `ps_405b1da200` | MIMG address, PC 403 | — |
| GTA `ps_405b1f5a00` | MIMG address, PC 427 | — |
| GTA `ps_405b1f7600` | MIMG address, PC 426 | — |
| GTA `ps_405b20f400` | Unresolved operand, PC 146 | — |
| GTA `ps_405b23ea00` | Accepted | — |
| GTA `ps_405b243a00` | Accepted | — |
| GTA `ps_405b253400` | Fresh-mask, PC 154 | `launch-input` 1, `unknown-launch` 17, `unmodelled-transfer` 1 |
| GTA `ps_405b257e00` | Fresh-mask, PC 154 | `launch-input` 1, `unknown-launch` 17, `unmodelled-transfer` 1 |
| GTA `ps_405b26ca00` | MIMG address, PC 372 | — |
| GTA `ps_405b3ddc00` | Fresh-mask, PC 1184 | `launch-input` 3, `unknown-launch` 53 |
| Kena `cs_50071e0000` | VCC exit, header 166 | — |
| Kena `cs_500a540000` | Accepted | — |
| Kena `ps_500aed0000` | Accepted | — |
| Kena `vs_5009400000` | Vertex MBCNT, PC 378 | — |

For GTA, two policy-1 programs have only unloaded roots; three have mixed roots. Kena has no
policy-1 refusal on this base: two programs accept and one compute program reaches the separate
VCC-exit refusal. Across the five game policy-1 refusals, two contain unloaded roots, three launch
inputs, three unknown launch roots and two unmodelled transfers (overlapping groups); none contains
a spill root.

The mixed GTA inputs remain Hard: [#5518](https://github.com/mattias800/prosper/issues/5518)
tracks partial ANCILLARY v2, including an actual bit-8 extraction outside the supplied layer bits;
[#5519](https://github.com/mattias800/prosper/issues/5519) tracks raw perspective-center v0/v1 in
`ps_405b3ddc00`. These follow-ups do not establish which runtime path executes. The separate
WQM/DPP proof loss, MIMG-address/unresolved-operand refusals, VCC exits and vertex MBCNT are
not made admissible by this census.

The corpus comparison reports **`changes: none`**. All 1,538 comparable accepted-module hashes
match; five previously accepted entries have no comparable module hash. Both surveys also have
388 incomplete answers and one unchanged inspector error; these are coverage limits. A valid-module
generator mutation is detected as one `output-changed` entry (exit 1), and that mutated module passes
`spirv-val`. Diagnostic details do not participate in verdict tags or module identity.
Replay compile-cases retain the primary refusal identity under their existing 256-byte bound;
complete terminal diagnostics retain the separate origin envelope.

Reproduce the measurement with two Release builds, an empty GAME_DUMP and at most four workers:

```sh
python3 prosper/tools/shader_inspect/recompile_corpus.py compare <CORPUS> \
  --base <MAIN_BUILD>/shader_inspect --candidate <BUILD>/shader_inspect \
  --out-dir <RESULTS> --jobs 4
python3 prosper/tools/shader_inspect/placeholder_origin_census.py \
  <RESULTS>/candidate.json --out <ORIGIN_REPORT>
shader_inspect <program.bin> --inputs <program.inputs> # add --stage compute for CS
```

#### Admission measurements (2026-10-27)

The admission candidate first preserves every successful strict compilation unchanged. A refusal
may retry only when an actual read has a nonempty root set consisting entirely of `unloaded`.
The retry supplies no new values, physical mask words or runtime validity facts. Successful
scalar loads terminate a dependency slice only after their actual complete, non-fabricated
publication; an unproved memory result or unsafe address-read obligation stays Hard. These name
`unproved-memory-result` and `unproved-strong-read`, respectively. The legacy READFIRST scalar
value, which the per-invocation lowering does not broadcast, names
`unmodelled-scalar-publication` and also stays Hard. The admission retry guards the actual
legacy lowering, including straight-line scalar relays and publications independent of the
admitted read. Existing agreeing-source and local waterfall proofs remain applicable, and real
dispatcher broadcasts do not pass through that lane-local lowering.
Runtime-relative bank readers check each potentially selected word independently; a bounded
index supplies no register value, and unloaded alternatives cannot hide a lost provided word.

The same 4,000-entry corpus compares 3,567 recorded-input replays and 433 stage probes:

- No corpus entry becomes accepted. The compute entry that passed the fresh CMPX reads at
  PCs 34 and 43 in an earlier candidate now reaches a Hard implicit old-destination read at
  pc46, with unproved memory-result roots. Passing an unloaded predicate cannot waive this
  separate obligation.
- No accepted output changes and no accepted program becomes refused. The comparison hashes
  1,538 existing outputs; supplemental exports through the identical public table-less compute
  entry point hash the other five, also unchanged. All ten supplemental modules validate.
- Eleven entries remain refused with a changed reason: eight stop earlier and three reach later
  Hard reads. These retain lost definitions, missing scalar publications, memory-result or
  unresolved-operand obligations. One previously refused fragment replay becomes incomplete
  because its old semantic-read trace cannot supply the new compilation sequence. No missing
  choice is guessed.
- The terminal policy-1 cohort has 571 entries. Origin groups overlap; none has only unloaded
  roots and none is unclassified.

| Origin | Terminal entries | Terminal roots | Entries including auxiliary causes | Roots including auxiliary causes |
| --- | ---: | ---: | ---: | ---: |
| unloaded (mixed with Hard roots) | 36 | 1,358 | 139 | 3,684 |
| launch-input | 473 | 478 | 476 | 482 |
| unknown-launch | 72 | 585 | 649 | 2,918 |
| lost-definition | 12 | 12 | 31 | 31 |
| unmodelled-transfer | 5 | 5 | 8 | 8 |
| unproved-memory-result | 18 | 357 | 43 | 1,113 |
| unproved-strong-read | 1 | 1 | 1 | 1 |
| unmodelled-scalar-publication | 3 | 3 | 3 | 3 |
| spill / trace-budget / unreachable | 0 | 0 | 0 | 0 |

Some Wave64 dispatcher summaries omit a lowering's named cause even when its auxiliary
record retains it (#5529). Retaining the full `PROSPER_DBG=1` diagnostic stream from the same
invocations identifies **958 refused entries with a named policy record**, including the 571
terminal entries and 387 additional auxiliary entries (330 recorded-input replays, 57 stage
probes). Auxiliary roots include failed alternative lowering paths; they are not all the final
refusal's dependencies. The last two columns union each refused entry's terminal and auxiliary
roots, without counting duplicate roots within an entry. Stage diagnostics reuse the corpus
runner's representative for identical stage/code hashes. Independent token accounting agrees,
and substituting an origin in a real diagnostic changes the aggregate. These wider counts are
separate from the terminal-only census command above.

The 27 dumped GTA/Kena programs retain the preceding measurement's nine accepted, fifteen refused
and three missing-sidecar results. All nine accepted modules validate and retain their hashes.
The remaining five fresh-mask GTA fragments report:

| Program | Hard roots retained alongside any unloaded roots |
| --- | --- |
| `ps_405ac25e00`, `ps_405ac2ba00` | unproved memory result at MIMG pc150 |
| `ps_405b253400`, `ps_405b257e00` | launch-input `v2`, unknown launch, unmodelled transfer, unproved memory result |
| `ps_405b3ddc00` | launch-input `v0/v1/v2`, unknown launch, unproved memory results |

The already tracked launch-input bugs are #5518 (`v2`) and #5519 (`v0/v1`); these remain Hard.
The other refused game programs retain the earlier table's separate MIMG, VCC-exit or MBCNT
reasons. This is an offline census, with no game run.

The three MOUSE UI captures add a further replay limitation: `ps_403f306100` and
`ps_4043ba9200` encounter unloaded-only predicates on the strict attempt, but their old semantic
read traces cannot replay the new strict-then-retry compilation sequence and now report incomplete.
This does not establish that their retries reached the old predicate PCs. `ps_4043ca3f00` still refuses with launch-input
`v0/v1` alongside unloaded roots. Fresh live captures and visible portrait verification remain
necessary; these are not claimed as accepted modules.

Twenty-four synthetic interpreter witnesses fail against the merged diagnostic-only compiler and
pass with admission. They cover integer/float predicates, CMPX, saved SDWA predicates, narrowed
writes, vector/scalar loop carries, scalar and READLANE relays, implicit scalar compares, ordinary
data reads beside an unloaded predicate, real scalar-buffer data, a provided spill containing
an unloaded value, and wide-word/implicit-SCC selectors of EXEC. Nineteen actual exact64 GPU
dispatches exercise both wave halves, CMPX's active lane count, the relay/data/spill paths and
both supplied condition values. Five existing selector/loop fixtures also execute both
scalar branch inputs; real missing spills remain refused. An agreeing READFIRST source retains its
established proof. All 38 exported synthetic modules validate. Three loop originals whose computed
scalar values are discarded retain `lost-definition`; the delayed vector-body original remains
Hard at its legacy READFIRST publication. Their genuine and complete-replacement controls retain
the existing successful output.

The final source qualification at `0f6e16f7e` builds the complete app/test configuration and
registers 7,144 CTests: 7,131 pass, ten skip, one is disabled, and only the existing
`indirect_pointer_descriptor_range` and `rdna2_to_spirv_exec` fail (exit 8). All 7,062 baseline
registrations remain with two documented policy/diagnostic renames. The 208 focused cases pass.
The high-word/SCC corrections leave the corpus and game census above unchanged; no game run.

Mutation controls deny unloaded admission, admit Hard roots, lose scalar-load publication, keep
a stale retry refusal, or disconnect the observer; each is detected. The actual #4943 missing-spill
twin also detects the relaxed-Hard mutation. Ordinary lane-harness instruction tapes match main
byte-for-byte, with a deliberate instruction mutation detected. Old Hard-input controls now use
an unknown possible scratch-offset launch word instead of an explicitly unloaded SGPR.

Four existing refusal assertions now name the actual earlier producer or missing ordinary input.
The loop fixtures also keep genuine Hard-input back-edge twins. The two M0 fixtures check the
actual local-index packet certificate independently of their earlier public input refusal;
replacing the real deferred M0 retirement guard with `true` makes both tests fail. These changes
preserve the guards' coverage rather than treating any refusal as sufficient.

`arbitrary-placeholder-admitted` is a default-on correctness alarm with a program-address run
summary. Window counts are successful retry compilations and static read sites, not draws or
executed lane reads. Failed retries publish their actual refusal and no admission count.

#### Ruled out during admission qualification (2026-10-27)

- **Checking a wide operand's low word covers its SCC/EXEC dependency.** BFE read its high
  word directly, substituting zero for unknown launch `s13`. A following CSELECT used SCC to
  select EXEC without reading the BFE destination. The missing-high regression fails before
  the guards; supplied high bits 0/1 produce 11/7 in the interpreter and on the exact64 GPU,
  and an unloaded pair retains 11. Both consumed words require the actual input boundary.
  An independent entry-SCC fixture also exposed stored `false` being mistaken for a launch
  input. Retry-only condition tracking distinguishes that unknown input from real SCC writers;
  unloaded-only computed conditions retain their roots at the actual read PC. #5528.

- **Register-map membership establishes that an actual source value exists.** MOVRELS reserves
  its destination before reading the bank. When that destination aliases a lost provided
  candidate, the temporary entry has ID 0 and hides the absence from a membership-only guard.
  The interpreter and exact64 GPU again store 64 zeroes instead of `[13,7,...]`. Treat an empty
  ID as missing at the shared VGPR input boundary. Provided and genuinely unloaded alias twins
  retain their values; a real zero-valued constant has a nonzero SPIR-V ID. #5528.

- **An unloaded alternative authenticates an entire relative-register bank.** With runtime
  `M0 = supplied_s4 & 3`, `v_movrels v1,v5` can select `v6` erased by a preceding WRITELANE.
  Genuine `v5/v6` producers terminate their slices, while unloaded `v7/v8` alternatives leave
  an unloaded-only root union at the later store. That union did not retain the lost selected
  computation. The interpreter and exact64 GPU store 64 zeroes instead of `[13,7,...]`.
  Check every potentially selected candidate at the actual bank reader, using the existing
  bounded-index query and each word's independent provenance. Provided and genuinely unloaded
  bank controls retain their values. [#5528](https://github.com/mattias800/prosper/pull/5528).

- **Explicit operand guards cover implicit destination reads.** A partial EXEC write after
  WRITELANE preserves the ordinary old destination. Without an actual old-word guard, the
  interpreter and exact64 GPU observe `[22,0,...]` instead of `[22,7,...]`. The existing
  `vector_write_depends_on_old_word` predicate identifies those reads, including accumulation
  and subword preservation; no new transfer facts are added. A complete overwrite must still
  compile, and named lane-slot updates must not be treated as ordinary old-VGPR reads. Both
  false-positive mutations are detected. The underlying producer loss also reproduces on
  strict merged main without an unloaded predicate; it remains separately tracked as
  [#5530](https://github.com/mattias800/prosper/issues/5530). This policy preserves strict successes.

- **The ordinary LDS callback protects every DS input read.** Compute ADDTID returns before
  that callback. After `v6=7; writelane v6,s0,0` with supplied `s0=13`, its direct store silently
  substituted zero for the erased ordinary VGPR. A genuine ADDTID reload hid the lost source.
  With an independent unloaded equality predicate, the interpreter stored zero in all 64 lanes,
  rather than lane0=13 and the other lanes=7; the module still passed `spirv-val`. Guard the actual
  early store. The audited early GDS/shuffle readers and implicit M0 reads use the same boundary.
  Missing computations retain `lost-definition` even when their original operands were genuine.
  The existing tracer already names that loss; reverting a proposed extra empty-root rule does
  not change this regression. That unnecessary rule was removed.

- **An unloaded predicate makes unrelated ordinary reads safe.** Synthetic retries still read an
  unknown possible scratch-offset `s12`, or export/store an unsupplied fragment launch `v0`.
  Bypassing the actual operand and direct-memory/export guards admits these real missing inputs.
  Both instruction orders and an implicit SOPK operand are covered. The provided unloaded twins
  retain their existing values; the unrelated Hard roots still veto admission.
- **A reaching computed value can be replaced by its unloaded input's storage zero.** A conditional
  `s40 = 7` is discarded at a join. Without retaining that lost definition, the retry traces the
  other path to an unloaded word and admits zero on both paths. The interpreter observes 22 instead
  of the required 11 on the written path. This remains `lost-definition`, independently of the
  missing path's Arbitrary value. Refs [#4560](https://github.com/mattias800/prosper/issues/4560).
- **Loop lane-bit checks cover ordinary carried spill reads.** A loop can store a previous scalar
  and then READLANE an untracked slot, without ever using that scalar as a predicate. The ordinary
  carried value hid the missing spill. Guarding the actual READLANE producer retains `spill` on
  both branch inputs. A slot actually written from an unloaded word remains admitted and executes
  in the interpreter and on the GPU.

- **The final public reason identifies every forced-dispatcher cause.** The delayed vector-body
  fixture's default route reports `exec-loop-widens`; its forced Wave64 dispatcher separately
  records the missing READFIRST publication at pc22. Its auxiliary record includes the origin,
  but the final generic unresolved-operand record overwrites the named cause. Tests inspect
  the full capture from the explicit route. Tracked separately in [#5529](https://github.com/mattias800/prosper/issues/5529).

- **An unloaded address alone makes every memory result Arbitrary.** The unchanged #5193
  atomic/store-return and cross-lane LDS refusal twins exposed a missing dependency in the
  diagnostic slice. The result's separate memory contract must be retained; an unproved result
  now names `unproved-memory-result`. A loop's existing strong address-read obligation
  likewise names `unproved-strong-read`. Neither is silently reduced to its unloaded address
  or back-edge registers. Refs [#5497](https://github.com/mattias800/prosper/issues/5497).
- **A never-loaded ordinary SGPR is a genuine Hard input fixture.** Several older fixtures used
  s30, s40 or s60 this way. Under the owner-approved policy those words are Arbitrary. Hard
  controls must instead name an unknown possible scratch-offset launch word, a genuinely
  unsupplied launch input or an untracked spill. Consumer and relative-branch positions are
  retained when correcting these fixtures. Refs [#5497](https://github.com/mattias800/prosper/issues/5497).

- **Guarding only loop-carried READFIRST dependencies covers scalar publication.** A straight-line
  `v6 = unloaded_zero + lane + 7; s20 = readfirstlane(v6); VCC = (s20 == v6)` also
  exposes the missing broadcast: the interpreter and exact64 GPU must select 11 only in lane0
  and 22 in the other 63 lanes. A lane-local copy instead selects 11 everywhere. The safeguard
  belongs at the actual legacy producer, even before or after an independent admitted predicate.
  An agreeing constant source retains its existing proof. Refs [#5497](https://github.com/mattias800/prosper/issues/5497).

- **Removing the loop's READFIRST guard only admits an Arbitrary value.** A synthetic loop
  sets `v6 = lane + unloaded_zero + 7`, reads it into `s20` on the next trip, and stores that
  scalar. With the guard removed, the interpreter and an actual exact64 GPU both store
  `7 + lane` instead of the hardware's whole-wave scalar `7`. The legacy lowering returns
  each lane's value instead of broadcasting the first active lane. This is a real missing
  scalar publication, named `unmodelled-scalar-publication`, and stays Hard. Refs [#3596](https://github.com/mattias800/prosper/issues/3596), [#5497](https://github.com/mattias800/prosper/issues/5497).

**Measurement before acceptance.**

1. **Kena's three programs.** For each fabricated word, name its origin: never written, a launch
   input missing from `rdna2_launch_vgprs`, or an untracked spill. A missing input is a
   launch-inventory bug, fixed on its own. A word hardware does not load is Arbitrary under the
   existing #5180 rule, and admitting it needs an execution witness.
2. **Corpus census of policy 1.** The same classification over the 642 entries, reported by
   origin.
3. **Policy-2 prototype** behind a switch, on the 523 domain-missing entries: verdict flips,
   `spirv-val`, and one execution witness per flip shape.

The three results decide whether this ADR is accepted as written, accepted without `Domain`, or
rewritten around a smaller analysis.

## Approval

Only the project owner can accept this ADR. Acceptance unblocks plan step 8b and adds `GPU-6` as
policy. Step 8a (the effects-table extension) is #5146 step 4 and needs no acceptance. Step 7 is
covered by ADR 0012. Steps 9 and 10 additionally need a census with no open contradiction.

**Held (2026-10-23):** acceptance waits for the three measurements in *The prove-less
alternative*, because they decide how large the lattice in Decision 1 needs to be.
