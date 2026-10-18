---
kind: plan
status: current
---

# Control flow and definedness on the recompiler IR

ADR 0012 moves the RDNA2 recompiler onto an SSA IR one instruction family at a time. The families
are mostly done. What is left is control flow (`src/gpu/recompiler/rdna2_emit_cfg.cpp`) and the
definedness/proof machinery that is threaded through it. This document measures both, proposes a
target, and orders the migration into small steps that each keep the corpus gate (ADR 0006)
byte-identical or explain every change.

It is a proposal. Steps 1-7 fit inside ADR 0012 decision 1, which already names "structurisation"
as an IR pass. Step 8 changes who owns validity facts and should get its own ADR first (see
[Open questions](#open-questions)).

Measured on `origin/main` at `6cb4a9774` (2026-10-18). Line numbers refer to that commit.

## Scope

In scope:

- The control-flow emitters: structured if/else and loops, counted loops, and the
  whole-program CFG dispatcher (`emit_cfg_state_machine`).
- The facts those emitters carry across block edges: vector-word marks, relays,
  `ActiveVectorWords`, `SpilledMaskHalves`, spill-slot domains, mask-half aliases, Wave64/Wave32
  mask-domain MUST sets, entry-M0 tokens, loop blanket roots and EXEC-subset facts.

Out of scope:

- Instruction families that are already on the IR (`ir/HOW_TO_ADD_A_FAMILY.md`).
- The NGG subgroup shell, the fragment packet executors and the raster collectors, except where
  they call `emit_cfg_state_machine`.
- Resource provenance (SMEM pointer, descriptor and pc-relative tables), except where the
  dispatcher drops it at a reload.

Layer: `gpu/recompiler`, which is a pure function of its request (spec GPU-1, ADR 0006).

## Current state

### `rdna2_emit_cfg.cpp` by responsibility

7,362 lines in total, 6,825 of them in two functions. `emit_cfg_state_machine` is one function of
5,079 lines. Its helpers are lambdas that capture about a hundred locals, so `tools/refactor/` cannot
move them mechanically.

| Lines | Count | Responsibility |
| --- | ---: | --- |
| 1-95 | 95 | includes, file comment |
| 96-498 | 403 | whole-CFG use proofs: S_LOAD_DWORDX2 descriptor fragments (369), unpersisted B32 mask lifetimes (34) |
| 499-517 | 19 | `entry_m0_save_in_range` |
| **518-5596** | **5,079** | **`emit_cfg_state_machine`, the dispatcher** |
| 518-642 | 125 | admission and refusals (stage, packet masks, NGG shell, DPP quad-OR service) |
| 643-929 | 287 | per-instruction candidate recognizers: mask-domain pairs, mask compares, DPP row/quad shapes, WQM, FFBH |
| 930-1184 | 255 | block splitting (branch targets and every cross-lane service), service event tables, LDS scratch layout |
| 1185-1485 | 301 | persisted register set, entry-M0 MAY set, descriptor MAY-writes (worklist solver), Wave32 dispatch words |
| 1486-2321 | 836 | Wave64 MUST/MAY analysis over the blocks (worklist solver at 2197). Its transfer function `advance_wave64_b64_masks` (1672-~2190) is about 520 lines and re-states instruction effects |
| 2322-2572 | 251 | fragment saved-mask compares, the native exact-wave case, lane-slot invalidation (worklist solver at 2516) |
| 2573-2852 | 280 | `PROSPER_CFG_TRIP_BOUND` diagnostic |
| 2853-2965 | 113 | recycled VCC, initial EXEC, Function variables |
| 2966-3222 | 257 | `load_state`: rebuild a `RegState` at every case entry from the Function variables and 8+ per-block fact vectors |
| 3223-3287 | 65 | `save_state` |
| 3288-3387 | 100 | dispatcher loop header and `OpSwitch` |
| 3388-4736 | 1,349 | per-case emission: block bodies through `emit_alu`, terminators, successor selection, publishing service operands |
| 4737-5476 | 740 | synchronized common phases (DS_SWIZZLE, DS_BPERMUTE, votes, DPP rows, LDS atomics, MBCNT, scalar wave services, DS_APPEND) |
| 5477-5596 | 120 | vote and liveness reduction, loop exit, exporting the final state |
| 5597-5602 | 6 | comment |
| **5603-7348** | **1,746** | **`emit_body`, route selection plus the structured emitters** |
| 5603-5698 | 96 | once-per-program pre-analyses and pc-relative tables |
| 5699-5854 | 156 | barrier phases and the early whole-program dispatcher routes (portable wave services) |
| 5855-5999 | 145 | straight-line walker (`emit_range`) and fragment mask seeds |
| 6000-6381 | 382 | counted-loop route, including its forward-if prelude and postlude |
| 6382-6500 | 119 | divergent-loop detection and compute guards |
| 6501-6811 | 311 | route selection: complex-CFG tests, transactional dispatcher attempts, `[structured-wave-reject]` diagnostics |
| 6812-6865 | 54 | structured setup |
| 6866-7093 | 228 | `emit_divloop` (EXEC/VCC/SCC-exit loops) |
| 7094-7348 | 255 | `emit_structured` (forward if/else, merges, joins) |
| 7349-7362 | 14 | tail |

The dispatcher splits a block at every cross-lane service as well as at every branch target. The
per-case code tracks 21 kinds of block-ending event (MBCNT, DS_APPEND, synchronized LDS store, LDS
fmin/fmax, swizzle, bpermute, five DPP row/quad shapes, mask FFBH, mask reduction, packet WQM,
readlane, readfirstlane and five mask-compare/vote forms; declarations at 3427-3447). So the
dispatcher is a wave-synchronization device as much as a CFG device, and any IR replacement has to
model those services, not only the branches.

Code outside this file that belongs to the same job:

| File | Lines | Role |
| --- | ---: | --- |
| `rdna2_cfg_support.hpp` | 3,009 | pattern detectors: `detect_counted_loop` (471-635), `detect_divergent_loops` (636-1232), `detect_forward_ifs` (1233-1643), workgroup-uniform branch proofs (1914-2296), the linear `must_fact_at` helper (1859) |
| `rdna2_to_spirv_internal.hpp` 3420-4270 | ~850 | merge seeds, merge-edge marks, `LoopBlanketCheck`, loop-exit marks, scalar-write records |
| `rdna2_to_spirv_internal.hpp` 3032-3252 | 221 | `RegState`: about 70 fields, most of them facts rather than values |
| `rdna2_to_spirv_internal.hpp` 438-3015 | 2,577 | `SpirvCompute`: about 207 member lines, 34 of them per-pc proof sets or analysis results |

`rdna2_structured_saved_masks`, `rdna2_lane_slot_carry`, `rdna2_loop_vcc_carry`,
`rdna2_counted_loop_guard` and `rdna2_exec_skip_region` are the structured emitter's join and loop
helpers. #5146 counts 37 separate validity analyses in the recompiler. There is no shared CFG
type: 19 recompiler files refer to a `successors` relation, the dispatcher builds its block list and
successors locally (930-1485), and the three detectors pattern-match the instruction vector
independently.

### Proof facts: where each is produced, consumed and re-derived

Two separate models carry the same facts.

- **Structured path.** Facts live in `RegState` and are updated as a side effect of emission:
  `emit_alu` calls `record_vector_word_marks`, `record_exec_mask_subset` and the rest, and the
  structured emitter joins them at merges (`join_vector_word_marks`, `merge_edge_marks`,
  `join_structured_saved_masks`, `join_lane_slots`). Facts are stated relative to **SPIR-V ids**:
  `ScalarSourceMarks::outside_exec` and `RegState::exec_subset_of` hold the id of an EXEC value.
- **Dispatcher.** `load_state` builds a fresh `RegState` at each case and does not copy
  `vreg_source_marks`. Facts come back from a static pre-pass over the dispatcher's blocks, with its
  own transfer functions (`advance_wave64_b64_masks`, `analyze_wave32_dispatch_words`,
  `analyze_wave64_mask_half_aliases`, the `SpilledMaskHalves` and `SpillSlotDomains` transfers).
  This model is **coarser**: `mark_relayed_words` (`rdna2_spilled_mask_halves.cpp:339`) re-creates
  only `placeholder`, `hard`, `arbitrary` and `memory`. Blanket roots and scoped roots are not
  carried, and `outside_exec` is re-anchored to the newly loaded EXEC only for the words
  `ActiveVectorWords` proves (`restore_active_vector_words`, `rdna2_active_vector_words.cpp:123`).

| Fact | Produced | Consumed | Re-derived at a dispatcher reload? |
| --- | --- | --- | --- |
| Vector-word marks (`ScalarSourceMarks`: placeholder, memory, hard, arbitrary, owner `outside_exec`, blanket and scoped roots) | `seed_vector_word_definedness`, `record_vector_word_marks` from `emit_alu`; joined by `join_vector_word_marks` and `confine_execz_skip_marks` | `fresh_mask_sources_defined`, `vector_word_defined_in_exec`, the loop checks, MIMG source proofs | **Partly.** Dropped by `load_state`; `mark_relayed_words` restores four flags; the owner is restored only for active words |
| Relayed words and vectors (`SpilledMaskHalves::relayed_*`, `nonarbitrary_vectors`, `memory_*`) | the dispatcher pre-pass (`rdna2_spilled_mask_halves.cpp`, `rdna2_wave32_dispatch_words.cpp`) | `load_state` through `mark_relayed_words` and `restore_wave32_dispatch_words` | Yes; this model exists only to re-derive |
| `ActiveVectorWords` (words genuine in every lane of the current EXEC) | `advance_active_vector_words` in the pre-pass, met at joins | `active_vector_sources_defined`; `restore_active_vector_words` at each case | Yes; it is the dispatcher's substitute for an owner mark (#5173) |
| Wave64 B64 mask domain, ambiguous pairs, scalar words, M0 tokens, readlane words, SCC validity | `advance_wave64_b64_masks` (1672) | `load_state` filters (which Bool and scalar variables to reload), compare and branch admission sets | Yes |
| Wave32 B32 mask lifetimes | `analyze_wave32_dispatch_words`; `has_unpersisted_b32_mask_lifetime` (472) | `load_state`; region refusals | Yes |
| Mask-half aliases (LO/HI of EXEC in a slot or SGPR) | `analyze_wave64_mask_half_aliases` | `load_state`, `v_readlane` reloads | Yes |
| Spill-slot domains (data or mask per `v_writelane` slot) | `rdna2_spill_slot_domain` transfer in the pre-pass | `load_state`'s `slot_holds` (#4600) | Yes |
| EXEC subset and complement (`exec_subset_of`, `exec_complement_in`) | `record_exec_mask_subset`, `exec_masked_predicate` (`rdna2_loop_exec_subset.cpp`) | `exec_within`, owner meets at joins | No: keyed by SPIR-V id, so a reload starts empty |
| Loop blanket roots | `mark_loop_carried`, `mark_vector_loop_carried` at a loop header | `LoopBlanketCheck::holds` at the back edge (`rdna2_to_spirv_internal.hpp:3610`), which refuses when an assumption fails | Not carried |
| Entry-M0 token | `s_mov_b32 sN, m0`; MAY set at 1185, MUST in the Wave64 pass | `load_state` re-tokenizes the reachable saves (#3133, #4706) | Yes |
| Descriptor and SMEM provenance (`sreg_srt`, pc-relative tables, raw-wide chains) | once-per-program seeds | memory lowering | `sreg_srt` and `lds_addtid` are deliberately not reconstructed (5581-5583); the per-program sets are copied |

The worklist solvers inside `emit_cfg_state_machine` are at 1437 (entry-M0), 2197 (Wave64 MUST) and
2516 (lane-slot invalidation); the others are in the helper files above. The structured path has no
pre-pass. Its loop facts are a **single-iteration optimistic check**: assume at the header, emit the
body, check at the back edge, refuse on failure. A dataflow fixpoint would instead iterate.

The parked failures follow from this split. #5202's second gap is a narrowed complete write of an
arbitrary source over a hard old word. `ScalarSourceMarks` cannot state "arbitrary inside the owner,
hard outside", so the owner is lost and a later read is refused. #4949 and #5172-#5188 are the same
class: a fact that one model carries and the other loses or re-derives differently.

### Corpus measurements

Corpus: the local all-stage corpus (`recompile_corpus.py harvest`), 4,000 files and **1,003
distinct programs** (stage plus code hash). It is harvested from `refused_shaders_*` dumps, so it
is **biased toward programs that some build refused**; it is not a census of shipping shaders.

Method: `shader_inspect` built from `6cb4a9774` with a **local, uncommitted** instrument (four
`stderr` lines behind an environment variable, at `emit_body`'s route decision, the counted-loop
route, the dispatcher's block count and the dispatcher's result). Each distinct program ran once,
on its recorded live inputs where a sidecar exists (`--inputs`) and otherwise with `--stage`
(`--synthetic-table` for graphics), with `PROSPER_DBG=1`. Refusal reasons come from an unmodified
`recompile_corpus.py survey` run over the same build.

Routes, by distinct program. "Accepted" is the program's overall verdict; a program can try the
dispatcher, fail, and still be accepted through barrier phases.

| CFG shape | Route taken | Compute | Fragment | Vertex |
| --- | --- | ---: | ---: | ---: |
| straight-line (0 branches) | linear | 37 (5 accepted) | 22 (8) | 60 (0) |
| branches, none structured (linearized as safe, or refused at the branch) | linear | 43 (6) | 51 (30) | 13 (0) |
| | dispatcher, emitted | 15 (15) | - | - |
| | dispatcher, failed | 15 (2) | - | - |
| acyclic, forward ifs only | structured | 20 (8) | **176 (88)** | 70 (0) |
| | dispatcher, emitted | 48 (46) | - | - |
| | dispatcher, failed | 35 (1) | - | - |
| loops, possibly with ifs | structured | 20 (2) | 37 (6) | 9 (0) |
| | dispatcher, emitted | 9 (7) | - | - |
| | dispatcher, failed | 44 (13) | - | - |
| counted loop | structured | 14 (0) | 6 (4) | 19 (0) |
| detectors refused the CFG (`cf_rejected`) | dispatcher, emitted | 54 (51) | - | - |
| | dispatcher, failed | 99 (0) | 3 (0) | 3 (0) |
| | linear | 10 (0) | - | - |
| early whole-program dispatcher (portable wave services) | dispatcher, emitted | 3 (1) | - | 1 (1) |
| | dispatcher, failed | 44 (0) | - | - |
| never reached CFG emission | - | 8 | 10 | 5 |

Vertex replays end `incomplete` (no vertex verdict on these inputs) far more often than `accepted`,
so the vertex accepted counts say nothing about the CFG route.

Summary:

- **Structured route:** 607 distinct programs (straight-line, linearized, structured and counted).
  Fragment is entirely structured or linear. Its most common shapes are one forward if spanning 3
  branch instructions (57 programs), one linearized branch (43), one forward if over 2 branch
  instructions (20) and two forward ifs over 2 (13).
- **Dispatcher route:** 366 compute and 7 graphics programs (373) try it. 130 emit a dispatcher (129
  compute, 1 vertex). The dispatcher has **2 to 248 blocks** per compiled region (median 15, p90
  125, n=305 regions; one program can compile several regions or retries).
- **The dispatcher fails more often than it succeeds**: 243 of the 373 programs that try it never
  emit one. First terminal refusal for those 243:

| Refusal | Programs |
| --- | ---: |
| `unresolved-operand` inside a dispatcher case (139) or before the case (13) | 152 |
| `wave64-ambiguous-mask-read` (51) / `wave32-ambiguous-mask-read` (11) | 62 |
| `fabricated-fresh-mask-source` | 15 |
| loop-carried word fabricated on the back edge | 4 |
| `poisoned-scc` | 4 |
| other (5 distinct reasons) | 6 |

  100 of the 152 `unresolved-operand` refusals happen on recorded live inputs. They split evenly
  between memory instructions (SMEM, MUBUF, MTBUF, MIMG, FLAT: 50) and ALU instructions (50).
  **Speculation, not measured:** some of these are facts or provenance that `load_state` drops at
  a case boundary and that the same program would keep on a structured route. Step 8's shadow
  census answers this program by program.

- **Over the whole corpus** (every file, by distinct program), the largest refusals are
  `unresolved-operand` (377 programs, 23 titles), `unsupported-in-stage` (209 programs, 19 titles),
  `wave64-ambiguous-mask-read` (32 programs, 7 titles) and `structured-saved-mask-domain-missing-edge`
  (23 programs, 5 titles; 524 files, because these programs recur with many sidecars). The last is
  the structured if/else join refusing when one edge lacks a mask view
  (`rdna2_structured_saved_masks`), so it belongs to the definedness machinery too.

## Decision

Proposed:

1. Control flow becomes IR: one CFG built once from the decoded program, and one region tree
   derived from it. Every emitter and analysis reads these instead of building its own block list,
   successor relation or pattern match. This is the "structurisation pass" ADR 0012 names.
2. EXEC becomes an IR value with static identity, so an owner fact means the same thing in the
   structured emitter, in the dispatcher and at a reload.
3. Definedness and the proofs become **one dataflow analysis over the IR**, solved once before
   emission, with one lattice and one transfer function driven by `rdna2_instruction_effects`.
   Emitters query it and stop threading facts through `RegState`. This changes who owns validity
   facts in the recompiler and needs its own ADR before step 8.
4. During migration, SPIR-V stays byte-identical. Where the unified analysis proves more than the
   dispatcher's coarser model, the resulting verdict changes are explained and execution-tested as
   ADR 0012 decision 2 requires. They are not silent.

## Target design

### Data structures

Everything below is plain data with no `SpirvCompute` calls, so building it cannot change emitted
bytes.

```cpp
namespace prosper::gpu::ir::cfg {

enum class Terminator : uint8_t {
    Fallthrough, Branch, CondScc0, CondScc1, CondVccz, CondVccnz, CondExecz, CondExecnz,
    End, Trap, IndirectUnknown,
};

// One guest basic block: a half-open instruction range in the decoded program.
struct GuestBlock {
    uint32_t first, last;          // indices into the instruction vector
    uint32_t pc;                   // pc of `first`
    Terminator term;
    std::array<uint32_t, 2> succ;  // block ids; kNone when absent
    uint32_t service_event = kNone;// a cross-lane service ending this block (dispatcher split policy)
};

struct GuestCfg {
    std::vector<GuestBlock> blocks;    // block 0 is the region entry
    std::vector<std::vector<uint32_t>> preds;
    std::vector<uint32_t> idom, rpo;   // dominators and reverse post-order
    std::vector<std::pair<uint32_t, uint32_t>> back_edges;
    std::vector<bool> reachable;
};

// How blocks are split. The structured emitter splits only at control flow. The dispatcher also
// splits after each synchronized service instruction (930-1184 today).
struct SplitPolicy { bool split_at_services; /* plus the service predicate */ };
GuestCfg build_guest_cfg(std::span<const Rdna2Inst>, const SplitPolicy&);

// The region tree. Leaves are block ranges. Unstructured is a sub-CFG the structurizer cannot
// express, and is emitted as a dispatcher.
struct Region {
    enum Kind : uint8_t { Seq, Block, IfThen, IfElse, Loop, CountedLoop, Unstructured } kind;
    uint32_t header, merge;            // block ids; merge is kNone for a region that ends the program
    BranchCondition condition;         // SCC / VCC / EXEC, with polarity and the wave-vote kind
    std::vector<uint32_t> children;    // Region ids, in emission order
    uint32_t continue_block = kNone;   // loops
};
struct RegionTree { std::vector<Region> regions; uint32_t root; };
RegionTree build_region_tree(const GuestCfg&, const RegionPolicy&);

}  // namespace prosper::gpu::ir::cfg
```

`RegionPolicy` holds the admission rules the detectors apply today: which branch conditions a
stage may structure, the `safe` linearizable `s_cbranch_execz` set, the workgroup-uniformity
proofs and the barrier constraints. Those rules move as they are; only where they are evaluated
changes.

### EXEC and the other masks

EXEC, VCC, SCC, M0 and every SGPR/VGPR word become **state variables**. SSA construction over
`GuestCfg` places a phi wherever a variable has more than one reaching definition. The structured
emitter already does this implicitly: "EXEC is phi'd across the merge like any other value", says the
comment above `emit_structured` (line 6831). The IR makes the placement explicit and computes it in one place,
instead of in `loop_written_regs`, the per-arm `then_written` sets and `lane_slot_carry`'s slot set.

EXEC values get static identities:

```cpp
struct ExecValue {
    enum Kind : uint8_t { Entry, Write, Phi, DispatchReload } kind;
    uint32_t block, inst;      // where it is defined
};
// Static relations between ExecValues, computed by the analysis (replaces RegState::exec_subset_of
// and exec_complement_in, which today are keyed by SPIR-V ids):
//   subset_of[a] = b   when a = b & x for a VOPC/V_CMPX or a proved narrowing
//   complement_in[a] = (b, c) when a = b & ~c
```

A `DispatchReload` is the EXEC a dispatcher case loads at its entry. Facts at a dispatcher block
entry are stated relative to that block's `DispatchReload` value, which is exactly what
`restore_active_vector_words` re-anchors by hand today.

Emission keeps a map from `ExecValue` to the SPIR-V id it emitted. That map is the only place an
owner fact touches a SPIR-V id.

### Definedness as one analysis

One lattice element per state word and per program point:

```cpp
struct WordFact {
    Domain domain;             // Data, Mask, MaskHalf{lo|hi, instance}, M0Token, Ambiguous, Absent
    Definedness inside;        // within `owner`: Defined, Arbitrary, Memory, Hard
    Definedness outside;       // outside `owner`: same lattice (#5202 needs inside != outside)
    ExecValueId owner;         // kAll for a whole-wave fact
    RootSet assumptions;       // loop-header assumptions this fact depends on (today's blanket roots)
};
// Spill slots (v_writelane lanes) and the LDS addtid slots are state words too.
```

- **Transfer**: one function per instruction, driven by `rdna2_instruction_effects` (what is read
  and written, widths, lane flow, EXEC-update kind). It replaces the ~520-line
  `advance_wave64_b64_masks`, the Wave32 dispatch-word transfer, the `SpilledMaskHalves` and
  `SpillSlotDomains` transfers, and the `record_*` side effects in `emit_alu`.
- **Join** at a merge: the existing join rules, written once. Domain conflicts go to `Ambiguous`,
  owners meet at the narrower one by `subset_of`, as `join_vector_word_marks` does with a `current`
  argument.
- **Edges**: an `s_cbranch_execz` skip edge applies `confine_execz_skip_marks`'s rule as an edge
  transfer.
- **Loops**: a real fixpoint. Header facts are the meet of the preheader and back-edge facts,
  iterated until stable. `LoopBlanketCheck`'s per-root voiding is kept as the way an assumption that
  fails voids only the reads that relied on it, but the solver iterates instead of refusing after
  one pass.
- **Solved once** per compiled region, before emission. The result is `facts_at(block, inst)` and
  `facts_at_entry(block)`.
- **Consumers query**, they do not record. `emit_alu`'s admission checks (`fresh_mask_sources_defined`,
  `vector_word_defined_in_exec`, the mask-domain reads) read `facts_at`. The dispatcher's `load_state`
  materializes `facts_at_entry` into `RegState` instead of filtering through eight parallel vectors.

One analysis serves both routes, so a fact cannot exist on one route and be missing on the other.
That is the class of bug behind #4949 and #5172-#5188.

### Emitting SPIR-V structured control flow

| Region | SPIR-V |
| --- | --- |
| `Seq`, `Block` | the blocks' instructions in order through the existing `emit_alu` path |
| `IfThen`, `IfElse` | header ends in `OpSelectionMerge merge` + `OpBranchConditional`. The condition is the region's wave vote: native subgroup vote, or the exact 32/64-lane scratch vote the structured compute path uses today. Phis at the merge for the state variables SSA placed there |
| `Loop` | header with `OpLoopMerge merge continue`, phis for the carried variables, the exit check, the body, and a continue block that branches back. This is the shape `emit_divloop` and the counted route already emit |
| `CountedLoop` | the same as `Loop`, with the counted-loop trip proof |
| `Unstructured` | the dispatcher: `OpLoopMerge` around `OpSwitch` over this region's blocks, Function variables for the state the region carries across its own edges, and the synchronized common phases after the switch |

Today the dispatcher is always whole-program. In the target, `Unstructured` can sit inside a
structured region and cover only the blocks the structurizer cannot express. **That is a behaviour
change, explained when it is made (step 11). It is not required for byte identity.**

### What keeps output byte-identical

| Rule | Why |
| --- | --- |
| CFG, region tree, SSA placement and analysis are pure data, built with no `SpirvCompute` call | SPIR-V ids are allocated in call order (`b.id()`). Pre-allocating one would renumber everything after it |
| Emitters walk regions in the order the old cursor walked `Fs`/`Ls` | label and phi ids are allocated in that order |
| Phi sets keep their old iteration order (`std::set<int>` over register numbers) | phi operand and result ids follow that order |
| Each new structure is **shadow-checked** against the old one before any consumer switches (region tree vs. `detect_forward_ifs`/`detect_divergent_loops`/`detect_counted_loop`, block lists vs. the dispatcher's `starts`, analysis facts vs. `RegState` and the pre-pass vectors) | a disagreement is found by a test or the corpus census, not by a changed hash |
| Refusal text stays the same, or the change is listed | the gate compares `stage-recompile` and `[recompile-reject]` lines as well as SPIR-V |

The gate's limits matter for this work. Compute programs compare SPIR-V only when they have a
sidecar and are accepted (raw compute blobs give only a verdict), and vertex replays mostly end
`incomplete`. **The dispatcher route, which this plan changes most, is the route the gate observes
least.** Steps 9-11 therefore also need execution tests.

## Interfaces

All private to `gpu/recompiler` (ADR 0012 decision 1). New code goes under
`gpu/recompiler/ir/cfg/` and `gpu/recompiler/ir/analysis/`, one file per role, each under 500 lines
and never over 1,000 (ADR 0012 decision 5).

| Interface | Consumers | Stable? |
| --- | --- | --- |
| `build_guest_cfg`, `GuestCfg` | every emitter and analysis that builds a block list today (19 files refer to `successors`) | yes, after step 1 |
| `build_region_tree`, `RegionTree` | `emit_body`, `emit_structured`, `emit_divloop`, the counted route, `compute_wave_route.cpp` | yes, after step 4 |
| `ExecValue`, `exec_relations` | the analysis, `exec_within`, owner meets | after step 7 |
| `WordFacts::facts_at` / `facts_at_entry` | `emit_alu` admission checks, `load_state` | after step 8 (shadow), consumed in steps 9-10 |

`RegState` keeps the SSA values (`vreg`, `sreg`, `exec`, ...) the emitter needs. Its fact fields
shrink as each consumer moves over.

## Failure modes

- A region the structurizer cannot express is an `Unstructured` region, not a refusal. Refusals
  inside it stay as today (`compute-cfg-reject`, `graphics-cfg-reject`; FAIL-1).
- A shadow check that disagrees is a test failure (`ctest`) or a named line in the corpus census.
  It is never an output change.
- Facts err only in the refusing direction. A read whose fact is not `Defined` (or `Arbitrary`
  where arbitrary is allowed) refuses loudly, as today. **The unsafe failure is a wrong fact that
  grants definedness**: the shader is accepted and reads garbage, and nothing reports it. Every step
  that moves a fact consumer needs a mutation arm that removes the source of the fact and shows the
  read refuses.

## Tests

| What | How |
| --- | --- |
| CFG builder | unit tests on synthetic streams for each `Terminator`, plus equality with the dispatcher's blocks and successors over every committed `tests/data/*.hex` fixture |
| Region tree | equality with the three detectors' outputs (pc level) over the fixtures and, as a `shader_inspect` census, over the corpus |
| Byte identity | the ADR 0006 corpus gate for every step: `recompile_corpus.py compare` against a `main` build |
| SSA placement | phi sets equal to the old write sets, per region, over the fixtures |
| Analysis (shadow) | per consult point, the new fact equals or refines the old one. Refinements are counted and listed by reason |
| Verdict flips (steps 9-11) | a GPU execution witness per flip group, plus `spirv-val` on newly accepted modules |
| Kena | a title-screen burst per step (`common-rules`), because #5026 is the reason for the work |

No test sees a fact that grants definedness wrongly on a program outside the fixtures and the
corpus. The mutation arms are the only defence there.

## Milestones

Each step is one PR, gated by the corpus gate. "Codex" means mechanical enough to hand over with a
recipe. "Lead" means it needs judgment about reverse-engineered semantics or proofs.

| # | Step | What moves | Risk | Tested by | Owner |
| --- | --- | --- | --- | --- | --- |
| **1** | **CFG and acyclic regions, shadow only** | New `ir/cfg/guest_cfg` and `ir/cfg/region_tree` for `Seq`/`Block`/`IfThen`/`IfElse`. A `shader_inspect --cfg-regions` census. Nothing consumes them | Low: no consumer, so byte-identical by construction. The work is matching `detect_forward_ifs`' admission rules exactly | unit tests; equality with `detect_forward_ifs` on every fixture; corpus census reporting matches and mismatches for the 266 acyclic structured programs | Codex builds it from a terminator table the lead writes; the lead reviews the policy |
| 2 | `emit_structured` takes if/else regions from the tree | the `bi` cursor over `Fs` becomes a walk over `IfThen`/`IfElse` children. `RegState` is still threaded | Medium: emission order must stay the same | corpus gate byte-identical; the fragment if/else fixtures | Lead review; Codex can implement |
| 3 | Mechanical split of the top-level helpers | lines 96-517 into their own files with `tools/refactor/split_file.py` | Low | byte-identical; `split_file.py`'s pure-move proof | Codex; can run in parallel with any step |
| 4 | Loop regions | `Loop` and `CountedLoop` in the tree, shadow-checked against `detect_divergent_loops`/`detect_counted_loop`, then consumed by `emit_divloop` and the counted route | Medium-high: loop carry and the counted prelude/postlude | corpus gate; loop fixtures; Kena burst | Lead |
| 5 | Dispatcher blocks from `GuestCfg` | `start_set`, `block_for_pc` and `successors` (930-1184) come from `build_guest_cfg` with the service split policy; `Unstructured` regions are marked | Medium: `proven_exit_target`, `s_trap` and indirect edges are modelled differently today | equality first, then byte-identical | Codex implements; lead reviews the edge cases |
| 6 | SSA placement for state variables | the phi sets (`loop_written_regs`, `then_written`, lane-slot carry) come from one placement over `GuestCfg` | Medium: iteration order is the byte-identity contract | phi-set equality; corpus gate | Lead |
| 7 | EXEC as an IR value | `ExecValue` identities and static subset/complement relations; emission maps them to SPIR-V ids; `exec_subset_of` becomes a view of them | **High**: a wrong mapping grants definedness silently | shadow equality with `exec_subset_of` at every `exec_within` call; mutation arms | Lead |
| 8 | **One definedness analysis, shadow only** (needs an ADR) | `ir/analysis/word_facts`: the lattice, one transfer driven by `rdna2_instruction_effects`, a loop fixpoint. A census compares it with `RegState` marks (structured) and the pre-pass vectors (dispatcher). Nothing consumes it | Medium: large, but no output change | fixture and corpus census: equal, refines, contradicts. A contradiction blocks step 9 | Lead |
| 9 | Dispatcher consumes the analysis | `load_state` materializes `facts_at_entry`; the dispatcher pre-pass is deleted (most of 1185-2572) | High: **expected verdict flips**, refused to accepted, where the old model lost a fact (e.g. some of the 62 ambiguous-mask and 152 unresolved-operand dispatcher refusals) | explained corpus diff grouped by reason; an execution witness per group; `spirv-val`; Kena burst | Lead |
| 10 | Structured path consumes the analysis | `emit_alu` checks read `facts_at`; the `record_*`/`join_*` side effects and `LoopBlanketCheck` go. The two-sided mark (#5202) arrives here | High: same flip class as step 9, on the route the gate sees best | as step 9, plus the parked tests | Lead |
| 11 | Dispatcher lowered from the region tree | `Unstructured` regions emitted from the tree; the 21 service kinds become synchronized service nodes at block ends. Optional local dispatcher (a behaviour change) | High | corpus gate; execution tests for every service kind; Kena burst | Lead |
| 12 | Delete the old paths and cap the files | the detectors in `rdna2_cfg_support.hpp`, the dead `RegState` and `SpirvCompute` fact fields; every file at or under 1,000 lines; ratchet rows lowered with `--update` | Low once 9-11 have landed | byte-identical | Codex |

**First step: step 1.** It is the smallest step that adds value: one CFG and one if/else region
tree that every later step builds on, checked against the existing detector on the 266 acyclic
programs the structured route compiles, with no output change.

**Step 1 result** (`src/gpu/recompiler/ir/cfg/`, contract in its `AGENTS.md`). `shader_inspect
--cfg-regions` over the 1,003 distinct corpus programs: 362 agree with structured ifs (1,931 ifs
equal at the pc level; fragment 176, vertex 74, compute 112), 266 agree with none, 147 agree on
refusing, 226 hold a recognized loop and are not compared, 1 disagrees, and one corpus file is
empty. The fragment count equals the 176 in the route table above. The structured counts are
larger for vertex and compute because the census compares the structurers whatever route
`emit_body` then takes (compute sends most of them to the dispatcher). The disagreement is a
truncated 64 KiB compute dump with no `s_endpgm`: `detect_forward_ifs`' forward bounds use
`end_pc`, which stays `UINT32_MAX` without one, so it admits an else merge outside the decoded
stream, and the tree refuses it.

**Step 4a result** (loops in the tree, shadow only; `region_loops.cpp`, contract in
`ir/cfg/AGENTS.md`). The loops come from `GuestCfg`'s back edges and dominators; their admission
calls the detectors' own rules, factored out of `detect_counted_loop` and `detect_divergent_loops`.
The census now compares the 226 loop programs step 1 left out: 151 agree on divergent loops (471
loops and 2,086 ifs equal at the pc level, breaks included; 100 programs hold more than one loop
and 133 have ifs around or inside them), 42 agree on a counted loop, 32 agree on refusing, and 1
disagrees. All 1,003 programs: 362 structured, 266 linear, 151 loops, 42 counted, 179 refused, 2
disagree, 1 empty file. The new disagreement is `compute/cs_d9ed6de0de75c8778e229814.bin`, and
it is the shadow's scope, not a structurer difference: the counted detector finds its loop, so the
shadow compares only that loop, but `emit_body`'s counted route declines the program (its prelude
scan refuses the VCC early-out at pc 41, whose region holds an `s_barrier`) and the general route
then refuses it at the same branch. The tree refuses it too, first at an EXECZ if at pc 67 nested
under that branch. Every route refuses; the shadow does not model the counted route's prelude
decline. Step 4b consumes `Loop` in `emit_divloop` and `CountedLoop` in the counted route.

**Step 6 result** (`ir/cfg/state_placement`, contract in `ir/cfg/AGENTS.md`). One pass over the
region tree gives each if merge its phi set and each loop header its carried, condition and
lane-slot sets, with the writes taken from the instruction-effects table. The emitter computes its
own sets as before and takes the placement's only when they are equal element for element, so the
corpus gate reports `changes: none`. `shader_inspect --cfg-placement` over the 4,000 corpus entries
(raw, synthetic-table and `--inputs` recompiles): 2,245 entries reach a guarded join. Per entry
that is 10,155 if merges and 737 loop headers, each header with three sets: 12,366 sets compared.
All are equal except 12 if merges in 8 entries, and no join lacks a placement. Each of the 12 is the high
SGPR of an `s_bfe_u64` (seven vertex programs) or of an `s_bfm_b64` into EXEC (one compute
program): the emitter's scan names three 64-bit SOP2 opcodes and charges the low word for the rest,
and the table cannot tell those rows apart, so the placement charges the pair. That is an existing
defect, #5436. Over the committed fixtures, the 104 ifs and 9 loops the detectors find are all
equal, and every fixture instruction's inventory equals the scan's.

**Step 4b result** (`ir/cfg/structured_loop_cursor`, contract in `ir/cfg/AGENTS.md`). `emit_body`'s
divergent-loop cursor and the counted route's `CountedLoop` read the tree's records when the whole
sequence equals the detector's field for field, and the detector's otherwise. The corpus gate
against `fac83c577` reports `changes: none` (4,000 entries, 1,444 SPIR-V modules compared). A
temporary instrument (not committed) counted the routes over the 1,003 distinct programs, once per
`emit_body` call that reached the decision: the divergent cursor took the tree in all 64 calls that
held loops (compute 21, fragment 33, vertex 10) and had no loops in 274; the counted record took the
tree in 40 calls (compute 16, fragment 5, vertex 19) and fell back in 1. That one,
`compute/cs_05e14dc59e6261dada40eac5.bin`, has a bottom-tested EXEC loop before its counted loop:
the tree takes the general route (two divergent loops), so the records differ, and `emit_body`'s
prelude scan declines the counted route as well. No call was `Unstructured` or `DetectorRefused`;
the latter cannot occur at the cursor today, because `cf_rejected` clears `Ls` first.

**Step 5 result** (`ir/cfg/dispatch_blocks`, the service split in `guest_cfg`; contract in
`ir/cfg/AGENTS.md`). `build_guest_cfg` with `split_at_services` reproduces the dispatcher's blocks
and successors; the service predicate stays in `emit_cfg_state_machine`, which passes the pcs it
synchronizes as `SplitPolicy::services`. The dispatcher takes `starts` and `successors` from the
GuestCfg only when they are identical to its own. `shader_inspect --cfg-dispatch-blocks` over the
corpus (each distinct program once by `--stage`, `--synthetic-table` for graphics, and every
sidecar by `--inputs`; 4,570 runs) saw **2,213 dispatcher regions, all equal**, from 377 distinct
programs (compute 2,172 regions, fragment 14, vertex 27); the one run without an answer is the
empty corpus file. The corpus gate against `a43dcc9cc` reports `changes: none` (4,000 entries,
1,444 SPIR-V modules compared). Two mutation arms: a builder that ignores the services makes 678
regions differ and still reports `changes: none`, because the guard keeps the dispatcher's own
blocks; the same builder without the guard reports 29 changed programs (3 newly accepted, 3 newly
refused, 14 changed output). The one shape known to differ is outside the corpus: a branch target
inside an instruction, where the dispatcher starts a case at a pc no decoded instruction begins
(`TheDispatcherKeepsItsOwnBlocksForATargetInsideALiteral`). That is why the dispatcher's own
computation and the guard stay: without them such a program would no longer get a case at
that pc. `Unstructured` regions are not marked yet.

**Step 7a result** (`ir/cfg/exec_values`, `ir/cfg/exec_value_census`; contract in
`ir/cfg/AGENTS.md`). Every EXEC definition has a static value (`Entry`, `Write`, `Phi`,
`DispatchReload`), and the subset and complement relations are the ones the emitter's recognisers
record, stated over those values. Nothing in emission reads them: the corpus gate against
`f0be48728` reports `changes: none` (4,000 entries, 1,444 SPIR-V modules compared). `shader_inspect
--cfg-exec-values` over the corpus (each program by `--stage`, `--synthetic-table` for graphics and
every sidecar by `--inputs`; 10,214 runs, 2,470 of them reaching `exec_within`) answers each
`exec_within` call and complement read again from the static relations: of 426,613 `exec_within`
answers, 421,455 are equal, 4,100 static-stronger, **0 static-weaker** and 1,058 unmapped; of
333,524 complement reads, 327,411 are equal, 6,113 static-stronger and 0 static-weaker. The first
run had 5 weaker answers and 4 weaker complement reads, all from one shape (`s_cselect_b64 s[N:N+1],
exec, 0` then `s_mov_b64 exec, s[N:N+1]`): a move into EXEC from a pair holding no known mask now
names that pair with the new value, as S_MOV_B64 copies it. The stronger answers are a SPIR-V id the
emitter replaced while the guest value did not change (2,220 at joins where every edge carries the
same value, 476 at a move from a saved EXEC the emitter rebuilt from words, 91 at dispatcher reloads)
and chains of relations the emitter records only in part (1,313). The unmapped ones are ids no
instruction hook sees: 534 with the reader unmapped, such as the merge phi of two structured ifs
that share one guest merge block (the graph has one phi there); 446 with the owner unmapped,
mostly at dispatcher case entries, all of which the SPIR-V relation answers no too; 78 with both. Removing the
compare rule from the static pass makes 147,327 answers static-weaker, and removing the complement
rule 34,440 complement reads, so the census does see a missing rule. Step 7b maps `ExecValue` to
SPIR-V ids in emission and makes `exec_subset_of` a view of these relations.

**When the parked red tests become fixable:**

- `indirect_pointer_descriptor_range` has two gaps (#5202). The first, charging a Wave32 lane-mask
  source as a scalar pair, was a source-width bug: the source marks now charge it one SGPR, except
  VCC_LO, whose never-written VCC_HI is the marks' only witness of an unprovided entry VCC. That
  moved the fixture's refusal from pc214 to pc582 (`v_cmp_eq_u32_sdwa s4, vcc_lo, v0` after
  `v_readfirstlane_b32 vcc_lo, v0`). Whether pc582 is the second gap is not yet attributed. The
  second gap needs the "arbitrary inside the owner, hard outside" form that `ScalarSourceMarks`
  cannot express. In this plan that form is part of `WordFact` (step 8) and reaches the structured
  emitter in step 10. Alternatively a field can be added to `ScalarSourceMarks` now, to be subsumed
  later (open question 2).
- `rdna2_to_spirv_exec` fails 11 of its 1192 checks, the same 11 on main and with the #5202 fix.
  The breakdown is on #5039. Two groups belong on this timeline. In the first, the GTA Wave64
  VCC_LO dispatch fixture is refused at pc15 because a constant-lane `v_readlane_b32` of the one lane
  a narrowed write defined is marked hard. This is the same lane-confinement fact as #5202's second
  gap. In the second, the three #2481 checks are refused at pc257 (`s_mov_b64 exec, s[6:7]`) because
  the definedness of `s[6:7]`, written once on every path, does not survive the uniform branch
  arms. Two groups are not yet classified: the pc213 same-site mutation (refused at pc232) and the
  six portable/native ballot checks (native refuses with `wave64-ambiguous-mask-read`).

## Open questions

1. **Does step 8 need its own ADR?** Recommended: yes. Steps 1-7 restructure control flow, which
   ADR 0012 already covers. Step 8 makes one analysis the single owner of validity facts. That
   replaces most of the 37 analyses #5146 counts, and step 9 knowingly flips verdicts. That is an
   architectural decision for the owner, not a family migration.
   Drafted as ADR 0038 (`docs/adr/0038-one-definedness-analysis.md`, proposed), which also
   recommends an answer to question 4.
2. **#5202 now or in step 10?** Adding an inside/outside split to `ScalarSourceMarks` now would
   unblock `indirect_pointer_descriptor_range` and some Kena guards sooner, at the cost of one more
   field that step 8 must reproduce exactly. Owner's call.
3. **Local dispatcher regions (step 11).** A smaller dispatcher keeps SSA facts in the structured
   parts around it. Untested speculation: this would recover some of the 243 programs whose
   dispatcher attempt fails. `CONFIDENCE: LOW` until step 8's census says how many of those
   refusals come from facts lost at case boundaries.
4. **Gate coverage.** Compute programs without a sidecar compare verdicts only, and vertex replays
   mostly end `incomplete`. Should steps 9-11 wait until the corpus has more compute sidecars and
   complete vertex inputs, or rely on execution witnesses?
5. **Who builds step 1?** It suits Codex if the lead first writes the terminator table and the
   `RegionPolicy` contract (which branch conditions each stage may structure, the `safe` set,
   uniformity proofs) as a section of `ir/HOW_TO_ADD_A_FAMILY.md`.

## Ruled out

- **#5202's second gap was the refusal at pc507.** With only the structured walk charging one SGPR,
  the fixture is refused at pc507 (`v_add_co_u32 v11, vcc_hi, v27, v12`). When the dispatcher's
  `spilled_vector_source_marks` charges one SGPR too, pc507 passes and the refusal moves to pc582.
  So pc507 was the first gap on the dispatcher route, not the second gap (#5202).
