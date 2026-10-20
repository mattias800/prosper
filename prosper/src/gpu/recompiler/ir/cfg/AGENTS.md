# `gpu/recompiler/ir/cfg` — the guest CFG and its region tree

Steps 1, 2, 4, 5, 6 and 7a of `docs/architecture/IR_CONTROL_FLOW_PLAN.md`. One control-flow graph built once
from the decoded program (`guest_cfg`), and the structured regions derived from it
(`region_tree`): ifs (step 1) and loops (step 4). Both are plain data: building them allocates no
SPIR-V id and calls no `SpirvCompute`, so they cannot change an emitted byte. The shadow check
(`region_shadow`) compares the region tree with `detect_forward_ifs`, `detect_divergent_loops` and
`detect_counted_loop` in a gtest over the committed fixtures and in `shader_inspect --cfg-regions`
over the corpus.

**Consumers (steps 2 and 4b).** `structured_if_cursor` walks the tree's IfThen/IfElse children
and `structured_loop_cursor` reads its loop records (see [Consumers](#consumers-step-4b)), each only
when the tree's complete record sequence equals the detector's. Any disagreement keeps the detector
records for the whole program. Register state, joins, proof facts and the dispatcher stay with their
existing owners. A consumer switches over after its own shadow check agrees. The CFG dispatcher
(step 5) reads its block list and successors through `dispatch_blocks` under the same guard; see
[The dispatcher's split](#the-dispatchers-split-step-5).

| path | role |
| --- | --- |
| `guest_cfg.hpp/.cpp` | `GuestBlock`, `GuestCfg`, `build_guest_cfg`: blocks, successors, preds, rpo, idom, back edges, reachability |
| `region_tree.hpp/.cpp` | `Region`, `RegionTree`, `RegionPolicy`, `build_region_tree`; the walk, `Seq`, `Block`, `IfThen`, `IfElse`, `Unstructured` |
| `region_loops.cpp` | `Loop` and `CountedLoop`: finding the loops in the CFG and admitting them |
| `region_builder.hpp` | the builder class the two `.cpp` files share; internal, not an interface |
| `structured_if_cursor.hpp/.cpp` | guarded tree preorder for `emit_structured`, with whole-program legacy fallback |
| `region_shadow.hpp/.cpp` | the pc-level comparison with the three detectors, shared by the test and the census |
| `structured_loop_cursor.hpp/.cpp` | step 4b: the DivLoop cursor and the CountedLoop record `emit_body` consumes, from the tree when equal to the detector |
| `dispatch_blocks.hpp/.cpp` | step 5: the dispatcher's `starts`/`successors` from the service split, the guard that adopts them only when identical, and `DispatchBlocksCapture` for the shadow test and census |
| `state_placement.hpp/.cpp` | step 6: the phi sets of every if merge and loop header, the guard the emitter takes them through, and the census that observes it |
| `exec_values.hpp/.cpp` | step 7a: a static identity for every EXEC value and the subset/complement relations between them |
| `exec_value_census.hpp/.cpp` | step 7a: the shadow census, which answers every `exec_within` call and complement read again from `exec_values` |
| `exec_relation_consumer.hpp/.cpp` | step 7b part 1: scoped id-to-value mapping and equality-guarded answers, with id-based fallback |

Tests: `tests/gpu/recompiler/ir/test_ir_cfg_*.cpp`, one per source file.

## Terminators (the contract later steps depend on)

A block ends after an instruction whose `Rdna2InstructionEffects::control` is not `None`, and before
every in-stream branch target. **The PC effect comes from the effects table; this folder keeps no
opcode list.** Add a PC-changing instruction to the table, never here.

| `control` (table) | Instructions | `Terminator` | `succ[0]` | `succ[1]` |
| --- | --- | --- | --- | --- |
| `None` | everything else | `Fallthrough` (when the next instruction is a leader) | next block | - |
| `Jump` | `s_branch` | `Branch` | - | target block |
| `BranchScc0/Scc1/Vccz/Vccnz/Execz/Execnz` | `s_cbranch_*` | `CondScc0` ... `CondExecnz` | next block | target block |
| `End` | `s_endpgm`, `s_endpgm_saved`, `s_endpgm_ordered_ps_done`, `s_code_end` | `End` | - | - |
| `Trap` | `s_trap` | `Trap` (the wave ends, as the dispatcher models it) | - | - |
| `Indirect` | `s_setpc_b64`, `s_rfe_b64` | `IndirectUnknown` | - | - |
| `IndirectCall`, `Call` | `s_swappc_b64`, `s_call_b64` | `IndirectUnknown` | return point (next block) | - |
| `DebugBranch` | `s_cbranch_cdbg*` | `IndirectUnknown` | next block | - |

A branch target is `pc + len + SIMM16` in dwords. A target that is not the pc of a decoded
instruction (past `s_endpgm`, inside a literal, outside the stream) leaves `succ[1] = kNone` and is
kept in `taken_pc`. `back_edges` are retreating DFS edges (`succ[0]` visited first).

`SplitPolicy::split_at_services` is the dispatcher's split, which differs from this table in more
than the services: see the next section.

## The dispatcher's split (step 5)

`build_guest_cfg` with `SplitPolicy::split_at_services` reproduces `emit_cfg_state_machine`'s
blocks and successors exactly. **The service predicate stays in the dispatcher**: it records the pc
of every instruction it synchronizes (`split_service`, the DPP quad-OR events included) and passes
that set as `SplitPolicy::services`; graphics stages pass `safe` as `linearized`. This folder keeps
no list of services.

| | control-flow split (default) | service split |
| --- | --- | --- |
| Instructions | the whole stream | up to the first `is_end` (`s_endpgm`); none is an empty graph |
| `s_endpgm` | `End` (also `s_endpgm_saved`, `_ordered_ps_done`, `s_code_end`) | `End` only for `is_end`; the other forms are ordinary instructions |
| `s_trap` | `Trap`, block ends after it | `Trap`, and a block also starts **at** it |
| Services | ordinary instructions | a block of their own, `service_event` = its pc |
| `s_setpc`, `s_rfe`, `s_swappc`, `s_call`, `s_cbranch_cdbg*` | `IndirectUnknown` | ordinary instructions: the dispatcher does not split at them |
| A branch in `linearized` | a branch | an ordinary instruction (the stage predicates it) |
| Target past `s_endpgm` | `succ[1]` when decoded there | no edge; `proven_exit_target` is the per-case emission's business |

`dispatch_blocks` turns a graph into the dispatcher's form: `starts` (block pcs) and, per block, the
taken successor then the fall-through one, without repeats. `adopt_guest_cfg_blocks` builds that
and replaces the dispatcher's own `starts`/`successors` **only when they are identical**, so
emission is byte-identical by construction; otherwise the dispatcher keeps its own. The one shape
known to differ is a branch target inside an instruction (a literal): the dispatcher starts a case
at that pc and the graph cannot, since no decoded instruction begins there. The dispatcher's own
computation is kept for that reason; the census found no such program.

Observation: `DispatchBlocksCapture` records every verdict on its thread
(`test_ir_cfg_dispatch_blocks`, `shader_inspect --cfg-dispatch-blocks`). A region the dispatcher
refuses before building its blocks (a service contract, scratch size, a target before the region)
makes no record.

## `RegionPolicy`: what the tree may structure

The rules are the detectors' own (`../../rdna2_cfg_support.hpp`). The tree calls the same
predicates, never a copy. Ifs: `classify_forward_if_branch`, `resolve_forward_if_target`,
`compute_execz_region_admissible`, `wave_vcc_region_admissible`, `forward_if_workgroup_uniform`.
Loops: `counted_loop_backedge_candidate`, `counted_loop_from_backedge`,
`counted_route_exec_guard`, `div_loop_backedge_candidate`, `div_loop_from_backedge`,
`sort_div_loops_by_header`, `validate_div_loops`. A rule change goes there, and both structurers
follow it.

| Rule | Graphics (`allow_vcc`) | Compute (`compute_wave_branches`) |
| --- | --- | --- |
| `s_cbranch_scc0/1` | an if, unless in `linearized` (kill-mask tests, waterfalls) | an if, unless in `linearized` |
| `s_cbranch_vccz/vccnz` | an if (a vote) | an if decided per guest wave (`wave_vote`) |
| `s_cbranch_execz` | an if, unless in `linearized` (`safe_execz_branches`) | an if (a subgroup vote), unless in `linearized` |
| `s_cbranch_execnz` | refused, unless it is a loop's back edge | refused, unless it is a loop's back edge |
| Neither flag set | VCC and EXECZ branches are refused | - |
| Target | forward and at most `end_pc`; past `s_endpgm` only when the code there ends at once (an early-out, `merge = kNone`) | same |
| Else | the block before the target ends in a forward `s_branch` whose target lies after the if's target and at or before `end_pc` | same |
| EXECZ region | - | no `s_barrier`; a scalar write to `vcc_lo/hi` must be dead at the merge |
| VCC region | - | unless proved workgroup-uniform (#1554): no `s_barrier`, `ds_swizzle`/`ds_append`/`ds_consume`/`ds_bpermute` or `v_mbcnt_lo/hi` (`wave_vcc_region_admissible`) |
| Workgroup uniformity | - | VCC (#1554) and SCC (#3713) proofs set `uniform_workgroup` |

Structure: each arm is built over its own block range, so a branch from one arm into another, a
plain if leaving its range, an unclaimed `s_branch`, a call, or an unadmitted back edge refuses. An if/else
whose merge lies past its enclosing range is accepted as the detector accepts it (the
shared-outer-merge cascade) and recorded with `end_block < merge`. **Any refusal makes the whole
program one `Unstructured` region**, which is today's behaviour: the detector refuses the program
and the whole-program dispatcher takes it. A local `Unstructured` inside structured code is plan
step 11.

Known differences from the detector, both in the refusing direction (the tree refuses, the detector
admits): an if in an if/else's then-arm whose target lies in the else-arm (the detector bounds it by
the merge, not the arm), and a target or merge outside the decoded stream when the stream has no
`s_endpgm` (the detector's bounds are then vacuous, #5415). The first is pinned by
`test_ir_cfg_region_shadow`, and `emit_structured` refuses that program anyway (the vertex stage
stops at the then-arm's `s_branch`), so the detector's admission is not a silent miscompile. The
second is one truncated corpus dump.

## Loops (step 4)

**Finding them.** Every CFG back edge must end in a backward branch whose target block dominates
its source (`idom`); anything else is `irreducible back edge`. Each back edge goes to the detector
whose candidate it is: `counted_loop_backedge_candidate` (a backward `s_branch`, or a backward
`s_cbranch_scc0/1` that is not a waterfall) and `div_loop_backedge_candidate` (those plus a
backward `s_cbranch_execnz`, minus the stage's `linearized` set). A backward branch that is
neither and is not linearized is refused. A linearized one (a waterfall, a safe EXECZ) is not a
loop: its block is a plain `Block`. The CFG sees only reachable code, the detectors scan pcs; a
loop nothing can reach is the one known difference (`AnUnreachableLoopIsADisagreement`).

**Which route.** `emit_body`'s order:

1. **Counted.** Exactly one counted candidate, and `counted_loop_from_backedge` accepts it. If a
   divergent candidate lies before its header, `emit_body`'s prelude scan refuses and falls back to
   the general route, and so does the tree. One after it (the counted postlude) is refused: the tree
   does not model the postlude route yet.
2. **Divergent.** Otherwise the divergent candidates in header order, through
   `div_loop_from_backedge` and `validate_div_loops` (the pairs, per-loop, children and entry-edge
   rules). One refusal refuses the program.

On top of the rules, the tree checks in block terms that each loop is the contiguous range
`[header, latch]`, entered only at its header and left only for the block after its latch.

**Shapes.** A `Loop`/`CountedLoop` region covers `[header, latch + 1)`. `header` is the back
edge's target, `continue_block` the latch (the block ending in the back edge), `merge` the block
after it (`kNone` past the stream), and `exit_branch` the block ending in the canonical exit test.
`children = {body}`, a `Seq` over the same range that starts with the header block. The back edge
and the canonical exit are claimed: their blocks are plain `Block`s, never ifs.

| Shape | Back edge | Exit test (`exit_branch`) | `condition.flag` | `loop.continue_on_set` |
| --- | --- | --- | --- | --- |
| top-tested EXEC | `s_branch` | first branch after the branch-free condition region: `s_cbranch_execz` to the merge | `Exec` | true |
| top-tested VCC | `s_branch` | `s_cbranch_vccz` to the merge, if `vcc_exit_is_wave_uniform` or `fragment_wave_breaks` | `Vcc` | true |
| top-tested SCC | `s_branch` | `s_cbranch_scc0` (continue while set) or `scc1` (leave while set) | `Scc` | scc0 |
| execnz, header exit | `s_cbranch_execnz` | the header itself is the `execz` exit | `Exec` | true |
| bottom-tested EXEC | `s_cbranch_execnz` | the back edge (`loop.bottom_tested`); the instruction before it writes EXEC | `Exec` | true |
| bottom-tested SCC | `s_cbranch_scc0/1` | the back edge; one straight-line SOPC decides it (`scc_do_while_condition_pc`) | `Scc` | scc1 |
| `CountedLoop` top-tested | `s_branch` | the single `s_cbranch_scc0/1` to the merge | `Scc` | scc0 |
| `CountedLoop` do-while | `s_cbranch_scc0/1` | the back edge; no other exit | `Scc` | scc1 |

**Breaks** (`Loop` only): a further forward `s_cbranch_vccz`/`execz` to the merge, listed in
`breaks`. Each is an `IfThen` with `loop_break` whose arm is the rest of the loop body (latch
included): the lowering skips to the back edge, where the cleared lane leaves at the next test. A
break must sit at the top level of its loop body; inside an inner if it refuses
(`break leaves its region`), as the detector's nesting check does. `loop.direct_exec_breaks`
(an execz break with an `s_branch` back edge) and `loop.direct_wave_breaks` (a fragment vccz
break) are the detector's flags, copied.

**Nesting.** Ifs nest inside loop bodies and loops inside if arms and other loop bodies, each
over its own block range; a loop leaving its enclosing range refuses. A forward if whose target
follows an inner loop's latch is not an if/else: the latch's `s_branch` is a back edge.

**Counted-loop proof inputs.** `CountedLoop` records what `counted_loop_from_backedge` proved:
the header, the one exit test (or the do-while back edge), the back edge, the exit and its SCC
polarity. The trip count itself is not proved here; the counted lowering reads it from the SCC
compare as it does today. Inside the counted route an `s_cbranch_execz` in the loop, or one
before it targeting the header or later (`counted_route_exec_guard`), is a plain `Block`: the
counted lowering linearizes a guard it proves (`mark_counted_loop_exec_guards`) and `emit_alu`
refuses the rest, so neither is an if.

**Known shadow difference.** For a counted program the shadow compares only the loop. The
counted route checks its prelude and postlude with separate `detect_forward_ifs` calls over
truncated streams, and declines to the general route when the prelude scan refuses; the shadow
models that decline only for a loop in the prelude. One corpus program shows the rest (the step 4a
result in the plan): every route refuses it, and the shadow reports the counted detector's loop
against the tree's refusal.

## Consumers (step 4b)

`emit_body` reads two things from `structured_loop_cursor`: the divergent-loop cursor that
`emit_structured` walks (`next_before`, `take`, `next_pc`, replacing the `li` index over `Ls`), and
the counted route's `CountedLoop`. Both take the tree's records **only when the complete sequence
equals the detector's, field for field and in order**: every `DivLoop` field (`header_pc`,
`exit_branch_pc`, `backedge_pc`, `exit_pc`, `break_pcs`, both break flags, `bottom_tested`,
`condition`, `continue_on_set`), and every `CountedLoop` field. The tree's loops are read in
preorder, which is header order. Anything else keeps the detector's records for the whole program,
and `route()` says why: `NoLoops`, `DetectorRefused` (`cf_rejected`), `Unstructured`, or
`Mismatch` (a different field, count or order, a loop of the other kind, or a malformed walk).
Equal records make emission byte-identical by construction, so the corpus gate must report
`changes: none` for this step. `RegState`, joins, loop carry, phis, proof facts and
`LoopBlanketCheck` are untouched.

The divergent cursor is built from emit_body's **final** `Ls`, after the compute guards and the
`cf_rejected` clear, so a loop the compute guards dropped is `NoLoops` rather than a mismatch. The
`RegionPolicy` is `structured_loop_policy`: the detectors' own inputs (`safe`, the stage, the
fragment wave64 vote).

## State placement (step 6)

`place_state` walks the tree from its root and records, per if merge (keyed by the branch pc) and
per loop header (keyed by the header pc), the state variables the structured emitter phis there:

| Join | Set | Blocks |
| --- | --- | --- |
| IfThen, IfElse | `merge`: written in either arm | each arm child's `[first_block, end_block)` (an IfElse else-arm is clamped to the enclosing range, as the emitter's `else_hi` is) |
| Loop, CountedLoop | `carried`: one header phi each | `[header, continue_block]` |
| | `condition`: the check block's value leaves the loop | `[header, exit_branch]` |
| | `lane_slots`: constant-lane V_WRITELANE `(vgpr, lane)` | `[header, continue_block]`, before the program's first `s_endpgm` |

The block ranges hold one more instruction than the emitter's pc ranges (the then-arm's `s_branch`,
the back edge, the exit test), and a branch writes nothing. VCC, SCC, EXEC and the saved lane
masks (`sreg_bool`) are not in these sets: the emitter phis them from `RegState`. In the code,
`then_written` is a copy of `RegState::sreg_written` (a fact, not a phi set); the phi sets are
`written_v/written_s`, `ifv/ifs` and `wv/ws`.

**The inventory** (`add_phi_writes`). The effects table and the shared width helpers answer; this
folder still keeps no opcode list. It reproduces the emitter's scan (`loop_written_regs`, owned
packet false), which is not the textbook "every register written":

| Format | Charged | Against the textbook |
| --- | --- | --- |
| VOP1 | `v_readfirstlane` (`rdna2_lane_read_writes_sgpr`): its SGPR. The one row whose destination alone is M0-relative (`v_movreld`): every VGPR from VDST to the program's highest. Otherwise VDST | finer: a 64-bit result's second VGPR, `v_swap_b32`'s SRC0, `v_movrelsd`/`v_swaprel`'s relative target; coarser: `v_nop` and the other no-result rows charge their VDST field |
| VOP2, VOP3P | VDST | carry-out VCC is VCC's own phi |
| VOP3 | `v_readlane`: its SGPR. Otherwise the table's VDST width (`rdna2_vgpr_write_count`) | finer: VOP3 `v_readfirstlane` (width 0), the M0-relative forms (VDST only), VOP3B carry masks (mask domain) |
| VINTRP | VDST, except P1 (`rdna2_vintrp_is_p1`: the native lowering folds it into P2) | - |
| DS | `rdna2_vgpr_write_count` | - |
| MUBUF, MTBUF, MIMG, FLAT | `rdna2_vgpr_write_count`, plus the TFE status VGPR | - |
| SOP1 | one word; the pair for a 64-bit result its lowering writes as scalar data (`s_mov_b64`, `s_bitreplicate_b64_b32`, `s_getpc_b64`, intrinsic numeric `s_wqm_b64`, `s_quadmask_b64`); the low word for a mask-domain pair and for EXEC-only writes (`sop1_phi_words`) | finer: SAVEEXEC/WREXEC, `s_not_b64` and mask-domain `s_wqm_b64` charge their low word; coarser: `s_mov_b64`/`s_bitreplicate_b64_b32` into EXEC charge `s126` although they write `rs.exec` only, an M0-relative destination charges the field, and rows that write nothing (`s_setpc_b64`) charge the field |
| SOPK | the SDST word when the table says it writes one (or does not know) | - |
| SOP2 | one word; the pair for a 64-bit scalar-data result; nothing for a 64-bit shift into EXEC | finer: a 64-bit bitwise op of two pairs (`s_and_b64`, ...) charges its low word: its value lives in the mask domain |
| SMEM | what SDATA receives; one register for a reserved encoding | - |
| VOPC, SOPC, SOPP, EXP | nothing | VCC, SCC and EXEC have their own phis |

**Where the placement and the emitter's scan differ: nowhere, per instruction.** Both charge SOP2
through `sop2_phi_words` and SOP1 through `sop1_phi_words` (`state_placement.hpp`). Until #5436
the scan named three opcodes (`s_cselect_b64`, `s_lshl_b64`, `s_lshr_b64`) and charged the low word alone for `s_ashr_i64`,
`s_bfe_u64`, `s_bfe_i64` and `s_bfm_b64`, although their lowerings write both words: a high SGPR
written in an arm got no merge phi and kept an arm-local value that does not dominate the merge.
It also charged one or two words for `s_lshl_b64`, `s_ashr_i64` and `s_bfe_*` into EXEC, which now
charge nothing. `s_bfm_b64` into EXEC charges the pair, since its sources are not a pair; its
lowering writes only the per-lane EXEC value, so that is an over-charge. The CFG dispatcher's
persisted set comes from the same scan, so it now also keeps `s107` after `s_bfe_u64`/`s_bfm_b64`
into VCC and `s127` after `s_bfm_b64` into EXEC. SOP1 had the same defect until #5439: one word
for every row, so `s_mov_b64` and `s_bitreplicate_b64_b32` into an SGPR pair dropped the high word
at every merge. #5441 extends the pair charge to intrinsic numeric WQM and QUADMASK using
`scalar_write_is_b64_mask`, the lowering's existing predicate. Structured arms clear
`scalar_presence_has_no_placeholders`, so WQM's state-dependent SGPR numeric route cannot apply
there. Mask WQM (including quad-closed legacy inline masks), NOT and EXEC-only rows retain their
low-word charge; no scalar source-domain analysis is added.

**The guard** (`take_placed_if_equal`). The emitter computes its own set as before and takes the
placement's only when the two are equal element for element; a missing placement or any
difference keeps its own. Since #5436 the per-instruction inventories agree, so a difference can
only come from the region ranges; the census over the shader corpus finds none. `if_join_regs` and `loop_join_regs` (`../../rdna2_cfg_registers.hpp`)
and `LaneSlotLoopCarry::open` are the call sites. The dispatcher's sets, `loop_scalar_may_writes`,
the PROSPER_DBG loop trace and `rdna2_vector_word_marks` still call `loop_written_regs` directly.

**The census.** `StatePlacementCensus` (thread-local, RAII) records each guarded join: equal,
differs (with the registers on each side) or absent (the tree has no such region). `shader_inspect
--cfg-placement` prints it after any recompile (`--stage`, `--synthetic-table`, `--inputs`);
`test_ir_cfg_state_placement` runs it over the committed fixtures, drives the guard directly with
a differing placement, and uses `withhold` mode to show that taking an equal placement keeps the
module.

## EXEC values (step 7a)

`build_exec_values` walks a `GuestCfg` and gives every EXEC definition a value: `Entry`; a `Write`
at each instruction that writes EXEC (`rdna2_instruction_may_change_exec` or a scalar write
covering s126/s127, through `for_each_scalar_write`); a `Phi` at a block whose predecessors carry
different values; and, with `dispatch_reloads` on a service-split graph, a `DispatchReload` at every
block entry whose `same_as` is the value reaching that case. A copy into EXEC (`s_mov_b64`,
`s_wqm_b64`) is a `Write` whose `same_as` is the copied value. Lane masks in VCC and SGPR pairs are
values too (a compare's result, a 64-bit mask operation's), because they become EXEC through a copy;
a SAVEEXEC's destination holds the old EXEC value itself. The step-6 placement keeps EXEC out of
its sets (the emitter phis it from `RegState`), so the phis here come from the same rule over the
same graph: a phi where reaching definitions differ, which at a structured if merge or loop header
is where the emitter phis EXEC.

**The relations are the emitter's recognisers, stated over values:**

| Emitter | Relation | Where it reaches further |
| --- | --- | --- |
| `write_vopc_mask` (V_CMPX) | EXEC' within EXEC | - |
| `exec_masked_predicate` (any other compare) | the mask within EXEC | also when EXEC is not narrowed (the emitter returns the compare unmasked then) |
| `record_exec_mask_subset` from `s_and/s_or/s_andn2_b64 exec` and `s_and/s_or/s_andn2/s_andn1_saveexec_b64` | AND within the operand that was the previous EXEC (else the left one), ANDN2 within its left operand, OR within the wider operand when one is within the other; ANDN2 is the left operand's complement of the right one when the right one is within it | - |
| `merge_exec_arm` | the phi within an incoming value every other incoming value is within | at every join, not only one-armed if merges |
| `make_narrowing_loop_exec_phi` / `LoopExecSubset` | the header phi within its forward value, assumed and then checked at the back edges; a failed check refuses that header | at every loop, not only the admitted ones (see `CONFIDENCE` in `exec_values.cpp`) |

The 32-bit forms record nothing in the emitter and nothing here. `within` follows `same_as` and
`subset_of`, so a copy or a reload proves what its source proves.

**The census.** `ExecValueCensus` (thread-local, RAII) is fed by hooks that only observe: the
structured route's program from `place_state`, the dispatcher's from `adopt_guest_cfg_blocks`
(`dispatch_reloads`), the SPIR-V ids in EXEC, VCC and the saved masks around every `emit_alu`
(`ExecCensusObservation`), and every `exec_within` call and `exec_complement_in` read in
`rdna2_vector_word_marks.cpp`. Each id maps to the static values it was seen holding; a comparison
is `equal`, `static-stronger` (only the static relation proves it), `static-weaker` (only the SPIR-V
one does: a defect here), `unmapped` (an id no hook saw) or `cross-context`. `recompile_coverage`'s
linear walk suspends it (`ExecCensusSuspend`). `shader_inspect --cfg-exec-values` prints the
weaker cases, the stronger proof shapes with one example each, the unmapped sides and a
`cfg-exec-values-end` summary; `test_ir_cfg_exec_values` runs it over the committed fixtures and
over table-backed programs that reach each recorder, and fails on any static-weaker comparison.


## Guarded EXEC consumer (step 7b, part 1)

`ExecRelationScope` owns the mapping for one `emit_body` or dispatcher emission. The existing
program and instruction hooks feed it independently of the optional shadow census. Nested routes
restore their caller's map; a new program clears the map, and conflicting observations permanently
invalidate that id for the program. Unmapped or foreign-builder observations grant no answer.

`exec_within` and both complement reads take a static answer only when it equals the id-based
answer. A complement must name the same two SPIR-V endpoints, not just report a relation present.
Every disagreement retains the existing answer, so this stage cannot extend loop induction beyond
admitted loops or strengthen the MED-confidence WQM-as-copy model. Chain relations retain their
existing behavior. `ExecRelationCapture` observes equal/differing/unmapped answers; its withhold
and corrupt modes exercise the fallback through real recompiles without an environment switch.
