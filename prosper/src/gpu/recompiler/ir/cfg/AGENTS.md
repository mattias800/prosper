# `gpu/recompiler/ir/cfg` — the guest CFG and its region tree

Steps 1, 2 and 4 of `docs/architecture/IR_CONTROL_FLOW_PLAN.md`. One control-flow graph built once
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
existing owners. A consumer switches over after its own shadow check agrees.

| path | role |
| --- | --- |
| `guest_cfg.hpp/.cpp` | `GuestBlock`, `GuestCfg`, `build_guest_cfg`: blocks, successors, preds, rpo, idom, back edges, reachability |
| `region_tree.hpp/.cpp` | `Region`, `RegionTree`, `RegionPolicy`, `build_region_tree`; the walk, `Seq`, `Block`, `IfThen`, `IfElse`, `Unstructured` |
| `region_loops.cpp` | `Loop` and `CountedLoop`: finding the loops in the CFG and admitting them |
| `region_builder.hpp` | the builder class the two `.cpp` files share; internal, not an interface |
| `structured_if_cursor.hpp/.cpp` | guarded tree preorder for `emit_structured`, with whole-program legacy fallback |
| `region_shadow.hpp/.cpp` | the pc-level comparison with the three detectors, shared by the test and the census |
| `structured_loop_cursor.hpp/.cpp` | step 4b: the DivLoop cursor and the CountedLoop record `emit_body` consumes, from the tree when equal to the detector |

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

`SplitPolicy::split_at_services` is the hook for plan step 5: the dispatcher also ends a block after
each synchronized cross-lane service and splits around `s_trap`. Only `false` is implemented;
`true` returns an empty graph.

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
