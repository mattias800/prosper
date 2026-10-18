# `gpu/recompiler/ir/cfg` — the guest CFG and its region tree

Step 1 of `docs/architecture/IR_CONTROL_FLOW_PLAN.md`. One control-flow graph built once from the
decoded program (`guest_cfg`), and the structured regions derived from it (`region_tree`). Both
are plain data: building them allocates no SPIR-V id and calls no `SpirvCompute`, so they cannot
change an emitted byte. **Nothing in emission reads them yet.** Their only user is the shadow
check (`region_shadow`), which compares the region tree with `detect_forward_ifs` in a gtest over
the committed fixtures and in `shader_inspect --cfg-regions` over the corpus. A consumer switches
over in a later step, after its own shadow check agrees.

| path | role |
| --- | --- |
| `guest_cfg.hpp/.cpp` | `GuestBlock`, `GuestCfg`, `build_guest_cfg`: blocks, successors, preds, rpo, idom, back edges, reachability |
| `region_tree.hpp/.cpp` | `Region`, `RegionTree`, `RegionPolicy`, `build_region_tree`: `Seq`, `Block`, `IfThen`, `IfElse`, `Unstructured` |
| `region_shadow.hpp/.cpp` | the pc-level comparison with `detect_forward_ifs`, shared by the test and the census |

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

The rules are `detect_forward_ifs`' own (`../../rdna2_cfg_support.hpp`). The tree calls the same
predicates, never a copy: `classify_forward_if_branch`, `resolve_forward_if_target`,
`compute_execz_region_admissible`, `wave_vcc_region_admissible`, `forward_if_workgroup_uniform`.
A rule change goes there, and both structurers follow it.

| Rule | Graphics (`allow_vcc`) | Compute (`compute_wave_branches`) |
| --- | --- | --- |
| `s_cbranch_scc0/1` | an if, unless in `linearized` (kill-mask tests, waterfalls) | an if, unless in `linearized` |
| `s_cbranch_vccz/vccnz` | an if (a vote) | an if decided per guest wave (`wave_vote`) |
| `s_cbranch_execz` | an if, unless in `linearized` (`safe_execz_branches`) | an if (a subgroup vote), unless in `linearized` |
| `s_cbranch_execnz` | refused (only a loop back edge, step 4) | refused |
| Neither flag set | VCC and EXECZ branches are refused | - |
| Target | forward and at most `end_pc`; past `s_endpgm` only when the code there ends at once (an early-out, `merge = kNone`) | same |
| Else | the block before the target ends in a forward `s_branch` whose target lies after the if's target and at or before `end_pc` | same |
| EXECZ region | - | no `s_barrier`; a scalar write to `vcc_lo/hi` must be dead at the merge |
| VCC region | - | unless proved workgroup-uniform (#1554): no `s_barrier`, `ds_swizzle`/`ds_append`/`ds_consume`/`ds_bpermute` or `v_mbcnt_lo/hi` (`wave_vcc_region_admissible`) |
| Workgroup uniformity | - | VCC (#1554) and SCC (#3713) proofs set `uniform_workgroup` |

Structure: each arm is built over its own block range, so a branch from one arm into another, a
plain if leaving its range, an unclaimed `s_branch`, a call, or a back edge refuses. An if/else
whose merge lies past its enclosing range is accepted as the detector accepts it (the
shared-outer-merge cascade) and recorded with `end_block < merge`. **Any refusal makes the whole
program one `Unstructured` region**, which is today's behaviour: the detector refuses the program
and the whole-program dispatcher takes it. A local `Unstructured` inside structured code is plan
step 11.

Known differences from the detector, both in the refusing direction (the tree refuses, the detector
admits): an if in an if/else's then-arm whose target lies in the else-arm (the detector bounds it by
the merge, not the arm), and a target or merge outside the decoded stream when the stream has no
`s_endpgm` (the detector's bounds are then vacuous). The first is pinned by
`test_ir_cfg_region_shadow`; the second is one truncated corpus dump.
