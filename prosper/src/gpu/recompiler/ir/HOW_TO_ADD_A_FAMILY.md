# How to move an instruction family onto the IR

This is the recipe the scalar ALU (SOP1, SOP2, SOPC, SOPK) followed in ADR 0012's first step. Follow
it in order. Each step says what the scalar-ALU migration actually did, so you can compare.

The acceptance bar for a family has three parts:

1. The corpus gate reports **byte-identical SPIR-V**, or every difference is explained and
   execution-tested (see "Byte-identical, or explained" below).
2. The family's cases are **deleted from `rdna2_emit_alu.cpp`** (or `rdna2_emit_cfg.cpp`), and that
   file's ratchet row is lowered.
3. New files obey the **size rules**: at most 1,000 lines per file (CI enforces it), aiming at 500,
   and functions aiming at 80 lines.

## 0. Set up

Work in your own worktree. Build two copies of `shader_inspect`: a BASE build from the commit you
branched from, and a CANDIDATE build from your tree. Build the base first and never rebuild it.

```bash
git worktree add .claude/worktrees/<slug> -b <branch> origin/main
# base: configure /tmp/prosper-build-<slug>-base from the untouched tree, build shader_inspect only
# candidate: /tmp/prosper-build-<slug>, build shader_inspect and the tests you touch
```

Builds run in the `ps5ys` distrobox with `TMPDIR` inside the build directory. `prosper/CMakeLists.txt`
globs `src/*.cpp`, so new files under `ir/` need no source-list edit; new tests do (step 5).

## 1. Read the family's cases

```bash
grep -n 'case Rdna2Format::' prosper/src/gpu/recompiler/rdna2_emit_alu.cpp
```

For each case, list its top-level statements: lines indented 12 spaces that are not comments or a
closing brace. In the scalar ALU almost every one was an `if (shape) { ...; return true; }`, followed
by a plain-data tail ending in a `switch (in.opcode)`.

Then list every name the case body reads from `emit_alu`'s scope. The scalar cases needed only
`b`, `rs`, `in`, `ok`, `allow_wave` and the `val` lambda. VALU cases will also need `vreg`, `vcc`,
`write_vop3b_carry_output`, `allow_exec_update` and others: grep the body for each local `emit_alu`
declares before its `switch`.

**Grep the whole case body, not one case.** The scalar migration first assumed `allow_wave` was
SOPC-only, and the build failed because a SOP1 route read it too.

## 2. Relocation commit (no behaviour change)

Move the case bodies out of `emit_alu` with `tools/refactor/extract_case_routes.py`. Never retype them.

1. Write a plan: one line per top-level statement run, `<first line> <route name> <output file>`.
   Put the plan line on the comment block above an `if`, so the comment moves with it. Use `-` as
   the route name for text that must not become a route: the plain-data tail, case-scope lambdas,
   and case-scope `const bool` predicates. Point those at a scratch file outside the repository.
2. Run `--dry-run` first. It checks that the plan tiles the body, that dedenting is reversible, that
   every route has balanced brackets, and that no route has a top-level `break`/`continue`. Then run
   it for real with `--replace 'return ir::lower_<fmt>(b, rs, in, ok, allow_wave);'`. Go bottom-up
   through the file, so the earlier line numbers stay valid.
3. Declare one class per format in `lower/<family>_lowering_internal.hpp`, deriving from a context
   struct whose members are `emit_alu`'s locals **under the same names**. Then the moved text
   compiles unchanged. Declare each route as `std::optional<bool> route_x();`; the tool prints the
   declarations and the dispatch lines.
4. Convert each case-scope lambda into a member function by hand: an explicit return type, `const`
   when it only reads, and `}` in place of `};`. A case-scope `const bool` becomes a member that
   `lower()` computes at the point where the old chain computed it.
5. `lower()` calls the routes in the old order (`if (auto routed = route_x()) return *routed;`), then
   `lower_data()`, which holds the tail verbatim.
6. Build the candidate and run the corpus gate (step 6). It must report `changes: none`. Commit the
   relocation **alone**. Before you push the behaviour commit, run the FULL ctest with
   `--no-tests=error` as well. The corpus is not every input the recompiler has.

The scalar relocation took `rdna2_emit_alu.cpp` from 9,398 to 6,849 lines and reported
`changes: none`, with 1,449 accepted modules byte-identical.

## 3. Definitions: one per guest operation

In `ops/<family>.cpp`, write each operation once, as a function that appends IR to a `Builder`:

```cpp
Out add_i32(Builder& ir, const In& x) {
    const Value d = ir.iadd(x.s0, x.s1);
    const Value o = ir.and_(ir.not_(ir.xor_(x.s0, x.s1)), ir.xor_(x.s0, d));
    return {d, ir.ine(ir.lshr(o, ir.c32(31)), ir.c32(0))};
}
```

Then register it in the family's table, keyed by `(format, opcode)` and with its operand shape.
`eval_scalar_op` (the folder) and `emit_scalar_op` (the lowering) both run this one function, so they
cannot disagree.

Byte identity depends on matching the old emitter exactly:

- **Append in the old order.** If the old code computed SCC before D, as `s_min`/`s_max` did, the
  definition appends SCC first.
- **Keep the old expression nesting.** `ir.and_(ir.not_(...), ir.xor_(...))` for
  `b.ibin(Op_BitwiseAnd, b.iun(Op_Not, ...), b.ibin(...))`. The compiler evaluates nested arguments in
  its own order, and an identical nesting gives an identical order. That order decides which constant
  ids are allocated first.
- **Keep `val(...)` in the same syntactic position.** `f(val(src0), val(src1))` evaluates in argument
  order, while `uint32_t a = val(src0), c = val(src1);` evaluates left to right. The scalar migration
  replaced `logical_u32(val(...), val(...))` with `emit_data_op(val(...), val(...))` for this reason.
- **Keep the lowering's side effects where they were.** That covers map insertions
  (`uint32_t& d = rs.sreg[dst]`), `retire_*` calls, descriptor-tag erasure, and the `!rs.scc` and
  `ok` checks.

Use the instruction-effects table (`rdna2_instruction_effects.hpp`) for facts about operations,
never a private opcode list. For example, `rdna2_reads_scc_as_data` decides whether a poisoned SCC
refuses the instruction.

Delete every duplicate of the semantics you move: the old `switch`, helper lambdas that recomputed an
op (`logical_u32` and `bitfield_mask_u32` in SOP2), and the folder's copy in
`rdna2_recompile_shared.hpp`. The folder reaches definitions through `shader_constant_scalar_op`.
Mark `Fold::Yes` only on the definitions the old folder already folded, so the folder's domain stays
the same.

If a family needs a new IR opcode, add it in all five places: `Opcode` in `ir.hpp`, `kInfo` in
`ir.cpp`, `eval.cpp`, `emit/spirv_emitter.cpp` and `builder.hpp`.

## 4. Lowering: semantics stay in `ops/`, bookkeeping in `lower/`

`lower_data()` reads the operands, checks the refusals, calls `emit_<family>_op`, and then writes the
results where the old code wrote them: `rs.sreg[dst]`, `rs.scc`, erasing `rs.sreg_srt`. Routes keep
calling the existing mark, proof and relay code from where they already call it. Do not migrate that
machinery in a family PR.

A route whose value part is a plain operation should compute that value through the definition, as
`route_pack_halfwords` and the two EXEC_LO data routes now do, while keeping the rest of its body. If
the old code checked `ok` between the value and its SCC, the IR emits the SCC compare before that
check. The output differs only on a path the stage then refuses, and the corpus gate confirms that.

## 5. Tests

Put new tests in `tests/gpu/recompiler/ir/` and add each one to the `foreach(ir_test ...)` list in
`prosper/CMakeLists.txt`.

- **One focused test per lowering file** (`test_ir_lower_<fmt>.cpp`, using `lowering_fixture.hpp`).
  Pin what the lowering does: which registers and SCC it writes, the tag it retires, and every
  refusal (`ok == false`), such as a poisoned SCC, an EXEC destination, or an opcode with no
  definition. Assert SIMM16 or operand extension through `constant_cached()`.
- **The execution test** (`test_ir_scalar_alu_execution.cpp`). For every definition it builds a real
  RDNA2 program, compiles it with `recompile_compute`, runs it in the integer SPIR-V interpreter
  (`tests/gpu/recompiler/bpermute_spirv_oracle.hpp`), and compares every stored D and SCC against
  `eval_scalar_op` over edge and random operands. It also fails if any IR opcode is unused by every
  definition, and if a definition's SCC write or D width disagrees with the effects table.
  - If your definitions emit a SPIR-V instruction the interpreter lacks, add it to
    `Interpreter::scalar_alu_op` from the SPIR-V specification, and fail on undefined inputs.
  - A VALU family needs a per-lane variant: VGPR inputs that differ per invocation, with more than one
    lane.
- **Folder tests** (`test_ir_scalar_fold.cpp`). Use expected values **worked by hand from the ISA**,
  not taken from `eval`. The execution test proves that eval and SPIR-V agree. Only hand-worked
  values catch a definition that is wrong in both.
- **Mutation check before you push.** Break one eval case, one emit mapping and one operand wiring
  in the folder, and confirm a test reddens for each. Restore the file by writing its content back
  and `touch`ing it, because a preserved mtime makes ninja skip the rebuild. In the scalar
  migration, these went red: an IAdd eval change, an SMin->UMin emit change, a BfeU count clamp,
  swapped fold operands, and the SIMM16 zero-extension.

## 6. The corpus gate

```bash
python3 prosper/tools/shader_inspect/recompile_corpus.py compare ~/shader-corpus \
    --base /tmp/prosper-build-<slug>-base/shader_inspect \
    --candidate /tmp/prosper-build-<slug>/shader_inspect \
    --out-dir ~/<slug>-corpus-compare --jobs 8
```

It takes a few minutes for about 4,000 entries. Read the last lines:

- `changes: none` means no program changed verdict or SPIR-V. Exit status 0.
- `accepted by both, SPIR-V: N compared` gives N, the number of modules compared byte for byte. It
  was 1,449 here; quote it.
- `changes: output-changed K` (exit status 1) means K modules differ. `flips.json` in the out-dir
  lists them by program, under `groups` and `rows`.

**Run a positive control once per family.** Make a byte-changing but semantically neutral edit to one
definition, rebuild, and confirm the gate reports `output-changed`. Swapping `s_add_u32`'s IAdd
operands gave `output-changed 24`. Without that control, a clean result could mean the corpus never
reaches your code.

## 7. Size rules and the ratchet

```bash
python3 prosper/tools/ci/check_arch_ratchet.py --root . --base origin/main
python3 prosper/tools/ci/check_arch_ratchet.py --root . --update   # lowers the emit_alu row
```

- `dir-file-cap` fails any file under `gpu/recompiler/ir/` over 1,000 lines. Split by role: one file
  per family or sub-family, with routes grouped by theme at around 300-400 lines each.
- `host-throw` fails a `throw` in recompiler code. Use a loud abort with a message for a lowering bug
  (`ir.cpp` `malformed`).
- `recompiler-include` fails an include of `gpu/diagnostics` or `gpu/execute`. Do not copy
  `emit_alu`'s include list wholesale.
- A route moved verbatim may exceed the 80-line function target. Leave it whole in the relocation
  commit, and split it in a separate behaviour-neutral commit gated by the corpus.

## Vector families: what changes

The vector ALU (VOP2 integer/bitwise group) followed the same steps, with four differences.

**1. A VALU case is a pipeline, not a route chain.** VOP2's body was a fixed sequence of stages
shared by every opcode:

1. DPP routes that own the whole instruction.
2. SRC0's DPP transform.
3. SDWA selects or float abs/neg.
4. The `switch` that computes D.
5. SDWA destination placement.
6. OMOD/CLAMP.
7. The EXEC-predicated VDST write (`predicate_write`).

It was relocated with the same tool and plan format. Name each segment `route_*` if it can own the
instruction, or `stage_*` if it always falls through. `lower()` calls them in the old order
(`lower/vop2.cpp`).

The case-scope locals become members, with these exceptions:

- The two emit_alu reference aliases become reference members bound in the constructor:
  `vreg` (`decltype(RegState::vreg)&`) and `vcc` (`uint32_t&`).
- `uint32_t& d = vreg[in.dst.value];` stays a local in `lower()`, created where the old code created
  it, and is passed to the stages that use it (`compute(uint32_t& d)` etc.). Do not bind it in the
  constructor: `vreg[...]` inserts a map entry, and `vreg_old()` runs before it.
- A helper emit_alu kept in its anonymous namespace and that both files now need moves to a header
  (`inline_float_f16_bits` went to `rdna2_alu_support.hpp`, made `inline`).

The shared context is `lower/lowering_context.hpp`. Vector classes live in
`lower/vector_lowering_internal.hpp`, and emit_alu calls `ir::lower_vop2(...)`.

**2. Only the operation goes through the IR.** A vector definition (`ops/vector_alu.hpp`,
`VectorOpDef`) is `D = f(S0, S1)` for one lane. The stages around it are the same for every opcode
of the format, so they stay in `lower/`. The `compute()` stage takes the definition first and falls
back to the remaining `switch` cases:

```cpp
if (const VectorOpDef* def = find_vector_op(in.fmt, in.opcode)) {
    d = emit_vector_op(b, *def, a, c);
    return std::nullopt;
}
```

Then delete the migrated `case` labels. Keep each operand ORDER exactly: the *rev* ops take the
shift amount or the subtrahend from S0.

**3. The execution test is per lane.** Call `tests/gpu/recompiler/ir/vector_lane_harness.hpp`:

```cpp
const LaneInputs in = lane_inputs(seed);   // 64 lanes, own S0/S1/old VDST each, ~1/4 inactive
const LaneRun run = run_vector_op_per_lane({vop2(def.opcode)}, in, wave);   // wave 64 and 32
// each lane: in.active[l] ? eval_vector_op(def, in.s0[l], in.s1[l]) : in.old_d[l]
```

The harness loads four per-lane arrays from one buffer, sets EXEC with `v_cmpx_ne_u32 0, v4`, runs
your instruction words (written as `op v1, v2, v3`), restores EXEC and stores v1 per lane. For
another encoding, pass its words: a VOP1 is `op v1, v2`, and a VOP3 is its two words naming the same
registers. Copy `test_ir_vector_alu_execution.cpp`, including its
`HarnessSeesPerLaneOperandsAndExec` arm. Each of these mutations turned the test red:

- swapping the UMax emitter mapping;
- dropping AShr's sign in eval;
- deleting `predicate_write`;
- deleting the harness's `v_cmpx`.

The lowering test (`test_ir_lower_vop2.cpp`) pins the stage behaviour: a defined op emits exactly
its instruction under full EXEC, a narrowed EXEC adds the predicated write, and undefined opcodes
still take the switch.

**4. The positive control is louder.** Swapping `v_add_nc_u32`'s operands changed 754 modules
(752 by replay, 2 by stage). VALU ops are everywhere, so a vector family's gate is a strong check.

## Family list and order

Remaining in `rdna2_emit_alu.cpp`, in the order to migrate them. Codex takes the open groups; the
reserved ones stay with the lead. **The reservation rule:** anything that reads or writes a wave mask
(EXEC, VCC, SGPR masks), anything that moves data between lanes, anything that touches definedness
marks or proof routes, and control flow. When a group mixes the two, migrate only its open opcodes
and leave the reserved ones in the `switch`.

| order | group | who |
| --- | --- | --- |
| 1 | VOP2 float: `v_add/sub/subrev/mul_f32`, `v_min/max_f32` (NMin/NMax), `v_mac/fmac_f32`, `v_madmk/madak/fmamk/fmaak` | Codex |
| 2 | VOP2 f16: `v_add/sub/mul/max/min_f16`, `v_pk_fmac_f16`, `v_cvt_pkrtz_f16_f32` | Codex |
| 3 | VOP1 integer and bit ops (`v_not`, `v_bfrev`, `v_ffbh/ffbl`, ...), then float unary (`rcp/sqrt/exp/log/sin/cos/fract/trunc/...`), then conversions | Codex |
| 4 | VOP3-only integer ops (`v_bfe`, `v_bfi`, `v_lshl_add`, `v_add3`, `v_mad_u32_u24`, `v_alignbit`, `v_perm`, `v_sad*`, `v_min3/max3/med3`), then VOP3 float (`v_fma`, `v_mad`, `v_ldexp`, `v_div_fixup`, ...) | Codex |
| 5 | VOP3 (e64) encodings of VOP2/VOP1 ops that emit_alu does not admit today | Codex, but ADMITTING them is a behaviour change: corpus-explained, separate PR |
| 6 | VOP3P packed 16-bit ops | Codex |
| 7 | VINTRP | Codex, with the lead reviewing: it reads M0 |
| - | VOP2 `v_cndmask_b32`, carry ops `v_add/sub/subrev_co_ci_u32` (VCC), every VOPC (writes VCC/EXEC), VOP3B (`v_add_co_u32`, `v_div_scale`, `v_mad_u64_u32`: SGPR carry masks), `v_mbcnt` | lead (wave masks) |
| - | DPP (the VOP1/VOP2 DPP routes and SRC0 stage), `v_permlane*`, `v_readlane/writelane/readfirstlane`, `ds_bpermute/permute`, quad and swizzle ops | lead (cross-lane) |
| - | `v_movrel*`, `s_movrel*`, and the memory families SMEM, MUBUF/MTBUF, FLAT, MIMG, DS (descriptor provenance, raw windows, active-word marks) | lead (proof and definedness) |
| - | SOPP, `rdna2_emit_cfg.cpp`, the SOP `*_routes.cpp` | lead (control flow, mask routes) |

## Byte-identical, or explained

"Byte-identical" means the gate's verdict for every program and the sha256 of every accepted module
are unchanged.

"Explained" means that for each changed module you can name the instruction whose emission changed
and why, and you have an execution test that shows the new module computes the same result as the
old one (or the correct one, if the old one was wrong). A changed fold that specialises a new branch
counts as a change and needs the same treatment.

The corpus is not the only witness, though. The scalar migration first let the folder evaluate every
definition, and the gate still said `changes: none`. Two existing ctest cases went red all the same:
`ScalarPairMask.Wave64VccLoAndReadingAOnePathWordStillRefuses` and `rdna2_spirv_struct`'s GTA FFBH
arm. Folding `s_bitcmp1_b32` and `s_movk_i32` proved branches that those fixtures need to stay
runtime. So each definition carries `Fold::Yes` only where the pre-IR folder already folded that
operation, and widening that domain is a separate decision.

## When to stop and report instead of changing tests

Stop, leave the tests as they are, and report to whoever dispatched you when:

- the corpus gate shows a change you cannot attribute to a specific instruction;
- an existing test fails and the fix would mean editing its assertion or expected value;
- matching the old bytes would require a definition that is wrong by the ISA. Report the ISA
  disagreement with the encoding and the `llvm-mc` check rather than encoding the bug;
- a route's old behaviour looks wrong. Moving it is in scope; fixing it is a separate,
  corpus-gated PR;
- the interpreter would need an operation whose SPIR-V semantics you cannot cite;
- the family touches mark, proof or relay state in a way that cannot stay a call from the lowering;
- you want the folder to evaluate more operations than it did before. That is a behaviour change with
  its own tests to revisit (see "Byte-identical, or explained").
