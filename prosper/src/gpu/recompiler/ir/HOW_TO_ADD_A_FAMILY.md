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

**If the positive control reports `changes: none`, the corpus does not reach your family.** That
happened with the VOP1 f16 unary ops (#5233), and it makes the gate vacuous for that PR. Build a
synthetic witness corpus:

1. Write small RDNA2 programs that use every op in the family, in each encoding form the lowering
   handles (plain, SDWA, both destination halves, and so on). Reuse the per-lane harness's program
   builder.
2. Save them as sidecars in a scratch directory such as `~/<slug>-witness/`. Never add them to
   `~/shader-corpus`.
3. Run the same `compare` over that directory, base build against candidate. It must report
   `changes: none` with every program accepted.
4. Repeat the positive control on the witness corpus. It must now report changes. Revert it and
   re-run clean.
5. Put the witness command, the program count and both reports in the PR body. Commit the generator
   if it is small; otherwise describe it.

## 7. Size rules and the ratchet

```bash
python3 prosper/tools/ci/check_arch_ratchet.py --root . --base origin/main
python3 prosper/tools/ci/check_arch_ratchet.py --root . --update   # lowers the emit_alu row
```

- `dir-file-cap` fails any file under `gpu/recompiler/ir/` over 1,000 lines. Split by role: one file
  per family or sub-family, with routes grouped by theme at around 300-400 lines each.
- Every family adds to the same core files: `ir.hpp`'s `Opcode`, `ir.cpp`, `eval.cpp`,
  `emit/spirv_emitter.cpp` and `builder.hpp`. Do not let them grow towards the cap. When `eval.cpp` or
  `spirv_emitter.cpp` passes about 400 lines, split it by category (integer, f32, f16, and so on)
  into `eval_<category>.cpp` / `emit/<category>.cpp`, in a behaviour-neutral commit gated by the
  corpus. Order the `Opcode` enum by category too, so concurrent families conflict less.
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

## Mask-writing compares (VOPC)

VOPC is already out of `rdna2_emit_alu.cpp`, and the F32 group (`v_cmp_*_f32` and
`v_cmpx_*_f32`, both encodings) already goes through the IR. The remaining groups move by adding
definitions; nobody relocates anything again. A compare is two halves, and only the first is yours:

1. **The value.** The lane's test result, a Bool, as a function of its two source words. It is
   different for every opcode, and it is what you migrate.
2. **The mask write.** What every compare then does with that Bool: a V_CMPX narrows EXEC, any
   other compare writes EXEC & value to VCC or to its SGPR destination, and the mask-domain
   bookkeeping and proof facts go with it. It is the same for all 193 compares, and it already
   lives in one function, `write_vopc_mask` (`lower/vopc_mask_write.cpp`).

Where things are:

| file | holds |
| --- | --- |
| `ops/vector_compare.hpp`, `ops/vector_compare.cpp` | `VectorCompareDef {opcode, build}`, the group list, `find_vector_compare`, `eval_vector_compare`, `emit_vector_compare` |
| `ops/vector_compare_f32.cpp` | the worked example: 16 definitions, one per V_CMP_*_F32 opcode |
| `ops/vector_compare_i64_u64.cpp` | the 64-bit worked example: `pair_sources` definitions that read both halves of each source |
| `lower/vopc.cpp` | `VopcLowering::lower()`: source reads, admission and refusals, source modifiers, SDWA selects, 64-bit halves, the value overrides, then the value and the mask write |
| `lower/vopc_compare_values.cpp` | `compare_value()`: the value overrides, then a definition if one exists, then the remaining `switch`, one function per not-yet-migrated group |
| `lower/vopc_mask_write.cpp` | `write_vopc_mask`, the shared mask write |
| `tests/gpu/recompiler/ir/vector_compare_harness.hpp` | the per-lane MASK harness and the witness exporter |

### What goes in `ops/`

One file per group, `ops/vector_compare_<group>.cpp`, holding a table of `VectorCompareDef`. Each
`build` appends the relation and returns a **Bool**. Then:

- declare `vector_compare_<group>_defs()` in `ops/vector_compare.hpp`;
- add it to `vector_compare_groups()` in `ops/vector_compare.cpp`.

A definition is keyed by the **V_CMP_*** opcode. The V_CMPX_* twin (opcode + 0x10, see
`vopc_is_cmpx`) and the VOP3 (e64) encodings decode as VOPC with the same number, and the lowering
looks them up by the compare opcode (`eff`), so one definition serves all four forms. Never add a
CMPX row.

A definition receives the sources **after** the lowering has prepared them. Preparing a source is
operand bookkeeping, not the relation, so it stays in `lower/`:

- the float ABS/NEG modifiers (`source_modifiers`);
- the integer SDWA byte and word selects (`integer_sdwa_sources`);
- the f16 unpack, including inline-constant encodings and the SDWA WORD_1 select (`compare_f16`'s
  `f16v`);
- the 64-bit source halves (`integer64_halves`);
- for `v_cmp_class_f32`, the integer-domain ABS/NEG of the RAW S0 word (`class_raw`). CLASS reads
  `ra`/`rc`, not the float-modified `a`/`c`, so its lowering passes `class_raw` and `rc` to the
  definition. `source_modifiers` still runs first and its unused output stays, for byte identity.

What belongs in the definition is everything after that: for the 16-bit integer group, the
`bfe_s`/`bfe_u` extension of bits [15:0] is part of the relation, so it moves into the definition.

Match the old bytes the same way every family does: append in the old order, with the old nesting.
`cmp_o` appends both `F32Eq`s before the `BoolAnd`, as the old `case 0x07` did.

### NaN and signed zero: what the F32 group pins

AMD 70648 12.9, Table 61. These rules apply to every float group (F16 and F64 too):

- LT, EQ, LE, GT, LG and GE are **ordered**: false when either source is a NaN.
- Each N-form is the plain negation of its ordered relation (`NGE = !(S0 >= S1)`), so it is
  **unordered**: true when either source is a NaN. The ISA says so in a note on each N-form ("With
  NAN inputs this is not the same operation as <").
- LG is ordered not-equal (`F32Lg`, `OpFOrdNotEqual`). NEQ is `!(S0 == S1)`, unordered
  (`F32UnordNe`). They differ only on a NaN.
- O is "neither is a NaN" and U is "either is a NaN". F is constant false and TRU constant true.
- +0 equals -0, and each infinity is the extreme of its sign.

The IR has all twelve binary32 relations: `F32Lt/Eq/Le/Gt/Lg/Ge` and
`F32UnordLt/Eq/Le/Gt/Ne/Ge`. The F16 group compares its unpacked values with the same opcodes,
because f16 embeds exactly in f32. The integer groups use `IEq/INe/ULt.../SGe`.

### What you must never touch

Each of these is either mask or proof state, or an override of the value that a proof pass reads.
Changing any of them is a lead change, even when it looks mechanical:

- `lower/vopc_mask_write.cpp`, and with it `exec_masked_predicate`, `exec_subset_of`, `sreg_bool*`
  and `vcc_wave_uniform`;
- `VopcLowering::lower()`'s call order;
- `compare_wave_uniform`, `scalar_data_operand` and `scalar_integer64_operand`;
- the value overrides `zero_compare` and `magnitude_compare`, and `record_guest_compare_facts`.
  `compare_value()` returns an override before it looks for a definition, so a definition never
  replaces one;
- the refusals: V_CMPX without `allow_exec_update`, integer64 compares with ABS/NEG, an inline
  float in a 64-bit integer compare, and `ok = false` for an opcode with no value.

### Steps for one group

1. Write the definitions (above). If you need an IR opcode, add it in all five places
   (`HOW_TO_ADD_A_FAMILY.md` step 3), and add its SPIR-V instruction to the interpreter
   (`tests/gpu/recompiler/bpermute_spirv_oracle.hpp`) if it is missing. The F32 group added the
   eight missing float relations to both.
2. In `lower/vopc_compare_values.cpp`, delete the group's `case` labels and its function. The
   definition lookup runs first, so they are dead once the definitions exist.
3. **Admit nothing new.** Only the opcodes the old switch handled get definitions. The switch
   handles the F16 group only at 0xC9-0xCE (LT, EQ, LE, GT, LG, GE), and it had no F64 (now
   admitted, see "F64 values"), no `class_f16` and no `v_cmp_f/t` for the integer types. Adding any of those is a behaviour change:
   do it in a separate, explained PR, as in Group 5.
4. Run the corpus gate, a witness corpus and the controls (below).

**64-bit groups.** I64 and U64 are done (`ops/vector_compare_i64_u64.cpp`), and they are the
worked example of "64-bit values" below. F64 is done too (`ops/vector_compare_f64.cpp`), as an
admission rather than a migration, and it is the worked example of "F64 values" below.

### How to test the mask

Copy `test_ir_vector_compare_execution.cpp` into `test_ir_vector_compare_<group>_execution.cpp`
and change its inputs and its hand-worked table. The harness (`vector_compare_harness.hpp`) runs
each opcode in five forms:

- VOPC e32 to VCC;
- VOP3 to an SGPR pair;
- VOP3 to VCC;
- V_CMPX in each of its two encodings.

Each form runs in Wave64 and Wave32, and reads the **mask** back rather than a VGPR. After a
compare, it restores EXEC and runs `v_cndmask_b32_e64 v1, 0, 1, <mask>`, so every lane, active or
not, stores its mask bit. An inactive lane must read **0**: `VCC[n] = EXEC[n] & test` (ISA 3.9).
After a V_CMPX, it runs `v_mov_b32 v1, 1`, so exactly the lanes where EXEC & test holds read 1, and
the rest keep their old v1.

Keep these four tests for each group:

- **Every definition, every form, both waves**, against `eval_vector_compare`.
- **A hand-worked truth table from the ISA.** Never take it from eval. For float groups, include a
  NaN on each side, +0 against -0, and the infinities. For integer groups, include the signed and
  unsigned extremes, and for 16-bit groups, the bits above [15:0].
- **The harness's own positive control.** `InactiveLanesReadZeroAndActiveLanesDiffer` fails if
  EXEC is not observed: at least one inactive lane must have a true test.
- **Effects agreement.** Each definition's name comes from `rdna2_instruction_effects`. Its V_CMPX
  twin sits at +0x10, and each new IR opcode is used.

`test_ir_lower_vopc.cpp` pins the mask write for every group: the VCC and SGPR destinations, CMPX
and its `exec_subset_of` record, the `vcc_wave_uniform` proof, and the refusals. You do not need to
extend it, and its assertions must not change.

The F32 group's mutations, each of which turned a test red:

- `F32Le` evaluated as `<`;
- `F32UnordLt` emitted as `OpFOrdLessThan`;
- LG defined as NEQ;
- O reading S0 twice;
- the mask write without `exec_masked_predicate`;
- CMPX writing the raw compare to EXEC;
- CMPX without its `exec_subset_of` record;
- the definition lookup switched off;
- the definition's sources swapped;
- the source modifiers skipped;
- the interpreter treating unordered relations as ordered.

### Gate and controls

Run the gate on the recorded corpus and on a witness corpus, exported with
`PROSPER_IR_WITNESS_DIR=~/<slug>-witness ./test_ir_vector_compare_<group>_execution
--gtest_filter='*Exports*'`, then `recompile_corpus.py harvest ~/<slug>-witness --root
~/<slug>-witness`. Run one positive control **per comparison kind**: each is a byte-changing but
neutral edit, such as a mirrored relation with swapped operands, or swapped `BoolAnd` arguments.
The F32 group's results:

| control | recorded corpus (1,448 modules) | witness (160 programs) |
| --- | --- | --- |
| ordered LT..GE mirrored | output-changed 1,325 | output-changed 60 |
| N-forms mirrored | output-changed 271 | output-changed 60 |
| O/U arguments swapped | output-changed 22 | output-changed 20 |
| F/TRU rewritten as `F32Lg(s0,s0)` / `F32UnordEq(s0,s0)` | **none**: the corpus never runs them | output-changed 20 |

So the witness corpus is required. The recorded corpus alone could not show that the constant
compares still work.

## Mask-reading and mask-writing VALU

`v_cndmask_b32` and the carry ops are done, and they are the worked example. They read a lane mask
(the cndmask selector, a carry-in), write one (a carry-out), or both. They split the way the VOPC
compares do, into a per-lane **value** that is yours and the **mask** around it that is not.
`write_vopc_mask` could not be reused: a compare masks its result with `exec_masked_predicate`
(a `bsel`), while a carry op uses `land`. Changing either is a byte change and a poison-semantics
change, so the carry ops have their own shared write.

| file | holds |
| --- | --- |
| `ops/vector_carry.hpp`, `ops/vector_carry.cpp` | `VectorCarryDef {fmt, opcode, reads_carry_in, build}`: D and the carry Bool, from S0, S1 and the lane's carry-in **bit** |
| `ops/vector_cndmask.hpp`, `ops/vector_cndmask.cpp` | `VectorCndmaskDef`: D = bit ? S1 : S0 |
| `lower/carry_mask_write.cpp` | the shared mask write: `exec_masked_carry` (ISA 3.9), `write_vop2_carry_mask` (VCC; in Wave32 the VCC_LO B32 word, #5348), `write_vop3b_carry_mask` (SDST: VCC, an SGPR pair, or one Wave32 B32 word; emit_alu's old `write_vop3b_carry_output`) and `write_mad64_carry_mask` (v_mad_u64_u32 / v_mad_i64_i32, emit_alu's old write) |
| `lower/vop3_carry.cpp`, `lower/vop3_cndmask.cpp` | the VOP3 routes: the mask READ (verbatim), then the definition, then the write |
| `lower/vop2_mask.cpp` | the VOP2 forms: VCC is the selector, the carry-in and the carry-out |
| `tests/gpu/recompiler/ir/vector_mask_harness.hpp` | the per-lane harness and the witness exporter |

### The rules this example follows

- **A definition never sees a mask register.** It gets the lane's bit as a `Type::Bool` input. The
  mask read stays in the lowering, verbatim, because each read carries its own rules. VCC_HI never
  falls back to `rs.vcc`. A tracked B32 lifetime is preferred over `rs.vcc`. The NGG one-lane
  projection reads scalar DATA as a mask. An untracked source refuses (`ok = false`). None of that is
  per-opcode semantics.
- **One build function per operation, one table row per encoding.** VOP2 `0x28` and VOP3B `0x128`
  are the same `add_co_ci` function. `DefinitionsAgreeWithTheEffectsTable` checks that each VOP2 row
  has its VOP3B twin at +0x100 with the same `build`.
- **Two outputs are fine.** A carry definition returns `{d, carry}` from one Block, and the lowering
  writes D first and the mask second, as the old code did.
- **Append order.** `Select(cin, 1, 0)` comes first, then the partial sums, and the partial carries
  join as `BoolSelect(c1, true, c2)`. That is the old code's `b.bsel(c1, b.btrue(), c2)`, not a
  `BoolOr`.
- **An emit function's argument order is part of the bytes.** `emit_vector_cndmask(b, def, mask, s1,
  s0)` takes its arguments in `SpirvCompute::sel`'s order. The old `b.sel(m, fv(1), fv(0))` becomes
  `emit_vector_cndmask(b, *def, m, fv(1), fv(0))` with the same argument expressions, so the
  compiler evaluates `fv(0)` and `fv(1)` in the same order. This was measured. Hoisting `fv(1)` into
  a local first changed 2 recorded modules and both modifier witnesses. The same applies to the VOP2
  SDWA path's `word(c, ..)`, `word(a, ..)`.
- **Keep the chain position.** Each VOP3 lowering returns `false` when the opcode has no definition
  of its kind, and emit_alu calls it at the old `else if` position. `route_carry_out` takes only
  definitions without a carry-in, and `route_carry_in_out` only those with one.

### What you must never touch

These are consulted around the lowering, never from inside it. Before migrating, grep the case body
to confirm that it does not call them:

- the fresh-mask proof `refuse_mask_write` / `fresh_mask_sources_defined` (emit_alu's prologue);
- the destination-width facts `vop3b_fresh_carry_output` and `vop3_writes_mask_sdst`, which the CFG,
  Wave32 dispatch-word and fragment-loop passes read;
- `write_vop3b_carry_mask`, `write_vop2_carry_mask` and `exec_masked_carry`, and each route's mask
  read. A known defect lives there: a VOP2 carry leaves a stale Wave32 VCC_LO B32 lifetime (#5348).
  Fixing it is a separate, corpus-gated PR.

### Testing the mask

Copy `test_ir_vector_carry_execution.cpp` or `test_ir_vector_cndmask_execution.cpp`. Each program
first writes the mask the op reads from S2 (v5, `mask_lane_inputs`), then runs the op, then
optionally reads the written mask back into v1 for every lane. So one definition gets two programs:
one reads D (an inactive lane keeps its old VDST), and one reads the carry mask (an inactive lane
reads 0). Run every form in both wave sizes: VOP2, VOP3B to VCC and to an SGPR pair, and a carry-in
from VCC and from an SGPR pair.

The carry-in is produced the way guest code produces one, by an earlier carry op (`carry_from_s2`,
`v_add_co_u32 v7, <mask>, v5, -1`); the cndmask selector by `v_cmp_ne_u32_e64` (`mask_from_s2`).
Two harness traps, each found the hard way:

- **In Wave32, the VOP3B `_co_ci` forms are refused** (`fabricated-fresh-mask-source`): the proof
  reads SRC2 as the Wave64 pair the effects table charges it, and the second word is never written
  (CONFIDENCE: MED, from reading the code; not A/B-tested). The refusal comes from emit_alu's
  prologue (`refuse_mask_write`), which this migration does not touch, so it is the pre-IR verdict.
  `Wave32RefusalsStay` pins it, and the witness exporter leaves those programs out of Wave32.
- **`read_mask` reads VCC back with `v_cndmask_b32_e32`.** It was written that way because the
  e64 form read a stale VCC_LO lifetime after a VOP2 carry (#5348). That is fixed, and
  `E64CndmaskReadsTheVop2CarryNotTheStaleVccLo` covers the e64 reader. The VOP3B carry-in from
  `vcc_lo` is admitted in Wave32 once VCC_HI is defined (`s_mov_b32 vcc_hi, 0`), so
  `Vop3bCarryInFromVccLoReadsTheVop2Carry` covers that reader too.

The four tests per family: every definition in every form and both waves against eval; a
hand-worked ISA table (it must include rows that separate `<` from `<=`, such as a zero addend: the
`ULt -> ULe` mutation survived until it did); the harness's positive control (an inactive lane with
a raw carry of 1, and active lanes seeing both carry values and both carry-in values); and effects
agreement. `test_ir_lower_vector_mask.cpp` pins the write's destinations, the B32 domain, the EXEC
`land`, the mask reads and every refusal.

Mutations that went red: carry `ULt -> ULe` (after the zero-addend rows were added), the EXEC mask
dropped, the VOP3 operands swapped, the cndmask arms swapped, the B32 insert dropped, and the VOP2
carry-in ignored.

### Gate and controls

Witness: `PROSPER_IR_WITNESS_DIR=~/<slug>-witness/carry ./test_ir_vector_carry_execution
--gtest_filter='*Exports*'` (likewise `cndmask`), then `recompile_corpus.py harvest` on each. Each
control is a neutral byte change in one definition: swapped IAdd operands for add, and a mirrored
borrow `UGt(y, x)` for the subtractions.

| control | recorded corpus (1,442 modules) | carry witness (60) | cndmask witness (12) |
| --- | --- | --- | --- |
| the migration | none | none | none |
| `v_add_co_u32` | **none** | output-changed 44 (its 8, plus every carry-in producer) | none |
| `v_sub_co_u32` | **none** | output-changed 8 | none |
| `v_subrev_co_u32` | output-changed 2 | output-changed 8 | none |
| `v_add_co_ci_u32` | output-changed 10 | output-changed 12 | none |
| `v_sub_co_ci_u32` | **none** | output-changed 12 | none |
| `v_subrev_co_ci_u32` | **none** | output-changed 12 | none |
| `v_cndmask_b32` (`Select` on `BoolSelect(m, true, false)`) | output-changed 1,338 | output-changed 30 (the mask reads) | output-changed 12 |

The recorded corpus never reaches four of the seven definitions, so the witness corpus is required.

### The remaining mask-family members

| instruction | status | why |
| --- | --- | --- |
| `v_div_scale_f32` (VOP3B 0x16D) | Codex with this recipe, as an **admission** in its own explained PR | emit_alu has no case for it today. The value is an F32 definition, and the flag goes through `write_vop3b_carry_mask`. |
| `v_div_scale_f64` (0x16E) | Codex, after `v_div_scale_f32`, using the F64 recipe | an F64 result with a flag; it needs F64 arithmetic first (family 14) and the result-mode rules in "F64 values" |
| `v_mad_u64_u32` / `v_mad_i64_i32` (0x176 / 0x177) | values in `ops/vector_mad64.cpp`; carry write remains **lead** | The existing 32-bit graph computes U32 low, U32 high and Bool carry in one Block, all before the old-high lookup. Keep that order. `write_mad64_carry_mask` remains protected: Wave32 `sdst = vcc_lo` writes VCC_LO, while an SGPR SDST still writes a pair, with no B32 branch or erase of the old SGPR word. Do not share or change that write. |
| `v_mbcnt_lo/hi` | lead | cross-lane, and it reads EXEC as a mask |
| the VOPC mask write | lead | unchanged |

## 64-bit values

The IR has one 64-bit type, `Type::U64`: a raw 64-bit word, as `U32` is a raw 32-bit word. The
opcode decides what the bits mean. In SPIR-V it is `SpirvCompute::t_u64()`, the 64-bit unsigned
integer whose `Int64` capability the pre-IR emitter already declared (lazily, on first use) for its
register-pair arithmetic. The VOPC I64/U64 compares were moved onto it as the worked example
(`ops/vector_compare_i64_u64.cpp`).

The 64-bit opcodes sit together at the end of `Opcode`:

| opcode | builder | SPIR-V (the pre-IR helper) | meaning |
| --- | --- | --- | --- |
| `PackU64` | `pack_u64(lo, hi)` | `u64_from_lohi` | two U32 words -> `hi << 32 \| lo` |
| `U64Lo`, `U64Hi` | `u64lo(v)`, `u64hi(v)` | `u64_lo`, `u64_hi` | a U64 -> one of its U32 words |
| `U64Eq` .. `U64Ge` | `u64eq` .. `u64ge` | `ucmp(Op_IEqual ..)` | unsigned relations, returning Bool |

There are **no signed 64-bit relations**. `pack_u64_signed_order(lo, hi)` packs the word with its
sign bit flipped, and the unsigned order of two such words is the signed order of the values. That
is what the pre-IR emitter wrote, so it keeps the bytes. `scmp` would bitcast its operands to a
32-bit signed type, and the interpreter's 64-bit relations are unsigned only, so an `S64Lt` would
need both of those changed and cited first.

### Evaluation

`Known` stays a 32-bit `std::optional`, so the folder and every existing caller are unchanged.
`evaluate_wide` holds every value in 64 bits (`Known64`), and `evaluate` is the same pass with a
U64 entry always unknown, because a U64 word has no 32-bit view. A Block whose inputs and result are
U32 or Bool, as every compare's are, needs nothing more than `evaluate`. The 64-bit opcodes are
evaluated in `eval_integer64.cpp`, which `evaluate_wide` reaches through `is_integer64_opcode`.

A new 64-bit opcode goes in the usual five places, with three more things to get right:

- keep it inside the `PackU64` .. last-integer-64-bit-opcode range, so `is_integer64_opcode`
  routes it; the binary64 relations follow that range (`F64Lt` ..), see "F64 values";
- say in `ir.cpp`'s `argument_type` which of its argument slots are U64 (today the range
  `U64Lo` .. `U64Ge` takes U64 in every slot, and `PackU64` takes two U32 words). A shift is
  `(U64 value, U32 count)`, so it needs a per-slot rule;
- `evaluate_integer64` receives every argument zero-extended to 64 bits.

### Reading a 64-bit operand

A guest 64-bit operand reaches the definition as **two U32 words**, low and high. Getting them is
operand bookkeeping, so it stays in `lower/`, and it must keep the old code's rules exactly
(`VopcLowering::integer64_halves` is the model):

- a VGPR, an SGPR or an addressable special pair: `val(source)` and `val(source + 1)`;
- an inline integer: the high word is its sign (`-1` gives `0xffffffff`);
- a 32-bit literal and `SGPR_NULL`: the high word is 0;
- an inline float: refuse. In a 64-bit operand it is a double, not a pair built from its f32 bits;
- keep each read where the old code read it. `lower()` reads the low words first (`ra`, `rc`) and
  the high words later, and those reads can emit.

The definition gets the four words and packs them itself, so the pack is part of its bytes. For a
compare, mark the definition `pair_sources = true`. It then receives `s0`, `s0_hi`, `s1`, `s1_hi`
in `VectorCompareIn`, the lowering calls `emit_vector_compare_pairs`, and tests call
`eval_vector_compare_pairs`. Calling the 32-bit pair of functions with a `pair_sources` definition,
or the reverse, aborts.

**Unused values stay.** The old signed compare packed both sources plainly and as order keys, then
used only one of the two pairs. The I64 definitions append all four packs for byte identity, so an
I64 EQ emits two dead order keys. Removing dead values is a byte change for its own PR.

### Writing a 64-bit result

A VOP3 64-bit result (the shifts) is written as two VGPRs. The old code is:

```cpp
const uint32_t result = b.u64_shift(Op_ShiftLeftLogical, b.u64_from_lohi(lo, hi), amount);
const uint32_t old_hi = vreg_old(b, rs, hi_dst);   // can emit uconst(0)
vreg[in.dst.value] = b.u64_lo(result);
vreg[hi_dst] = b.u64_hi(result);
predicate_write(b, rs, hi_dst, old_hi);
```

`vreg_old` sits between the value and the unpack, and it can allocate a constant, so the unpack must
stay after it. Build two Blocks: the definition's Block ends at the U64 value, and a second Block
takes that value as an input (`ir.input(Type::U64)`) and appends `u64lo` then `u64hi`. Never move
`vreg_old` or `predicate_write`: they are lowering bookkeeping.

### Testing it

- **Opcodes.** `test_ir_core_integer64.cpp` evaluates every 64-bit opcode on hand-worked values,
  enforces the U32/U64 argument types, and runs each opcode's SPIR-V per lane in the interpreter
  against `evaluate`. A new 64-bit opcode adds a row there and a hand-worked case.
- **The per-lane harness takes 64-bit sources.** `LaneInputs::s2` and `s3` load v5 and v6, so a
  64-bit op reads the pairs v[2:3] (`s0` low, `s1` high) and v[5:6] (`s2` low, `s3` high).
  `compare_program(op, form, 0, 0, kVsrc2)` names v[5:6] as SRC1, and `export_compare_witness`
  does that by itself for a `pair_sources` group.
- **Inputs that catch a half-read operand.** Include pairs that differ only in the high word, only
  in the low word with bit 31 set (an unsigned low comparison), the sign boundary
  `0x80000000_00000000` against `0x7fffffff_ffffffff`, `-1` against `0`, and pairs whose signed and
  unsigned orders disagree. Make the random lanes share high words in some lanes, so the low words
  decide there.
- **Constant operands.** Run inline integers, a literal and an inline float through the real
  lowering: the first two extend, and the last refuses
  (`test_ir_vector_compare_i64_u64_execution.cpp`, `ConstantSourcesExtendToSixtyFourBits` and
  `RefusalsStay`). Put the constant-operand programs in the witness corpus too.

The I64/U64 group's gate, with one positive control per definition (the relation mirrored with
swapped operands, or EQ/NE's operands swapped):

| | recorded corpus (1,442 modules) | witness (168 programs) |
| --- | --- | --- |
| the migration | `changes: none` | `changes: none` |
| each of the 12 controls | **none** for 11; `output-changed 1` for `ne_u64` (one *House of the Dead 2* compute program) | `output-changed 14` for each: its own ten forms and four constant operands |

The witness is 12 definitions in five forms and two wave sizes, plus four constant SRC0 operands
per definition. So a 64-bit family needs the witness corpus: the recorded one reaches one of
twelve definitions.

### Stop and report: 64-bit words

Stop and report when a family needs:

- a 64-bit literal: `uconst64` emits a two-word constant only for values above 32 bits, and its
  comment records a driver that misparses that form;
- a signed 64-bit relation that cannot be written with `pack_u64_signed_order`.

Binary64 (F64) values have their own section next.

## F64 values

The IR reads binary64 the way it reads binary32: there is **no F64 Type**. An F64 opcode takes
`U64` words and says they are IEEE binary64 values, as an F32 opcode takes `U32` words. The VOPC
F64 compares (`ops/vector_compare_f64.cpp`, `lower/vopc_float64.cpp`) are the worked example. They
were an **admission**: the old emitter refused every F64 instruction, so there are no old bytes to
match, and the gate is "every newly accepted program is attributed to an F64 site, passes
spirv-val, and nothing else changes".

**Measure first.** When the compares were admitted (2026-10-17), no F64 instruction occurred in
reachable code anywhere we had: 0 of the 4,000 recorded corpus programs (two `v_cmpx_*_f64` words
sat after `s_endpgm`, in data) and 0 of 4,777 distinct programs across 359 `refused_shaders_*`
dumps. So an F64 family un-refuses nothing measurable today, and its gate is the witness corpus.
Re-run the census before you start (`tools/rdna2_ref/decode.py` over the corpus, counting names
with `_F64` before the last `S_ENDPGM`), and report what it says.

### The pieces

| piece | where | what it does |
| --- | --- | --- |
| opcodes | `ir.hpp` `F64Lt` .. `F64UnordGe` | the twelve binary64 relations, `(U64, U64) -> Bool`, after the U64 block (after `U64Bfe`). `argument_type` has its own rule giving U64 to both slots of `F64Lt` .. `F64UnordGe`; a new F64 opcode extends that range or adds a per-slot rule |
| evaluation | `eval_float64.cpp` | `is_float64_opcode` routes them from `evaluate_wide`; the host evaluates in `double` with unflushed subnormals. The constant folder never sees an F64 opcode: keep it that way (no F64 folds) |
| emission | `SpirvCompute::t_f64`, `fcmp64` (`spirv/float64.cpp`) | bitcasts both U64 words to `OpTypeFloat 64`. `t_f64()` declares `Float64`, `DenormPreserve`, `SPV_KHR_float_controls`, `SPV_KHR_float_controls2`, and 64-bit DenormPreserve/SignedZeroInfNanPreserve modes on first use; both operand Bitcasts carry `FPFastMathMode None`. F64 declarations are lazy; existing F32 profile declarations are unchanged |
| device gate | `float64_support.hpp` | `float64_admitted()` is true only after a device owner published shaderFloat64 ENABLED + shaderDenormPreserveFloat64 + independent denormal modes (`denormBehaviorIndependence = ALL`) + SZI64 + enabled shaderFloatControls2 + float controls permitted. Publishers are ANDed (the renderer, `tests/fixtures/render_runner.h`, and the private compute device, `frontends/shared/live/live_compute.cpp`, through `publish_device_float64` in `gpu/execute/float_controls_probe.hpp`). `shader_inspect` publishes a synthetic capable target for offline corpus checks, not a measured device verdict |
| launch MODE | `lower/launch_float_mode.hpp` | the producing launch's FLOAT_MODE for fragment, compute and vertex stages; MODE cannot change inside an admitted program (`s_setreg` of MODE, `s_round_mode` and `s_denorm_mode` refuse) |

### Admission rules a lowering must apply

Every F64 lowering checks these before it emits an F64 opcode, and refuses **by name**
(`b.stage_reject_pc`, `b.stage_reject_reason`, `ok = false`; see `VopcLowering::refuse`):

- **Device:** `!float64_admitted()` refuses `f64-device-unavailable`. A module declaring Float64
  that a device cannot execute is invalid, not slow. Since the module requests only the 64-bit
  denormal mode, the device gate requires `denormBehaviorIndependence = ALL`; restricted
  independence refuses even when `shaderDenormPreserveFloat64` is true. Nonfinite comparison
  inputs additionally require SZI64 and enabled shaderFloatControls2 for explicit Bitcast transport.
  An instruction that reads no F64 value
  (`v_cmp_f_f64`, `v_cmp_tru_f64`) needs no gate.
- **MODE unknown:** a missing launch FLOAT_MODE refuses `f64-float-mode-unavailable`.
- **Input denormals:** FP_DENORM[7:6] (AMD 70648 6.4, Table 24) is shared by F64 and F16. Bit 6
  set (modes 1, 3) keeps input subnormals; `DenormPreserve 64` then keeps the host from flushing
  them. Bit 6 clear (modes 0, 2) flushes them: the lowering does it on the source halves, before
  the definition (a zero exponent field leaves only the sign bit), so the host never sees one.
  `kDefaultComputePgmRsrc1` is mode 0, so test both modes. CONFIDENCE: MED that V_CMP applies the
  input flush; the F32 EQ/NEQ zero relation assumes the same.
- **SDWA and DPP** refuse `f64-compare-sdwa-dpp`: neither describes a register pair.

Source halves (`VopcLowering::float64_source`; reuse it, do not re-derive it):

| operand | low word | high word | evidence |
| --- | --- | --- | --- |
| VGPR, SGPR, VCC/TTMP pair | `val(src)` | `val(src + 1)` | as `integer64_halves` |
| inline float 240-248 | the double's low word | its high word | Table 20: "single, double, or half-precision inline floats"; 1/(2*pi) is `0x3fc45f306dc9c882` |
| 32-bit literal | 0 | the literal | 6.2.1 "Literal Expansion to 64 bits": "64 bit float: the lower 32-bit are padded with zero" |
| inline integer 0, `SGPR_NULL` | 0 | 0 | +0.0 under every reading |
| any other inline integer | refuse `f64-inline-integer-unspecified` | | the ISA does not say whether it is the sign-extended bit pattern or the converted value |
| ABS, NEG | unchanged | `& 0x7fffffff`, then `^ 0x80000000` | the sign bit; hardware order ABS then NEG |

### What the compares did not need, and an arithmetic family will

These are **stop points**: an F64 arithmetic or conversion family (14, 15) must settle each one,
with evidence, or refuse the case by name. Do not guess.

- **Rounding.** FP_ROUND[3:2] is the F64/F16 round mode. A compare has no result to round. An
  arithmetic result needs a measured rounding profile: [Vulkan precision rules](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision-operation)
  leave rounding implementation-defined without an explicit execution mode. Nearest-even needs
  `RoundingModeRTE 64` and device support; the compare gate promises neither. Other guest modes
  need their own supported contract or a proven exact implementation. Strict fused arithmetic
  additionally needs an enabled strict FMA contract; GLSL Fma alone does not establish it.
- **Output denormals.** FP_DENORM bit 7 clear (modes 0, 1) flushes F64 *results*. Flush the result
  halves after the operation, the same way the inputs are flushed, or refuse.
- **NaN payloads.** A compare only classifies NaNs. An arithmetic result's NaN bits are not pinned
  by the interpreter (`bpermute_spirv_oracle.hpp` fails rather than invent them for F32) or by
  Vulkan; decide what the test asserts before writing it.
- **OMOD** on an F64 result already refuses (`float-omod-f64-representation-unavailable`).
- **The interpreter.** `bpermute_spirv_oracle.hpp` executes binary64 only in the twelve
  relations, and fails on a binary64 subnormal unless the module declares `DenormPreserve 64`.
  Each new F64 instruction needs its SPIR-V semantics added there, with the citation.
- **Graphics stages and captures.** The gate is process-global, so fragment and vertex modules are
  admitted on the same verdict as compute. A replayed capture or refused-shader sidecar records no
  F64 verdict; the replaying process publishes its own.

### Testing it

Copy the compare group's three tests:

- `test_ir_core_float64.cpp`: hand-worked binary64 truth for each opcode (NaN on each side, quiet
  and signalling, +0 against -0, the infinities against `DBL_MAX`, subnormals against zero and at
  the normal boundary, a pair differing only in the low word); the U64 argument types; Float64 and
  DenormPreserve, SZI64 and explicit transport declared only by a module that uses them; every
  opcode's SPIR-V per lane against
  `evaluate`; and the interpreter's own control, which must refuse a subnormal once the module's
  mode is rewritten to `DenormFlushToZero`.
- `test_ir_vector_compare_f64_execution.cpp`: the harness in both FLOAT_MODE input modes
  (`kFlush` is the default RSRC1, `kPreserve` sets [7:6] = 3), a hand-worked truth string per
  definition and mode (derive it independently, never from eval), constant operands, ABS/NEG, the
  named refusals (publish nothing, publish a device without the feature, RSRC1 `UINT32_MAX`), and
  the witness exporter. The device gate is process-global: publish in `SetUp`, reset in `TearDown`.
- `tests/gpu/execute/test_compute_f64_compare.cpp`: the same hand-worked rows on the real device,
  every form, both waves, both modes, recording the driver. Run it under
  `flock <GPU_LOCK>`. `tests/fixtures/compute_runner.h` enables `shaderFloat64` and
  `VK_KHR_shader_float_controls` when the device has them, and the test skips (with the reason)
  on a device without the complete F64 controls contract.

The following mutation and gate receipts describe the original admission, before the complete
float-controls correction (#5364). That correction changes F64 module bytes and requires its own
validation; these receipts do not qualify the corrected device profile.

The compare group's original mutations, each of which turned at least one test red:

- `F64Le` evaluated as `<`; `F64UnordLt` emitted as `OpFOrdLessThan`;
- LG defined as NEQ; O reading S0 twice;
- the source's high word not read; a literal placed in the low word; an inline float read as its
  f32 bits; ABS clearing the low word's bit 31;
- the input flush never applied, always applied, or leaving the low word;
- the device gate removed;
- `t_f64()` without `OpExecutionMode DenormPreserve 64` (the interpreter then refuses the
  subnormal lanes).

### Gate and controls

An admission's gate is three questions: is every newly accepted program an F64 site, does each
pass spirv-val, and does anything else change. The witness carries it (16 definitions x 5 forms x 2
waves x 2 input modes, plus 6 constant SRC0 operands per definition). One positive control per
definition, each a byte-changing neutral edit (the relation mirrored with swapped operands, O/U's
arguments swapped, F/TRU rewritten as `F64Lg(s0, s0)` / `F64UnordEq(s0, s0)`), run against the
uncontrolled candidate:

| | recorded corpus (4,000 programs) | witness (416 programs) |
| --- | --- | --- |
| the admission (base = `main`) | `changes: none` | `newly-accepted 416`, every one refused by base at a `V_CMP(X)_*_F64`; 416 of 416 pass `spirv-val`; 364 declare Float64, the 52 F/TRU programs none |
| each of the 16 controls | not run: no F64 site exists there | `output-changed 26` each, exactly its own 20 forms and 6 constants; every execution arm stays green (the opcode-coverage arm goes red for the eight mirrored relations, as it must) |

The recorded corpus cannot see this family at all, so the witness is the whole gate.

## Cross-lane operations

A cross-lane value depends on another lane's data. `v_readlane_b32`'s native-subgroup value is
done and is the worked example (`ops/vector_cross_lane.cpp`). It splits the way the compares and
the carry ops do. The **value** is yours. The **side** stays where it is: the scalar destination,
lane-slot spills, mask-half aliases, `retire_scalar_data_word_mask`, `sreg_bool` erasure,
`lane_local_scalars`, and the subgroup-size contract.

**Measure first.** Count programs by opcode and by DPP control, deduplicated by content, before the
first `S_ENDPGM` (`tools/rdna2_ref/decode.py`). On 2026-10-17 the recorded corpus had 1,003
distinct programs; 553 of them decode only partly, so these counts are lower bounds:

| op | programs | instructions |
| --- | --- | --- |
| `v_readfirstlane_b32` | 144 | 834 |
| `v_readlane_b32` | 126 | 2,697 |
| `v_mbcnt_lo/hi_u32_b32` | 78 / 63 | 242 / 223 |
| `v_writelane_b32` | 68 | 1,794 |
| `v_permlanex16_b32` / `v_permlane16_b32` | 45 / 7 | 191 / 42 |
| DPP16 `row_shr` (on `v_add_nc_u32`, `v_min/max_f32`, `v_or_b32`, ...) | 54 | 752 |
| DPP16 `quad_perm` (fragment `v_mov/v_sub_f32`; compute `v_mov`, `v_cvt_u32_f32`) | 54 | 516 |
| DPP16 `row_ror`, `row_xmask` | 1 | 16 |
| DPP8, `ds_swizzle`, `ds_bpermute`, `ds_permute` | 0 decoded | |

The 5,771 distinct programs of 443 `refused_shaders_*` dumps give the same ranking.

### The two IR opcodes

| opcode | builder | SPIR-V | meaning |
| --- | --- | --- | --- |
| `LaneId` | `lane_id()` | `subgroup_local_id` | this invocation's index in the **host** subgroup |
| `Shuffle` | `shuffle(value, lane)` | `subgroup_shuffle` (`OpGroupNonUniformShuffle`) | `value` of invocation `lane` |

These are the only cross-lane opcodes. Write every cross-lane value from them and the ordinary
integer opcodes: quad, row and wave lane arithmetic is plain U32 arithmetic on `LaneId`. Both
emitter helpers also raise the module's subgroup contract (`compute_min_subgroup_size` at least 4,
or the fragment shuffle reason), as the pre-IR calls did, so the bytes and the contract both stay.

**Evaluation is over a lane vector.** `evaluate` sees one invocation, so it returns unknown for
both opcodes. The constant folder can therefore never fold a cross-lane value. `evaluate_lanes`
(`eval_lanes.hpp`) evaluates a Block for every invocation of a subgroup at once, instruction by
instruction, so a `Shuffle` reads its source invocation's value. The vector's size is the host
subgroup size. Every invocation executes: guest EXEC is a Bool value, not host control flow. A
`Shuffle` lane outside the vector is unknown, which is what SPIR-V calls undefined. U64 values
are out of its domain and stay unknown.

### Value and side

- **A definition takes the wave size, not the subgroup size.** `readlane_b32(ir, s0, s1, W)`
  computes the source invocation as `(LaneId & ~(W - 1)) | (S1 & (W - 1))`. The guest wave is the
  aligned W-lane slice of the host subgroup. With an exact native subgroup the base is 0. With a
  Wave32 guest in a 64-invocation subgroup, the second wave reads within its own slice. The ISA
  says "Lane is S1 % (32 if wave32, 64 if wave64). Ignores exec mask." (AMD 70648, V_READLANE_B32).
- **EXEC is an input, never a mask register.** Readlane ignores it. An op that honours an inactive
  source lane (`v_permlane*` with FI=0, DPP with FI=0, `ds_swizzle`, `ds_bpermute`) receives the
  lowering's `rs.exec` as a `Type::Bool` input. The definition shuffles `Select(exec, 1, 0)` from the
  source lane, as the pre-IR helpers did.
- **The subgroup-size contract stays in the lowering.** `mark_subgroup_min16/32/64` is called before
  the definition, exactly where the old code called it. It describes the module, not the value.
- **What never moves:** the scratch route for a guest wave wider than the host subgroup
  (`guest_wave_readlane`), the lane-slot spill model (`vgpr_lane_slots`, `vgpr_lane_mask_slots`,
  `invalidated_vgpr_lane_slots`), the Wave64 mask-half aliases, `expire_wave64_mask_half`, the NGG
  one-lane projection, the portable dispatcher's CFG paths in `rdna2_emit_cfg.cpp`, and every
  `ok = false` refusal. A cross-lane read of a hard lane is hard under #5185. That rule lives in the
  definedness and proof passes over the decoded instructions (the vector-word marks,
  `rdna2_loop_arbitrary_carry`), not in the IR, so a definition only changes how the value is
  emitted.
- **One emit function per definition, taking bound ids in the old order.**
  `emit_readlane_b32(b, source, selector, W)` builds the Block with the source and selector as
  inputs and returns the Shuffle's id. The old code read `val(src1)` before the contract call and
  the lane arithmetic, so binding the inputs up front keeps the order. When an old operand read
  sits between two emitted values, as `vreg_old` does inside `v_permlane*`'s and `v_writelane`'s
  `sel(...)`, make it an Input at that position and emit with `emit_spirv_resolving_inputs`.

### Steps for one member

1. Relocate its case with `extract_case_routes.py` if it is still in `emit_alu`. The VOP3 cases
   are in `lower/vop3_cross_lane.cpp`: `route_writelane`, `route_readlane` and `route_permlane16`.
2. Write the definition in `ops/vector_cross_lane.cpp` from `LaneId`, `Shuffle` and the integer
   opcodes, appending in the old order. Cite the ISA sentence it implements.
3. In the route, replace only the value's `SpirvCompute` calls with the emit function. Leave every
   other line of the route as it is.
4. Gate the recorded corpus and a witness corpus, and run one positive control per definition.

### Testing it

- `test_ir_core_cross_lane.cpp` pins the opcodes. `evaluate` must never know them.
  `evaluate_lanes` gets hand-worked vectors, with an out-of-range lane and an unknown source. Each
  definition gets an ISA truth table worked by hand, not taken from eval, in three shapes: Wave64
  in 64 invocations, Wave32 in 32, and Wave32 in 64. Every Block's SPIR-V then runs in the
  interpreter's exact native subgroup (`vm.native_subgroup_width = lanes`, with `lanes` invocations)
  against `evaluate_lanes`. Add your definition to those loops.
- `test_ir_vector_cross_lane_execution.cpp` runs the real recompiler with
  `config.native_subgroup_size = W` and the interpreter in an exact W-lane subgroup. It covers
  lanes with differing data, the harness's partial EXEC, SGPR and inline selectors, and selectors
  at and past W. It asserts the module contains `OpGroupNonUniformShuffle`, so a test that silently
  took another route fails. Its positive control, `AnInactiveSourceLaneIsStillRead`, selects an
  EXEC-inactive lane whose value no active lane holds.
- **The interpreter cannot run a Wave32 guest in a 64-invocation subgroup through the recompiler.**
  It requires the native width to equal the invocation count, and `recompile_compute` enables the
  native subgroup only when it equals the wave size. So the Wave32-in-64 mapping is covered at the
  IR level only. In production it applies wherever the native-shuffle path runs in a host
  subgroup wider than the guest wave.

Mutations that went red: the Shuffle evaluated from its own lane; an out-of-range lane wrapped
instead of unknown; `LaneId` emitted as 0 (only the core test sees it, because an exact subgroup
has wave base 0); the Shuffle emitted with its own lane id; the wave mask doubled; the wave base
dropped (core only, for the same reason); the lowering passing a fixed wave size of 64; the
lowering passing a zero selector; the harness without its `v_cmpx`; and the interpreter's shuffle
reading its own lane.

### Gate and controls

Witness: `PROSPER_IR_WITNESS_DIR=~/<slug>-witness ./test_ir_vector_cross_lane_execution
--gtest_filter='*Exports*'`, then `recompile_corpus.py harvest`. It writes 40 entries: ten
selector forms in two wave sizes (20 distinct programs), each once with the native subgroup and
once without it. Each control is a neutral byte change in the definition:

| | recorded corpus (1,442 modules) | witness (40) |
| --- | --- | --- |
| the migration | `changes: none` | `changes: none` |
| `and_(selector, mask)` -> `and_(mask, selector)` | output-changed 5 | output-changed 18 of 20 programs (not the two whose selector is the constant `W - 1`, the same SPIR-V id as the mask) |
| `or_(base, lane)` -> `or_(lane, base)` | output-changed 5 | output-changed 20 of 20 programs |

The recorded corpus reaches this route in 5 modules, so the witness is required.

### The remaining cross-lane members

| member | status | why |
| --- | --- | --- |
| `v_readlane_b32`, fragment static lane with mask slots (`subgroup_shuffle(val(src0), uconst(lane))`) | Codex, with this recipe | `Shuffle(s0, Const(lane))`; check `val`'s order against the constant with the gate |
| `v_writelane_b32`, native dynamic selector | Codex, with this recipe | `Select(IEq(LaneId, S1 & (W-1)), S0, old)`; `vreg_old` sits inside the old `sel`, so it is an Input resolved at its position. The lane-slot forms are lead. |
| `v_permlane16_b32` / `v_permlanex16_b32` | Codex, with this recipe | the value is `subgroup_permlane16`'s lane arithmetic plus a `Shuffle`. With FI=0 it is a second `Shuffle` of `Select(exec, 1, 0)` and a `Select` against `0` or `vreg_old`. `vreg_old` is called twice, once inside the `sel`; keep both calls. |
| DPP16 `row_shr`, `quad_perm`, `row_ror`/`row_xmask` (compute) | Codex, with this recipe, after relocating VOP1's DPP stage (VOP2's is `lower/vop2_dpp_routes.cpp`) | SRC0 transforms built from `LaneId` and `Shuffle` (`subgroup_quad_permute`, `subgroup_row_xor`) |
| DPP16 `quad_perm` (fragment) | lead | `dpp_quad` uses `OpDPdx/OpDPdy` on `FragCoord`, not a shuffle, so it needs a derivative opcode with its own semantics |
| `ds_swizzle_b32`, `ds_bpermute_b32` (native arm in emit_alu's DS case) | Codex, with this recipe, after relocating the two arms out of the DS case | shuffle-only: no LDS. EXEC is a Bool input; `ds_bpermute_b32` reads `((ADDR + OFFSET) >> 2) & 31` within each 32-lane half |
| `ds_bpermute_b32` (portable), `ds_permute_b32` | stop and report | the portable form is a workgroup rendezvous in `rdna2_emit_cfg.cpp`, and `ds_permute` (forward) is not admitted |
| `v_readfirstlane_b32` | lead | emit_alu emits no cross-lane SPIR-V for it: it stores this lane's own value and taints it in `lane_local_scalars` (#3596). The real broadcast is the portable dispatcher in `rdna2_emit_cfg.cpp` |
| `v_readlane_b32` other paths: `guest_wave_readlane`, NGG one-lane projection, lane slots, mask halves | lead | scratch, collapsed-wave and proof state |
| `v_mbcnt_lo/hi` | lead | reads EXEC as a mask |
| DPP8 | lead | not admitted; no decoded use |

## Memory instructions

A memory instruction is three things, and only the first is a definition:

1. **The value**: the address arithmetic (base, offsets, index times stride, SOFFSET, the dword
   index), descriptor field extraction, and any pure conversion of a loaded word.
2. **The access**: the SPIR-V load, store, atomic or sample, its coherence, the robustness rule,
   the EXEC-predicated VGPR write (`vreg_old` / `predicate_write`), and the LDS bounds (#5193).
3. **The proof machinery**: descriptor provenance (`rt->by_fetch_pc`, `by_srt_offset`,
   `by_sgpr_base*`, SRT tags), PC-relative tables, raw and owned windows, the indirect-pointer
   proof, the per-title exact sites, memory-pattern and active-word marks, and the loaded-data
   arbitrary/hard rules (#5193).

The access and the proof machinery stay at their call sites with the same arguments. A family PR
moves the value and nothing else.

**Measure first.** Count programs by family and opcode, deduplicated by content, before the first
`S_ENDPGM` (`tools/rdna2_ref/decode.py`; it names SMEM only, so read the other opcodes from the
encoding fields as `rdna2_decode.cpp` does). On 2026-10-17 the recorded corpus had 1,003 distinct
programs, 227 of which stop decoding before `S_ENDPGM`, so the counts are lower bounds:

| family | programs | instructions | most used |
| --- | --- | --- | --- |
| SMEM | 962 | 22,914 | `s_load_dwordx4` 769, `s_buffer_load_dword` 732, `s_buffer_load_dwordx4` 669, `s_buffer_load_dwordx2` 565, `s_load_dwordx8` 499 |
| MIMG | 694 | 6,698 | `image_load` 305, `image_sample_lz` 277, `image_store` 205, `image_sample_b` 150, `image_sample_l` 115 |
| MUBUF | 576 | 6,962 | `buffer_load_dword` 218, `buffer_load_dwordx3` 202, `buffer_load_dwordx4` 189, `buffer_load_format_xyzw` 163, `buffer_load_dwordx2` 163, `buffer_atomic_add` 138 |
| DS | 221 | 6,304 | `ds_write_b32` 162, `ds_read_b32` 161, `ds_write2_b32` 81, `ds_read2_b32` 72 |
| MTBUF | 18 | 96 | `tbuffer_load_format_xyzw` 11 |
| GLOBAL | 15 | 71 | `global_load_dwordx4` 13 |

The 6,423 distinct programs of 437 `refused_shaders_*` dumps put MUBUF first (5,993 programs,
almost all `buffer_load_dword` and `buffer_store_dword`), then the same order.

### The worked example: the MUBUF/MTBUF address

`ops/buffer_address.cpp` is the byte address and dword index of every MUBUF and MTBUF access that
reaches emit_alu's resource path: loads, stores, atomics and format fetches all consume the same
`addr` and `idx`. It was chosen because it is used by every buffer access and is the least
entangled value in the memory code: it reads only OFFSET, OFFEN, IDXEN, the resolved STRIDE, VADDR
and SOFFSET, and the one provenance fact it depends on (whether the V# is a folded per-vertex
fetch, #206) is decided before it and passed in. SMEM's index math is just as simple but every SMEM
route sits behind a raw-window, owned-backing or descriptor-placeholder proof.

Three forms, selected by the lowering (`lower/buffer_address.cpp`, `shape_of`) from emit_alu's
facts exactly where the old `if` chose them:

| form | value |
| --- | --- |
| `Faithful` | `OFFSET + (IDXEN && STRIDE ? VADDR[0] * STRIDE : 0) + (OFFEN ? VADDR[IDXEN] : 0) + SOFFSET` |
| `FoldedElement` | `ELEMENT * STRIDE`; ELEMENT is `gl_VertexIndex`, `gl_InstanceIndex` or VADDR[0] |
| `FoldedZero` | `0` (a folded V# with STRIDE 0 and no OFFEN) |

and in every form `idx = addr >> 2`.

**A statement run that is not a route still moves with the tool.** The address was a run of
statements in the middle of the MUBUF case, declaring `addr` and `idx` for the code after it, so it
cannot become a `route_*`. Move it as a RAW segment (`-`) into a file you have started with the
function's signature, and close the function by hand:

1. Pass the case's facts in a struct (`BufferAddressFacts`), and bind them to locals under the old
   names at the top of the function, so the moved text compiles unchanged.
2. `--replace` the run with one call that returns the ids and binds them to the old names
   (`const uint32_t addr = address.addr, idx = address.idx;`). Check first that nothing after the
   run reassigns them.
3. Gate it (`changes: none`) and commit it alone.

**The guest reads interleave with the constants**, so the definition reads its operands as inputs
appended where the old code called `val(...)`, and the lowering emits it with
`emit_spirv_resolving_inputs`, calling `val`, `load_vertex_index` or `load_instance_index` at each
input's position. The Block records which word each input slot reads (`BufferAddressInput`), the way
`TernaryInputs` does. Keep the old nesting: `ir.iadd(addr, ir.imul(read(Index), ir.c32(stride)))`
for `b.ibin(Op_IAdd, addr, b.ibin(Op_IMul, val(src0), b.uconst(stride)))`, and the folded element
read in its own statement before STRIDE's constant.

What did not move, and must not: the descriptor resolution that decides STRIDE and the folded
flags, the PC-relative table route (which computes its own address before the resource gate), the
`load_dword` wrapper (indirect-pointer source clamp, the GTA V selected-sbuffer partition, coherent
loads), every `cbuf_load` / `cbuf_store` / atomic, `mark_cbuf_coherent`,
`capture_indirect_pointer_descriptor_source`, the packed and dynamic sub-dword format paths that
consume `addr`, `vreg_old` / `predicate_write`, and every refusal. emit_alu's MUBUF/MTBUF case calls
those with the same arguments it did before.

### Testing it

- `test_ir_buffer_address.cpp`: ISA addresses worked by hand for every form, including the
  modulo-2^32 sum and the truncating dword index; the input order of every form; an unknown word
  leaving the address unknown; and every form's Block run per lane in the integer interpreter
  against its own evaluation.
- `test_ir_buffer_address_execution.cpp`: `buffer_load_dword` through the real recompiler and the
  vector harness, Wave64 and Wave32, in six forms (no VADDR, IDXEN, OFFEN, both; SOFFSET as an
  SGPR, an inline constant and SGPR_NULL), every lane with its own index and byte offset, and the
  harness's partial EXEC. **Keep every address past the harness buffer's first array**: the
  interpreter runs the invocations one after another, and each one's final store overwrites its own
  word of that array, so a lane reading another lane's first-array word sees a stored result.
  The arm asserts it.
- `test_ir_buffer_address_robustness.cpp` (Vulkan): the existing contract that an address past the
  bound range reads 0, which prosper delegates to `robustBufferAccess` (`test_traversal_chase.cpp`,
  #2795). Half the lanes read data and half leave the range, so a load that always returned 0 fails.
  The interpreter cannot run this arm: it fails an out-of-range access chain instead of returning 0.
- `test_ir_lower_buffer_address.cpp`: which form the facts select, observed through whether SOFFSET
  is read. SOFFSET names an SGPR marked as an unavailable saved-mask word, so reading it clears `ok`;
  the folded forms must leave `ok` set because SOFFSET is already in their bound base.

Mutations that went red: SOFFSET dropped from the sum (core); STRIDE replaced by 4 (core; the
harness's own STRIDE is 4, so only the hand-worked values see it); the OFFEN register always
VADDR[0] (execution); the folded forms disabled (lowering); the dword index masked to 63 (core,
execution and robustness).

### Gate and controls

Witness: `PROSPER_IR_WITNESS_DIR=~/<slug>-witness-src ./test_ir_buffer_address_execution
--gtest_filter='*Exports*'`, then `recompile_corpus.py harvest ~/<slug>-witness --root
~/<slug>-witness-src`: 12 entries, the six forms in two wave sizes (6 distinct programs).

| | recorded corpus (1,442 modules) | witness (12) |
| --- | --- | --- |
| the relocation | `changes: none` | |
| the migration | `changes: none` | `changes: none` |
| `Faithful`: SOFFSET's add with swapped operands | output-changed 582 | output-changed 12 |
| `FoldedElement`: the multiply with swapped operands | output-changed 3 | not reachable (vertex only) |
| `FoldedZero`: `0` as `0 >> 2` | `changes: none` | not reachable (vertex only) |

The recorded corpus never reaches `FoldedZero`. Its reach comes from `test_folded_stride_zero`
instead: returning 4 from it turns that test red, and restoring it turns it green.

### The remaining memory groups

| group | who | why |
| --- | --- | --- |
| MUBUF/MTBUF PC-relative table route (`addr`, `idx`, and the constant select chain per dword) | Codex, value-only | pure arithmetic on OFFSET and VADDR plus a select chain over constants; `vreg_old` / `predicate_write` stay |
| MUBUF/MTBUF format decode of a loaded word (UNORM/SNORM/FLOAT16/sub-dword integer, 10_11_11, 2_10_10_10, the runtime byte-address join and `bfe`) | Codex, value-only | pure functions of loaded dwords and the byte address; `load_dword`, the `dyn_*` admission and `dst_sel` routing stay |
| MUBUF/MTBUF store data packing (format encode, the sub-dword atomic clear/set masks) | Codex, value-only | the masks and packed words are pure; the stores, atomics and coherence stay |
| DS address (`base + OFFSET`, `read2/write2` OFFSET0/OFFSET1 and `st64` scaling, the B64 pair indices) | Codex, value-only | pure index arithmetic; `lds_load` / `lds_store`, the LDS bounds (#5193) and the cross-lane arms stay |
| SMEM dynamic dword index (`(SOFFSET + imm) >> 2` plus k, the PC-relative select chain, the table selector `SOFFSET / stride`) | Codex, value-only, after relocating those statements out of the SMEM case | pure; but every SMEM route around them is proof state, so move only these expressions |
| SMEM bounded scalar-buffer load (`in_bounds`, safe index, zero select around `cbuf_load`) | lead | the value straddles the access: the clamp must stay before the load and the select after it |
| SMEM descriptor and raw-window routes (owned raw windows, owned/nested wide chains, pointer and x2/x16 descriptor loads, SRT tags, GTA V selected sbuffer) | lead | provenance and proof state |
| MUBUF/MTBUF descriptor resolution, folded vertex fetch decision, indirect-pointer source and per-title exact sites | lead | provenance |
| FLAT/GLOBAL window path (`addr_lo - base_lo`, plus 4 per component, `>> 2`) | Codex, value-only | pure; the window resolution (`by_fetch_pc`, `flat_base_sgpr`, the push-constant base load) stays |
| FLAT/GLOBAL relocated indirect carriers and the packed-pointer shadow | lead | the address is checked against the indirect-pointer proof's records |
| MIMG (T#/S# resolution, sampled arrays, coordinates, LOD, `image_get_lod`) | lead first, then Codex for the coordinate and result-packing values | descriptor provenance and fragment derivatives; the coordinate conversions are pure once relocated |
| DS atomics, `ds_append/consume`, GWS | lead | memory side effects with ordering |

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
| 8 | VOPC I32 and U32 (`v_cmp/cmpx_{lt,eq,le,gt,ne,ge}_{i32,u32}`, both encodings) | Codex, using the compare recipe |
| 9 | VOPC I16 and U16 (the same six relations; the [15:0] extension goes in the definition) | Codex, using the compare recipe |
| 10 | VOPC F16 (`v_cmp/cmpx_{lt,eq,le,gt,lg,ge}_f16`: only the six the switch admits) | Codex, using the compare recipe |
| 11 | VOPC `v_cmp/cmpx_class_f32` | Codex, using the compare recipe |
| 12 | VOP3 64-bit shifts: `v_lshlrev_b64`, `v_lshrrev_b64` (the two the old code admits, with their exact SRC admission), using a new `(U64, U32)` shift opcode on `u64_shift`; `v_ashrrev_i64` is an admission | Codex, using the 64-bit recipe |
| 13 | VOPC F64 compares (`v_cmp/cmpx_*_f64`) | **done** (admission, `ops/vector_compare_f64.cpp`); the worked example of "F64 values" |
| 13a | VOPC `v_cmp/cmpx_class_f64` (0xA8) | Codex, using the F64 recipe and the `class_f32` model: an admission PR (integer-domain classification of the two halves) |
| 14 | VOP3 F64 arithmetic (`v_add/mul/fma_f64`, `v_min/max_f64`, `v_ldexp_f64`, `v_div_*_f64`, ...) | Codex, using the F64 recipe: one admission PR per coherent group (add/mul/fma first). Stop points: non-RTE FP_ROUND[3:2], the output-denormal flush, NaN payloads |
| 15 | 64-bit conversions (`v_cvt_f64_*`, `v_cvt_*_f64`, `v_frexp_*_f64`, `v_trunc/ceil/rndne/floor/fract_f64`) | Codex, using the F64 recipe: admission PRs. A 64-bit result is written as two VGPRs ("Writing a 64-bit result"); `v_cvt_f32_f64` rounds by FP_ROUND[1:0], the F32 field |
| 16 | VOP3B `v_div_scale_f32`: an admission (emit_alu has no case for it) | Codex, using the mask recipe ("Mask-reading and mask-writing VALU"), in its own explained PR; `v_div_scale_f64` follows family 14 with the F64 recipe |
| 17 | VOP3B `v_mad_u64_u32` / `v_mad_i64_i32`: values in `ops/vector_mad64.cpp` | Codex, value-only; preserve all three outputs before the old-high lookup and the existing `write_mad64_carry_mask` call |
| - | Admitting any compare the switch refuses today | lead, then Codex |
| - | `write_mad64_carry_mask` (its SGPR write differs from the shared one in Wave32), the VOPC mask write (`lower/vopc_mask_write.cpp`), the carry mask write (`lower/carry_mask_write.cpp`), `v_mbcnt` | lead (wave masks) |
| 18 | Cross-lane values: `v_writelane_b32` (native dynamic selector), `v_permlane16/x16_b32`, the compute DPP16 SRC0 transforms, `ds_swizzle_b32` and the native `ds_bpermute_b32`, and the fragment static-lane readlane | Codex, using the cross-lane recipe ("Cross-lane operations"); `v_readlane_b32`'s native value is done |
| - | `v_readfirstlane_b32`, fragment DPP `quad_perm` (`dpp_quad`), DPP8, the portable `ds_bpermute`, `ds_permute`, and the readlane/writelane scratch, lane-slot, mask-half and NGG paths | lead (cross-lane), see "The remaining cross-lane members" |
| 19 | Memory values: the MUBUF/MTBUF PC-relative table, format decode and store packing; the DS address; the SMEM dynamic index; the FLAT window address | Codex, value-only, using the memory recipe ("Memory instructions"); the MUBUF/MTBUF address is **done** (`ops/buffer_address.cpp`) |
| - | `v_movrel*`, `s_movrel*`, and the memory families' access and proof state: SMEM routes, descriptor resolution, FLAT/GLOBAL, MIMG, DS atomics (see "The remaining memory groups") | lead (proof and definedness) |
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
