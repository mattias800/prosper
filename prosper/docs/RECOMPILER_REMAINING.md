# RDNA2→SPIR-V recompiler — remaining work

> **Current note (2026-08-11): this is a historical 41-shader bring-up corpus, not the current GTA V
> coverage boundary.** Later routed gameplay exercises a much larger dynamic compute set and still
> exposes fail-visible instruction, resource and control-flow gaps. The program-tagged terminal census
> and current fixes live in #2481; do not reuse this document's old conclusion that the recompiler was
> "done for this title." The 2026-08-11 gameplay-entry sample contains 29 recompile-empty programs and
> 6 invalid-descriptor programs (35 unique); those route-specific live counts supersede this corpus.

**Date:** 2026-07-06. **Status: ~93.7% instruction coverage in-context; 38 of 41 shaders fully recompile.**
(The earlier "34/41" was a coverage-tool undercount — it ran a per-instruction check that didn't credit
`emit_body`'s loop/if reconstruction, so the MSAA-resolve loop shaders 031-034 were mis-flagged as blocked
even though they recompile. Fixed 2026-07-06; true count is 38/41. **The only genuine remaining blocker is
cross-lane `v_mbcnt`/`v_readlane` — the LDS wave-model.** **UPDATE 2026-07-06: the LDS wave-model is now
BUILT and verified** — `v_mbcnt_lo/hi` with `src0=EXEC` recompiles via a workgroup-as-wave + LDS
prefix-count (kernels 63 full-exec + 64 divergent, both exec-diff-green). The remaining 2-3 shaders use a
non-EXEC `src0` (a ballot-mask value) and `v_readlane`, which build on this same foundation — follow-up
extensions, not new architecture. Still 38/41 pending those. All bounded ALU gaps are closed: `v_add_co_ci_u32`/sub/subrev (VOP3B carry) landed (kernel 62); the 1 shader that used
it now stops at `s_and_b64` (SOP2 0xf, a 64-bit wave-mask op) and, like the other 2, needs the wave-model
underneath. So the LDS wave-model (`v_mbcnt`/`v_readlane` + 64-bit mask ops as workgroup/LDS cross-lane) is
the single remaining recompiler feature for 41/41 on this title — a real architectural change (design note
below), and one that gets no game frames on its own (those 3 are GPU-culling compute, off the first-frame
path; frames are gated on the GPU-executor build). Every other bounded win has been harvested this session.) Every bring-up-critical class is covered (position/blit/clear,
textured incl. 3D sampling + NSA, interpolated, image-copy 1D/2D/3D + arrayed + NSA + MSAA + MSAA_ARRAY,
integer divide/modulo), plus the big structural features: counted-loop reconstruction (`OpLoopMerge`/
`OpPhi`), divergent-if handling (EXEC-predicated linearization incl. provably-dead scalar writes),
64-bit scalar (`s_bfe_u64` via Int64), and NGG vertex shaders. All paths are `spirv-val`-gated
(`tools/spv_validate`, permanent ctest) and, where a Vulkan harness exists, execution-differential-tested.

What that gate does and does not prove (corrected by #1711): it emits **one representative module per
SPIR-V-emitting entry point** — every `recompile_*` stage plus `spirv_builder.cpp`'s hand-assembled
compute modules — and runs `spirv-val --target-env vulkan1.1` on each. It is not a per-title guarantee:
a shader a game submits at runtime is covered only insofar as it exercises the same emitter paths. Two
things it now refuses to do quietly, both of which it used to: pass when `spirv-val` is missing (it was
absent on every CI runner, so the gate had never validated anything there), and stay silent when a new
emitter is declared with no module here.

**Recompiling end-to-end (spirv-val VALID):** the textured/interpolated/image-copy families; the full
MSAA-resolve family **031/032/033/034** (counted loop + 2D_MSAA[_ARRAY]); 3D-sample **028**; compressed-
export **029**; the **NGG family 004/025/040** (position-only + indexed-fetch cull); the bloom/downsample
compute **006** (texture-sample → storage-store). Op families added 2026-07-06: v_mac/v_fmac_f32,
s_lshl4_add_u32, s_bfe_u64 (Int64), v_cndmask_b32_e64, v_cmp→SGPR-pair mask (SDWAB), 2D_MSAA_ARRAY.

**Correctness audit (2026-07-06, exec-diff-verified, kernels 52-58):** VOP3 **source modifiers** (neg/abs,
dword0[10:8]/dword1[63:61]) were being **silently ignored** — `a-b`, `abs()`, `-x` miscomputed in every
recompiling shader; now applied (OpFAbs→OpFNegate). VOP3 **output modifiers** CLAMP (saturate, dword0[15])
+ OMOD (×2/×4/×0.5, dword1[28:27]) now applied (were rejected). v_cvt_pkrtz_f16_f32 sources honor
modifiers. Added the VOP3-encoded forms of v_add/sub/subrev/mul/min/max_f32 (0x103/104/105/108/10F/110,
were rejected), all with source+output modifiers. This is a real fidelity gain for the 34 recompiling
shaders (they use fma/mad/med3/pkrtz **with** modifiers) — "recompiles" now also means "computes the right
values". Not a shader-count change; the remaining 030/037/038 are still cross-lane-blocked (below).

**REMAINING BLOCKERS — ACTUAL BREAKDOWN (2026-07-06, from `shader_histo` first-truly-unsupported-per-shader,
opcodes confirmed via llvm-mc round-trip disasm; SDWA source-modifiers now DONE so VOP2 0x8 is cleared):**
| Blocker | Op | # shaders | Tractability |
|---|---|---|---|
| **s_cbranch_scc0** (uniform cond. branch) | SOPP 0x4 | **4** | control-flow — a *scalar-uniform* if (all lanes same path). The recompiler linearizes divergent (EXEC) ifs and reconstructs loops, but rejects forward scalar branches. Needs structured OpSelectionMerge on the SCC bool. **Largest gap; tractable via the existing block/phi machinery but real CFG work.** |
| **v_mbcnt_lo_u32_b32** | VOP3 0x365 | 2 | cross-lane — needs the workgroup/LDS wave-model (design note below). `mbcnt=lane_id` holds only for full EXEC. |
| **v_add_co_ci_u32_e64** (add w/ carry-in+out) | VOP3 0x128 | 1 | bounded — sum=s0+s1+carryin(VCC/sgpr bool); carryout→mask. Common in 64-bit address math. This shader ALSO needs mbcnt, so fixing it alone completes 0 shaders (marginal for THIS title). |

Corrects the earlier notes: SOPP 0x4 is **s_cbranch_scc0** (uniform conditional), NOT unconditional s_branch;
and VOP2 0x8 (SDWA source-negate) is **now handled** (SDWA neg/abs decode+apply landed, kernels 59/60), which
moved shader-030's first blocker forward to VOP3 0x128. All remaining are correctly REJECTED rather than
faked, and none is on the first-frame path (frames gated on the GPU-executor, not the recompiler). Next wins
in priority: (1) structured s_cbranch_scc0 → +4 shaders (biggest); (2) LDS wave-model → +2 (mbcnt/readlane);
(3) v_add_co_ci_u32 → correctness/coverage for address math (0 completions here). Coverage now 93.2%
in-context (unsupported 107), 34/41 shaders.

**Historical conclusion (superseded):** this corpus once suggested that the recompiler was done for
the title. Routed gameplay later falsified that generalisation; see #2481 for the current dynamic set.

## How to eventually add the cross-lane wave ops (design note, 2026-07-06)
The remaining shaders (030/037/038) need `v_mbcnt_lo/hi` (this lane's index among active lanes) and
`v_readlane` (read another lane's value). The obvious lowering — SPIR-V subgroup ops
(`OpGroupNonUniformBallot` + `…BallotBitCount`, `OpGroupNonUniformShuffle`) — **will not work faithfully on
our test path:** llvmpipe reports `subgroupSize = 8` with `minSubgroupSize == maxSubgroupSize == 8` (Mesa
25.2 / LLVM 20), and RDNA2 waves are 32 or 64 lanes. A 32/64-lane wave can't be one 8-lane subgroup, so
`mbcnt`/`readlane` computed over a subgroup would use the wrong lane grouping → wrong results, and there's
no way to force a 32-wide subgroup on llvmpipe. **The faithful model is a workgroup-as-wave with LDS:**
dispatch one workgroup per wave (local_size = wave size), keep the per-lane active mask + values in
shared memory, and compute `v_mbcnt` as an LDS prefix-count over the active mask and `v_readlane` as an LDS
read + barrier. That works on any Vulkan (no subgroup-size dependency) and is execution-differential-testable
on llvmpipe. It is a real architectural change to the recompiler's per-invocation model (add an LDS/wave
layer), not a bounded add — the right investment for cross-title generality, best done deliberately.

**Concrete implementation plan (scoped 2026-07-06 — the compute shell ALREADY has what's needed):**
`begin()` sets `EM_LocalSize 64` and `barrier()` emits `OpControlBarrier(Workgroup)`, so a workgroup IS a
64-lane wave and inactive lanes still EXECUTE (exec is a predication bool), so they still reach barriers.
Steps: (1) in the compute shell declare `gl_LocalInvocationID` (BuiltIn 27, Input uvec3) → load `.x` =
`localid`, and an LDS array `uint active[64]` (StorageClass Workgroup=4). (2) `v_mbcnt_lo_u32_b32 dst,
src0, src1`: only when `src0` is EXEC (126/127) — else reject (we lack the general 32-bit mask value);
emit `active[localid] = (exec ? 1 : 0)`, `barrier()`, then an UNROLLED prefix-count `sum = Σ_{i=0..31}
(i<localid ? active[i] : 0)` (32 iters: `ucmp ULessThan(i,localid)` → `sel` → `iadd`), `dst = src1 + sum`,
trailing `barrier()` so the next op can't overwrite LDS mid-read. `v_mbcnt_hi` is identical over i=32..63.
Combined lo→hi (hi's acc = lo's dst) = full 64-lane compaction index = count of active lanes below `localid`
— exactly what culling/compaction wants (correct for PARTIAL/divergent exec, unlike a `localid`
approximation). (3) HAZARD: barriers must be wave-uniform — valid only when the mbcnt is NOT inside a
divergent structured-if/loop. For the current compaction shaders it's top-level (uniform), but a general
guard should reject mbcnt emitted inside `emit_body`'s if/loop paths. (4) TEST: a compute kernel that
`v_cmpx`-narrows exec by a per-lane predicate, then mbcnt, storing the compaction index — expected computed
per-64-lane-workgroup on the CPU (count of predicate-true lanes below each localid). `v_readlane` similarly
via an LDS `value[64]` write+barrier+read of `value[srclane]`. This is ~100 lines of careful SPIR-V + a
non-trivial divergent test — a focused/reviewed effort, not a tail-of-session rush.

The remaining shaders each need **one or more genuinely deep features** — verified below by disassembly.
**None is completable by a single bounded opcode/feature add**, and **none is needed for the first frame**
(the critical path to on-screen graphics is the GfxDevice boot wall; see `GFXDEVICE_BRINGUP_PROBLEM.md`).

## What each remaining shader needs

| Shaders | Class | Blocking features (all required together) |
|---|---|---|
| **032, 034** | loop + arrayed/MSAA sampling | loop reconstruction (done) + **2D_ARRAY / 2D_MSAA_ARRAY** *sampled* image dims (arrayed/MSAA storage is done; arrayed/MSAA **sampling** is not) |
| **004, 025, 040** | NGG vertex/primitive | **NGG preamble**: `s_sendmsg(GS_ALLOC_REQ)`, `exp prim`, and wave-packing EXEC setup (`s_lshr_b64 exec,-1,vcc`) — modellable as per-invocation no-ops, but needs care to prove correctness |
| **006** | multi-tap sample + inline sampler | **inline sampler descriptor construction** (`s[8:11]` reused as a buffer V# then rebuilt as an S# via `s_movk`/`s_bfm`/`s_lshl`/`s_mov`) + dmask≠0xF (3-component) sampling. NSA + implicit-LOD sampling itself is done. |
| **030, 037, 038** | large / wave-level | **wave-level ops**: `s_bfe_u64`/`s_lshr_b64` writing **EXEC** (wave-lane setup), **cross-lane** `v_mbcnt_lo/hi` (active-lane count), `v_readlane`; plus a long ALU tail. Hardest; multiple features each. |

## Structured uniform-if: single forward s_cbranch_scc0 — **LANDED** (2026-07-06)
The single forward `s_cbranch_scc0`/`scc1` case now recompiles: `detect_forward_if` + a structured-if path
in `emit_body` (OpSelectionMerge + OpBranchConditional on the SCC bool + a merge OpPhi per register written
in the conditional block). Additive (loop + straight-line paths untouched), verified by exec-diff kernel 61
`(a<b)?a+b:a`, spirv-val green. **This did NOT move the 34/41 count**: the 4 shaders that `shader_histo`
flags at SOPP 0x4 have more complex control flow (multiple/nested branches, likely if-else or
loop-with-inner-if) that the conservative single-if detector rejects — and `shader_histo`'s blocker report
is a *static per-opcode* check, so it still lists SOPP 0x4 regardless of the new whole-stream handling.
**Follow-up for the +4:** extend to a SEQUENCE of non-overlapping forward-ifs (still bounded), then a
general relooper-style structurizer for nested/if-else/loop+if. Needs per-blocked-shader CF dumps to size
(shader_histo dumps only the biggest, which is branch-free — a small tool to dump a *named/blocked* shader
would help). The single-if machinery (value-map + emit_phi_2way at a join) is the reusable foundation.

## (original plan retained) structured s_cbranch_scc0
Scoped 2026-07-06. Current state: `s_cbranch_scc0`/`scc1` are handled ONLY as a loop's single exit branch
(the `emit_body` loop reconstruction, rdna2_to_spirv.cpp ~line 819/1691); a general **forward** scc branch
that is NOT a loop exit is rejected (`emit_alu` SOPP case 0x04/0x05 → `ok=false`, ~line 1339). These are
scalar-**uniform** conditionals (all lanes take the same path — distinct from the EXEC-predicated divergent
ifs, which are already linearized). Implementation approach:
1. **Detect** in the pre-pass (alongside loop detection): a forward `s_cbranch_scc0 T` at pc P where T>P and
   [P+1,T) contains no other branch out and T is not a loop header — a single structured if. (Start with the
   non-nested single-forward-if case; nested/irreducible needs a general relooper — defer.)
2. **Emit** as structured selection: emit [.., P); then `OpSelectionMerge(Lmerge=T)` +
   `OpBranchConditional(scc_bool, Lfall, Lmerge)` where scc_bool is `rs.scc` (already tracked by s_cmp/
   s_and etc.). For scc0 the branch is TAKEN (skip to T) when SCC==0, so the fall-through block Lfall =
   [P+1,T) executes when SCC!=0. Emit Lfall, then branch to Lmerge=T, then continue [T,..).
3. **Values across the merge**: the skipped block writes SGPRs/VGPRs; registers written in Lfall and live
   after T need an **OpPhi** at Lmerge merging the pre-branch value with the Lfall value. This is exactly
   the per-block value-map + phi machinery the loop reconstruction already has — the real work is
   generalizing it from the loop shape to an if-merge (make the value-map per-block with phi at any join,
   not just the loop header/back-edge). `rs.scc` is a per-lane bool but the branch is uniform, so a normal
   OpBranchConditional is valid (no EXEC interaction).
4. **Verify**: a synthetic kernel `s_cmp_lt; s_cbranch_scc0 skip; <write>; skip: <use>` through the
   execution-differential harness, plus spirv-val. Then re-run shader_histo (expect the 4 SOPP-0x4 shaders
   to advance past this blocker).
This is a real but bounded CFG addition (parallels the loop work); best done deliberately with review since
it touches the control-flow core every shader flows through.

## The big lever: loop reconstruction — **LANDED**

Counted-loop reconstruction (`OpLoopMerge` + `OpPhi` for loop-carried registers, per-block value maps)
and divergent-if handling now recompile the MSAA-resolve fragment shaders 031/033 end-to-end. The design
notes below are retained as the reference for the value-map/phi machinery.

Worked example — **shader 031** (an MSAA-sample-average resolve):
```
  s_mov_b32 s10, 0                    ; loop counter
  <loop head>:
    s_cmp_lt_u32 s10, s11             ; SCC = counter < sampleCount
    s_cbranch_scc0 <exit>             ; uniform loop-exit branch
    s_and_saveexec_b64 / s_cbranch_execz   ; inner per-lane if (already handled)
    image_load ... dim:2D_MSAA        ; needs MSAA
    v_add_f32 v9..v6, ...             ; loop-carried accumulators
    s_add_i32 s10, s10, 1             ; counter++  (SGPR write in the loop body)
    s_branch <loop head>              ; BACKWARD edge
  <exit>: v_rcp/v_mul (average), exp mrt0 compr
```
Why the current model can't do it: the recompiler is **straight-line SSA** — `rs.vreg[i]`/`rs.sreg[i]`
map each register to a single current SSA id. A loop needs the value of `v9`/`s10` at the loop head to be
an **`OpPhi`** merging the pre-loop value with the back-edge value. That requires:
1. **Basic-block reconstruction** from branch targets (loop head, body, exit as distinct blocks).
2. **Structured control flow**: `OpLoopMerge` + `OpBranchConditional` on the SCC condition, with the
   back-edge; the inner divergent-if stays EXEC-predicated as today.
3. **`OpPhi` for loop-carried values**: identify registers written in the body and live across the
   back-edge (`s10`, `v6–v10`), and emit a head-block phi per such register — i.e. the value-map must
   become **per-block with phi merges at join points**, not a single global map.

This is a significant change to the recompiler core (not a bounded increment), and by itself still won't
complete 031/033 without **2D_MSAA** image support (a distinct feature). It is the right investment for
**generality** (any branchy/loopy shader in any title — cf. the UE4 cross-engine goal), but it is a
deliberate, multi-step effort best done with a human in the loop, not an overnight bounded patch.

## Recommendation

Given (a) all remaining shaders are advanced effects not needed for the first frame, and (b) the actual
graphics blocker is the boot wall, the highest-value next steps are, in order:
1. **The GfxDevice boot wall** (interactive/reference-backed session) — the true critical path.
2. NGG preamble (`s_sendmsg`/`exp prim`) — unblocks 004/025/040.
3. Arrayed/MSAA *sampled* image dims — unblocks 032/034 (loop reconstruction they also need is done).
4. Inline-sampler descriptor construction + 3-component (dmask) sampling — unblocks 006.

Marginal instruction-coverage ops (e.g. `v_cndmask_b32_e64`, `s_bfe_u64`) can still be added safely but
**complete no additional shader** on their own, so they are deferred in favour of the above.

## Read the reject's `dpp=` / `sdwa=` fields before believing `mode=unresolved-operand`

`unresolved-operand` means "the lowering exists and an operand did not resolve", which reads as a
descriptor problem — and it is, most of the time. But the modifier fields on the same line can
contradict it, and when they do they are the answer.

Worked example (2026-08-16, GTA V PPSA04263). Two 1920×1080 screen-space kernels rejected on
`V_MIN_F32 … row_xmask:4 row_mask:0xf bank_mask:0xf bound_ctrl:1` with `mode=unresolved-operand`. The
operand in question was `src=250`, which is the **DPP marker** — yet the same line printed
**`modifier=1 dpp=0`**. That pair is the tell: the second dword was consumed as a modifier word, the
DECODER declined the control value, so the instruction never became a DPP instruction and the DPP
lowering was never reached. Chasing the descriptor would have been chasing nothing.

The fix was to admit the control, not to write a lowering: `ROW_ROR:8` (`0x128`) and `ROW_XMASK:n`
(`0x160..0x16f`) are one family — `ROW_XMASK:n` is XOR n by definition and XOR 8 is exactly
`(row_lane - 8) mod 16`, so `ROW_ROR:8` **is** `ROW_XMASK:8`, and `subgroup_row_ror8` had always been
written as an XOR of the lane id. Every stride below 16 touches only bits 0..3, so the source lane
stays inside its own architectural DPP16 row — which is what makes the whole family exact
*independently of the host subgroup width*, and it was never specific to 8. `XMASK:0` is **included**:
it is the identity permutation, and lowering it as a lane-XOR by 0 is what the hardware does. An
earlier revision excluded it so an identity result could not be mistaken for a decode error, but
excluding it does not fail visibly either — it falls through to the generic reject and names a
defined control as unsupported, which is the worse signal of the two.

## VCC_LO as scalar scratch: `S_CSELECT_B32` with non-constant sources — **LANDED** (2026-08-20)

`is_wave64_vcc_lo_scalar_cselect` (`rdna2_alu_support.hpp`) used to require **both** sources to be
inline constants, because an inline operand is an exact dword independent of its value. LLVM also
emits the same recycling with ordinary SGPR sources, and chains it through VCC_LO itself, so the
predicate now admits any **scalar-DATA operand kind**: `InlineInt`, `InlineFloat`, `SGPR`, and
`Special` **only for value 106 (VCC_LO)**.

Three facts make that narrower than it reads, and all three are load-bearing:

* **`Special` is admitted for 106 alone.** The stock Unreal froxel kernel rejects as a *four-wide
  cascade* — `s_cselect_b32 vcc_lo, s37, s36` at pc217, then `s_cselect_b32 vcc_lo, s38, vcc_lo` at
  pc220/223, whose second source decodes as `Special` 106. An SGPR-only widening fixes the first
  reject and re-rejects at the second, which looks like success in a log. Admitting `Special`
  106..125 wholesale instead breaks two VCC_HI packet-drift guards
  (`test_rdna2_to_spirv.cpp`, "rejects packet drift and a live high-half mask" and "rejects pc232
  packet drift into the killed VCC_HI word"); 106 alone restores both.
* **The predicate decides SHAPE, never whether a source HOLDS scalar data.** That stays with the
  whole-stream pre-pass (`source_is_scalar_word`) and `operand_bits`, which rejects any source whose
  dword is not representable in the per-invocation model.
* **Acceptance still needs the separate per-PC proof** — `complete_scalar_pair` *or*
  `vcc_b32_low_only_pcs`. For the Unreal kernel it is `complete_scalar_pair`: the dead-high proof
  does **not** apply, since VCC_HI is that shader's loop counter and stays live past the select.
  `complete_scalar_pair` proves strictly more (it reconstructs the architectural predicate from both
  physical words rather than discarding the high one).

**What this established and what it did not.** Measured by a live env-gated A/B on *The Plucky
Squire* `0x3015fd0000` (control vs arm in one binary, same route, back to back): `executed=0
skipped=6` → `executed=6 skipped=0`, with no further reject behind it. That is a claim about
**programs executing**, which is the contract. It is **not** a claim that any image changed — see the
#2481 row in *Ruled out*, where widening this same family for GTA V left the terminal byte-identical.
#2747's own prediction is that this restores a lighting pass and moves **no** title off its rung.

Scope, precisely: of the six froxel programs #2747 names, this predicate unblocks **four, across three
titles** — `PPSA15319` `0x3015fd0000`, `PPSA17942` `0x3017400000`, `PPSA01826` `0x200ea80000` and
`0x200ead0000`. The other two, `PPSA15319` `0x3015ab0000` and `PPSA05143` `0x30114c0000`, reject on the
byte-identical `s_mov_b32 s14, m0` (`be8e037c`) and are **not** touched: that half of #2741 needs a
narrower liveness-proved form, and #134's `kernel X2 … is REJECTED` must not be reverted wholesale.
Both of those are their titles' *main-view* volumes, so *Plucky Squire* and *Little Nightmares III*
keep their main froxel pass absent. #2741, #2747.

## A saved wave-mask alias is not permanent provenance — **FIXED** (2026-08-20)

`RegState::sreg_bool` maps an SGPR root to the Bool spelling of the wave mask that physical pair
holds. `record_scalar_write`'s erase loop is guarded by `rs.sreg_bool_b32.contains(reg)`, so it ended
that lifetime **only for the Wave32 B32 spelling**: a saved **B64** mask stayed keyed on its root
register for the rest of the shader, surviving every later scalar write to it.

That was invisible while nothing treated the alias as a liveness fact. #2481's `operand_bits` reject
(`e09a0cec`, 2026-08-11) does treat it as one — a data read of a word whose root is in `sreg_bool` is
refused, on the grounds that "a persisted B64 wave mask has no ordinary scalar dword" — so an
unbounded lifetime turned an ordinary recycled register into a shader-wide reject.

**The shape that hits it is a compiler idiom, not a corner case.** Save a mask into a scratch SGPR
pair; later reuse the pair for a PC-relative embedded-table address. R-Type Delta's
`shader/sprite_i_vv.ags` does exactly that: `s_cselect_b64 s[0:1], exec, 0` in the NGG fetch prologue
at pc 38, then `s_getpc_b64 s[0:1]` / `s_add_u32 s0, lit, s0` / `s_addc_u32 s1, 0, s1` at pc 303-306.
The add rejected with `mode=unresolved-operand`, the vertex stage returned `{}`, and every sprite draw
in the title was dropped for nine days (#2783).

The transfer function now follows `expire_wave64_mask_half`'s existing rule for the promoted-half
spelling, deliberately narrower: that one expires on **either** word of the pair, this one only on the
root. Three scoping decisions are deliberate, and each was forced by a failing arm rather than chosen:

* **The snapshot is taken before `emit_alu`.** `record_scalar_write` runs *after* the emitter has
  already materialized the new lifetime, so "was this alias here, with this exact Bool id, before?"
  is the only way to tell a stale entry from one this same instruction published. Classifying
  publishers syntactically is not enough: `scalar_write_is_b64_mask` knows the SOP1/SOP2/VOPC/VOP3B
  writers, but the `vgpr_lane_mask_slots` path in `emit_alu` republishes a spilled mask alias from
  `v_readlane` with no syntactic marker at all, and a syntactic guard silently dropped it. The id
  comparison is a *proxy* for "did not publish", and that same reload is the one publisher that could
  in principle re-store an identical id. If it were reached the alias would be dropped, and the
  outcome is **fail-visible, not silent**: `src_mask` resolves a missing `sreg_bool` entry to 0 and
  every Bool-domain consumer then clears `ok` (`rdna2_emit_alu.cpp` `:794`, `:817`, `:873`, `:1058`,
  `:1074`), so the stage rejects. Silent zero is the *data*-domain outcome only. The residual is
  therefore bounded by being loud, not by being harmless — and it is the same failure class this
  change repairs, which is the reason to keep it in view rather than to discount it.
* **Only the ROOT word ends the lifetime**, and that is measured, not reasoned. Expiring on either
  word of the pair rejected **19** arms of `test_recompile_coverage`. Traced on the first of them
  ("a nested varying-VCC compute CFG preserves spilled EXEC"): its Wave64 EXEC reload keys the
  reconstructed mask on the **low** word (`v_readlane s14` → `sreg_bool[14]`, through the *non-native*
  `vgpr_lane_mask_slots` branch, which erases `sreg_wave64_mask_half` so the
  `publishes_wave64_mask_half` guard does not apply) and then writes the **high** word
  (`v_readlane s15`). Ending the lifetime on that high-word write destroyed the alias the very next
  instruction consumes, and `s_mov_b64 exec, s[14:15]` rejected. A high-word-only overwrite by
  unrelated scalar data therefore stays conservative — a **pre-existing** gap, not one introduced
  here, since before this nothing ended a B64 alias at all.
* **VCC (106/107) is out of scope — and not because it is unreachable.** SGPR-kind operands really
  can carry 106/107 (`sgpr()` masks to 7 bits, `rdna2_decode.cpp`), and SOPK's read-modify-write forms
  read their own destination through `val(in.dst)`, so `s_mulk_i32 vcc_lo, imm` does reach
  `operand_bits` with an SGPR-kind 106. What makes VCC different is that its mask state is *mirrored*
  in `rs.vcc`: expiring `sreg_bool[106]` without a matching policy for `rs.vcc` would leave the two
  spellings disagreeing, which is a separate change with its own risk and would also alter
  established behaviour for every `s_bfe_u32 vcc_lo` NGG preamble. No observed defect requires it —
  reaching the reject through VCC additionally needs a writer that overwrites VCC_LO while leaving no
  scalar SSA value behind, because `operand_bits` consults `rs.sreg` first. Recorded as **#2804** so
  the scoping decision is not mistaken for a proof of impossibility.

The regression arm is in `test_recompile_coverage` (pure, no Vulkan): the already-proven "T12 vertex"
PC-relative-table program as a positive control, and the identical program preceded by the mask save
into the pair it reuses. Only the second fails without the fix.

**Which way the exposure runs, stated precisely**, because "it only shortens a lifetime" is true of
the *state* and misleading about *acceptance*. Shortening a `sreg_bool` lifetime is two-directional:
a **data**-domain read of the recycled word goes reject → accept (the repair), while a **Bool**-domain
read of a word this model now considers dead goes accept → reject. That is not hypothetical — it is
exactly what the 19-arm result above measured, and it is why the root-only rule exists. The residual
risk is therefore a shader whose mask prosper believes is dead and hardware does not; it fails loudly
if it happens.

## The PC-relative embedded-table fold had no TYPED consumer — **FIXED** (2026-08-21)

`detect_pcrel_tables` (`rdna2_cfg_support.hpp`) folds a constant table the compiler placed inside the
shader blob and addressed with an `s_getpc_b64`-built V#. It recognised the table's **untyped**
consumer (`MUBUF` raw loads) and its **scalar** one (`SMEM` s_buffer loads) and not its **typed** one
(`MTBUF`, `tbuffer_load_format_*`). With a typed consumer the table never folds, so `s_getpc_b64` has
nothing to justify it (`rdna2_emit_alu.cpp` accepts the pair only when a table from this shader was
proven) and the whole program rejects `mode=unresolved-operand` **at the getpc, not at the load**.

*Sonic Frontiers*' three Cyber Space scene-width compute kernels hit this twice each. Two things about
it generalise:

* **The reject names the getpc, which is not where the gap is.** Nothing about
  `be801f00 fmt=1 op=0x1f` says "your table fold does not know about typed loads"; the instruction that
  carries the information is ten dwords further on. This is the same shape as #2481's rule — *a reject
  pc names where a fact was consumed, not where it was lost* — reached from a different direction.
* **A one-component 32-bit typed format converts nothing.** `rdna2_buffer_format` maps BUF_FMT 22 to
  `Float32`/n=1, so `tbuffer_load_format_x` at that format returns the stored dword unchanged and the
  existing constant-lookup fold was already exactly right for it. The typed case is admitted only when
  the format stores 32 bits per component, its component count matches the opcode (a narrower format
  default-fills the missing components rather than reading them), **and** the V#'s DST_SEL routes those
  channels straight through — a FORMAT load applies DST_SEL and a raw load does not, and Frontiers' own
  table descriptor carries `DST_SEL = (X, 0, 0, 0)`, so a four-component fetch through it would return
  one stored dword and three constant zeroes.

#2859, PR #2861. `tests/gpu/test_recompile_coverage.cpp` carries **five programs and ten checks**,
three of the programs negative controls — a converting format, a non-identity DST_SEL, and a branch
target entering after the `s_getpc_b64` — because without those the change is indistinguishable from
"accept every MTBUF". Each control is mutation-verified: breaking its predicate reddens its arms and
nothing else.

**Ask the detector, not the compile.** The branch-entry arm was first written as a
`recompile_vertex(...).empty()` check and it passed under a mutation reverting the exact predicate it
was meant to pin — that program does not compile anyway, because its forward branch is unlowerable,
so the assertion could never have failed. `detect_pcrel_tables` is pure, so the arms call it directly
and assert the recorded map. A recompile-outcome assertion cannot discriminate a change that only
decides whether a table was recorded.

## `s_mov_b32 exec_lo, -1` from an unnarrowed EXEC — **LANDED** (2026-08-22)

The graphics wave size is not plumbed into the recompiler, so the vertex shell sets
`b.allow_b32_masks = b.ngg_one_lane` — an exception for one byte-exact captured Astro Bot wrapper,
not a property of any stage. Every Wave32 mask idiom therefore fails closed in a graphics stage, and
that is correct in general: "write all-ones to the LOW EXEC dword" is the whole mask at Wave32 and
half of it at Wave64, and reading one as the other re-enables the wrong lanes.

**One case needs none of that.** `exec_narrowed == false` is prosper's invariant for "every lane is
on". From that state the instruction rewrites the low dword with what it already held and leaves the
high dword, if any, untouched — the postcondition equals the precondition at either width. It is
therefore an exact no-op, derived rather than approximated, and it is now lowered as one in every
stage (`rdna2_emit_alu.cpp`, immediately after the `allow_b32_masks` `s_mov_b32` block).

Scope, and why it cannot regress a working title: every earlier arm returns first — Wave64 compute
and fragment at the `(is_compute||is_fragment) && wave_size==64` gate, everything Wave32 at the
`allow_b32_masks` gate — so the rule is reachable only in a graphics stage that rejects outright
today. It can only turn a reject into an accept.

The narrowed form (Wave32 reconvergence) and any non-all-ones value still reject, and
`test_rdna2_spirv_struct` pins all three arms; both counter-arms are mutation-checked.

Why it mattered: *Yakuza Kiwami* (`PPSA31334`) emits it at pc=3 in **19 of the 20** distinct
programs a boot reaches, so the whole title's draw set was skipped for it. It now rejects one
instruction later, at `bf8a0000` (`s_barrier`, pc=8) — the **generic NGG merged-stage prologue**,
which is item 2 of the Recommendation list above and a frontier rather than a missing opcode. See
`YAKUZA_JUDGMENT_BRINGUP.md`.

## Fragment varyings never got SPIR-V `Flat` from the guest's own FLAT_SHADE bit — **FIXED** (2026-09-01)

`AgcShaderSemantic::is_flat_shaded()` was decoded into the guest's PS_INPUT_CNTL control word
(`agc_shader_layout.cpp:98`/`:123`, the `0x400` FLAT_SHADE bit) and then discarded: `flat_attrs`, the
only thing that decorated a fragment Input variable `Flat`, was populated from exactly one source —
`VINTRP opcode == 2` (`v_interp_mov`, #152/#897) — so an attribute the guest declared flat-shaded but
the shader happened to read with an ordinary `v_interp_p1`/`v_interp_p2` pair stayed
smooth-interpolated. On hardware FLAT_SHADE makes the parameter cache deliver only the provoking
vertex's value for that slot (P10=P20=0), so *every* VINTRP read of it is a constant across the
primitive regardless of opcode; Vulkan needs the `Flat` decoration to reproduce that.

`FragmentInterpolationLayout` now carries `flat_mask`, from the new
`PixelInputMapping::effective_flat_mask()` (mirrors `effective_passthrough_mask()`'s shape),
computed in `fragment_interpolation_layout()` alongside `passthrough_mask` and unioned with the
existing `flat_attrs` set in `frag_input()` — the two sources are unioned, not one substituted for
the other, since either alone misses cases the other catches. It also had to be threaded into the
`InterpolationCacheKey` memo (`gpu_executor.cpp`), which keyed on shader identity plus
`passthrough_mask` alone: without `flat_mask` in that key, two draws sharing one fragment shader's
bytes but different guest FLAT_SHADE state could answer from a stale layout computed for the other
draw's flat-ness — a caching defect the fix itself would have introduced silently.

No title in the corpus was found to exercise the bit before this fix — #3051's own investigation
measured 224,363 live `[interp]` control words on *Tomb Raider I-III Remastered* with none setting
`0x400` — so this closes a latent correctness gap rather than a live regression. `test_rdna2_spirv_struct`
pins it: the same smooth-only bytes as the pre-existing "stays smooth" negative case are recompiled
again with a synthetic `PixelInputMapping` control word carrying only `0x400`, and the emitted module
must now carry a `Flat` decoration it did not have before. #3051

## Ruled out

Cross-title falsifications where the **recompiler was blamed and exonerated**. One line per dead
hypothesis, the evidence that killed it, and where that evidence lives. Do not re-derive these
without contradictory new evidence.

| Hypothesis | Verdict and evidence | Source |
|---|---|---|
| *Stray*'s dropped full-screen title-screen draws are blocked on **`v_mbcnt` having no vertex-stage path**: give the vertex shell a general SGPR-mask MBCNT lowering and vertex `0x300f190000` recompiles | **Half falsified, and the surviving half is not implementable in the vertex shell.** (1) *Not sufficient.* With MBCNT satisfied by a throwaway probe, the live-config `gpu_replay --retry-failed-stage 1:1` reject moves ten instructions -- from `pc=4` (`v_mbcnt_lo_u32_b32 v3, -1, 0`) to `pc=14` `ds_write_b32` -- and `--retry-failed-chain 1` to `pc=116` `ds_read2_b32`. The program is a merged **NGG ES+GS threadgroup**, not a vertex shader: 288 instructions carrying LDS traffic at four offsets, three `s_barrier`s, `s_bcnt1_i32_b64` x2, `s_pack_ll_b32_b16`, two `v_add_nc_u32_dpp row_shr` cross-wave scans, `v_readlane_b32` x2, `s_sendmsg(MSG_GS_ALLOC_REQ)` and `exp prim`. MBCNT is one of eight wave/workgroup primitives it needs, and the first one only because it is first in program order. (2) *Not implementable as posed.* The vertex shell's guest lane id is MODELLED as the flattened vertex/instance invocation index (`guest_lane_id()`, `rdna2_to_spirv_internal.hpp`), and nothing in Vulkan promises that the invocations sharing a subgroup are the `wave_size` consecutive flattened indices that model calls one guest wave -- so a subgroup prefix scan would count over a different lane set and return a well-formed wrong compaction slot, the silent-miscompile shape. Compute's workgroup-as-wave LDS model (design note above) is unavailable too: a Vulkan vertex stage has no shared memory. The subclasses that ARE exactly lowerable are already covered -- `ngg_one_lane` (a wave of one lane, so any mask contributes zero) and the all-ones lane-index form under `ngg_logical_lane`. **The gap is a missing WAVE, not a missing lowering.** The architecturally honest home for a merged NGG program is a **mesh shader**, where a workgroup is the NGG threadgroup, `s_barrier` is a real barrier, LDS is shared memory and `requiredSubgroupSize` is available -- a frontier, not an opcode add. Corollary measured on the way: the live reject is at **pc=4**, the all-ones form, not at the pc 277-286 general-mask forms `STRAY_STATUS.md` named -- the all-ones pair is lowerable on its own and is disqualified by the general-mask forms further down, so the reject PC was again not the cause (#2481, #2790). | #3135, #3126, `STRAY_STATUS.md` |
| A vertex shader rejecting `s_mov_b32 exec_lo, -1` (`befe03c1`) is blocked on **plumbing the graphics wave size**, because the value's meaning depends on whether the wave is 32 or 64 lanes wide | **Falsified for the unnarrowed case, which is the one titles actually hit.** The dependence is real for a general value and for a restore from a NARROWED mask, and both still reject. It does not exist when every lane is already on: writing all-ones to the low dword then rewrites what that dword already held and leaves the high dword untouched, so the postcondition equals the precondition at 32 **and** 64 lanes. No wave width, no lane mapping, no peer lane. The prologue form was gating 19 of *Yakuza Kiwami*'s 20 programs and needed no wave-size contract at all. | #2872, `YAKUZA_JUDGMENT_BRINGUP.md` |
| A `[recompile-reject] mode=unresolved-operand` on an instruction with **no descriptor operand** -- a plain `s_add_u32 s0, lit, s0` -- means the resource table is incomplete | **Falsified.** `unresolved-operand` says only that the lowering exists and *some* operand did not resolve. It is equally raised when a scalar source has no representable dword in the per-invocation model, including because a **stale saved-mask alias** still claims the register. #2783's reject was on an instruction whose only SGPR source was recycled scratch, and no descriptor was involved at any point. Read *which* operand failed -- the reject line prints every source's kind -- before assuming the descriptor. | #2783 |
| GTA V's `unresolved-operand` rejects at **VCC reads** (`v_mov_b32 v1, vcc_lo`, `v_add_nc_u32 v0, vcc_lo, v0`) mean prosper's **VCC-as-scalar-scratch model is too narrow** — it admits VCC_LO scalar writes only through an enumerated packet list (`is_wave64_vcc_lo_scalar_cselect` (then named `is_gtav_…`), `is_wave64_vcc_lo_scalar_b32_candidate`, `b32_vcc_scalar_write`), and GTA V also writes VCC_LO with `s_lshl_b32`, `s_mulk_i32`, `s_add_i32`, `s_min_i32` and `v_readfirstlane_b32` | **Falsified — the reject is a symptom three instructions downstream of an unrelated cause.** Widening the write predicate to admit `s_lshl_b32`, extending it in the wave64 MUST dataflow, and adding a dual-domain admission all leave the terminal byte-identical, tested one at a time and then together. `operand_bits` was never the gap either: its `Special` case already reads `rs.sreg[106]` for 106..124. Instrumenting the dataflow gave the actual chain for `0x413d88400`: pc47 `s_mov_b32 s6, s14` makes s6 a MUST scalar word; pc337 **`s_bcnt1_i32_b64 s6, exec`** erases that fact, because `wave64_mask_reduction_source` deliberately returned −1 for architectural EXEC ("already resolved from architectural state by emit_alu"); pc351 `s_lshl_b32 vcc_lo, s6, 2` therefore has a non-scalar source, so VCC_LO's own scalar lifetime is dropped at the pc353 block boundary, and the failure finally surfaces at pc354 in a different register file. emit_alu *can* materialize the reduction, which is why the same packets compile fine inside one block — the coverage arm proving that had passed throughout. **A reject PC names where a fact was consumed, not where it was lost; instrument the MUST dataflow before widening any predicate at the reject site.** | #2481 |
| A synthetic kernel reproducing that shape (EXEC popcount, a guard, a consume past the merge) is enough to regression-test it | **Falsified.** Three synthetic shapes were built, including one whose guarded block contains an unpredicated scalar write live at the merge specifically to defeat `safe_execz_branches`. All three compiled on **both** sides of the fix: the structured/linearizing routes claim them before the CFG dispatcher — whose block-entry filter is the defect — ever runs. The exact production kernel plus its exact routed resource table **does** discriminate, and that is what `test_exec_population_count` pins; whether some smaller synthetic could also discriminate was not established, so read this row as "three attempts failed" rather than as a proof of minimality. | #2481 |
| GTA V's counted **`structured emission stopped` sites are an independent CFG family** that needs 28 separate structurizer fixes | **Falsified by program-tagged terminals and offline retries.** The message is a wrapper emitted after compact structured emission has already stopped at an earlier instruction/resource rejection. In the phase-anchored 28-tuple census every wrapper's `next-pc` matched the same invocation's earlier terminal PC; later exact fixes at `0x413cf6100`, `0x413cf5400`, `0x413e19200`, `0x413e1ac00`, and `0x413cf9200` removed the wrapper without any structurizer change. `0x413ce2a00` is the complementary positive control: compact route selection declines on a bottom-tested EXEC loop, but the generic dispatcher compiles it successfully, so its `backward else` line is route-selection noise rather than a skip. Attribute only program-tagged terminal records; stderr from concurrent shader compilations interleaves. | #2481 |
| GTA V `0x413dc6700` should receive a **fixed traversal-loop cap** to prevent the recurring RADV device loss | **Falsified as guest semantics.** Its sole backedge is the pc88..97 parent-link traversal and exits when `(parent_word >> 3) & 0x07ffffff` becomes zero; the shader contains no intrinsic numeric trip bound. Captured healthy `0x413ce6000 -> 0x413dc3400 -> 11 x 0x413dc6700` chains replay with an acyclic parent graph (observed maximum depth 6). The live guilty recovery follows a skipped/rejected `0x413ce6000` producer state, so a cap would conceal bad upstream data and truncate a valid deeper graph rather than implement RDNA2. Repair the producer/resource gap; keep the consumer loop exact. | #2481 |
| GTA V `0x413ce6000`'s **tag-7 selector records can be ignored** by the selected-SBUFFER domain classifier, because pc75 `V_CMPX_NE_U32 7,v0` removes them from EXEC | **Falsified by following the complete mask lifetime; reverted before commit.** pc69 saves the entering EXEC mask into `s[2:3]`; pc75 narrows EXEC only for the expensive pc77..138 calculation; the nested masks are restored at pc124/pc138; **pc139 `S_OR_B64 EXEC,s[2:3],EXEC` re-enables the entering lanes** before the descriptor waterfall, and pc142's CMPX-NE compares constant one against zero, so it does not remove them again. The focused regression mutating a pc70 record from tag 7 to tag 6 passed 1/1 — it proved the local classifier, not the shader's semantics — and a live route then rejected a fresh invalid record-4 state, which was the premise failing rather than the proof being incomplete. **A CMPX that narrows EXEC establishes nothing about a later consumer until its restore has been located.** | #2481 |
| GTA V `0x413ce6000` is blocked by its **`[compute-struct-reject]` control-flow terminal**, so clearing that terminal is on the path to its dispatches executing | **Falsified by measuring the emitted module rather than the diagnostic.** The program compiles **today**: `shader_inspect --stage compute` reports `status=ok spirv_dwords=13757`, and it emits the **byte-identical 13757 dwords with and without** a fix that clears the terminal entirely (`cf_rejected` 1→0, `structured_ifs` 0→6, `exact_wave` 0→1). The reject carries `role=route-decline` — the compact route declines and the generic dispatcher compiles it, exactly as the `structured emission stopped` row above says; that row was re-derived the hard way for this program. Its real blocker is **resource resolution**: `table_dependent=18`, every discovered resource carries `addr=0 size=0 srt=ffffffff sgpr=ffffffff`, and the live diagnostic is `[gta-selected-sbuffer] reject=selected-vsharp`. **Do not chase a compute CFG terminal without first checking whether the module is emitted anyway** — `spirv_dwords` answers it in one run. | #2481 |
| GTA V `0x413ce6000`'s dispatch state discriminates its failures — in particular `user_sgprs[10]`, which is 0 in all 94 failed realizations of the compute-only capture | **Falsified by finding a successful dispatch.** Two captures (`codex-ce6000-at1`, `codex-gta5-413dc6700-at1`) retain a **realized** ce6000 dispatch, and its entry SGPRs are `9cc963c0 00000020 00000000 001c0000 00000000 00005204 00000000 00080000 00000000 00005204 00000000` — **`s1..s10` byte-identical to the failures**, only the base pointer differs. `s10 == 0` in the success too, so it separates nothing and **no dispatch proof can be built on it**. The 94/94 was near-tautological besides: that population is `failure_diagnostics`, which is selected on the failure under investigation, so a healthy dispatch cannot appear in it. The discriminator is the **memory content** at `s[0:1]+0xa8`, and it is nondeterministic — identical routed runs give 30% and 61% success (#2516). | #2481, #2516 |
| GTA V `0x413ce6000`'s `[62,180)` interval — containing source pc70 and consumers pc153/156/158 — is **scalar-dead when `user_sgprs[10] == 0`**, because pc3 `S_LSHR_B32 s106,s10,2` yields zero, pc28 `S_CMP_EQ_U32 0,s106` sets SCC, pc35 `S_CBRANCH_SCC0 pc62` is not taken, and pc61 skips the interval | **Falsified by external disassembly. Two mnemonics in that chain are wrong, and correcting either one inverts the conclusion — the interval EXECUTES.** `llvm-mc -arch=amdgcn -mcpu=gfx1030 -disassemble` on the raw words gives `8f6a820a` → **`s_lshl_b32 vcc_lo, s10, 2`** (shift **left**, and the destination is VCC_LO) and `bf076a80` → **`s_cmp_lg_u32 0, vcc_lo`** (**not**-equal, opcode `0x07`; `0x06` is the equal form). So with `s10 == 0`: `vcc_lo = 0 << 2 = 0`, `SCC = (0 != 0) = false`, and pc35 `s_cbranch_scc0` is **TAKEN** to pc62. The dead interval is the other arm, pc36..61, which contains none of those four PCs. Nothing writes `s106` between pc3 and pc28 (every destination in that window enumerated with its width) and nothing writes SCC between pc28 and pc35 (`s_mov_b32` ×4, `s_bitset1_b32` ×2), so the chain is otherwise exactly as described. **Consequences: a dispatch proof pruning `[62,180)` would delete executing work behind a rigorous-looking object; pc70/153/156/158 are live and the current rejection is correct rather than a false positive; and the row below on raw selector-4 presence is NO LONGER SUPERSEDED — that hypothesis is open again.** The correct predicate is also `(s10 << 2) == 0`, not `>> 2`: these agree only at `s10 == 0` and a `s10 = 4` mutation arm cannot separate them (`4<<2` and `4>>2` are both nonzero) — use `s10 = 1` or `s10 = 0x40000000`. **The generalisable rule: prosper's own tables are downstream of the decoder that produced the listing you are reading, so they cannot check it. Disassemble the raw word with `llvm-mc` before recording a mnemonic in this table.** Three hand-derived semantics on this one program have now failed this way (pc75 tag-7, the probe zero, this); each was read rather than evaluated. | #2481, #2511 |
| A temporary two-bit `OpAtomicOr` probe emitted immediately before `0x413ce6000` pc153 **wrote `00000000`, therefore no surviving wave reaches pc153 in that dispatch** | **Conclusion true, derivation void — and that combination is why this row exists.** pc153 genuinely is not reached **in that failed 2,064-thread dispatch**, but the probe could not establish it: a zero from an opt-in *translated* diagnostic cannot distinguish "the guest never reached the site" from "the emitted diagnostic path did not execute", and the reading was withdrawn before the real cause (the scalar-dead `[62,180)` interval, row above) was found. The zero came first and was then interpreted together with the pc75 narrowing as support for the false tag-7 predicate. **Scope stays on the dispatch: ordinary valid `0x413ce6000` variants do execute that region, so nothing here licenses deleting pc153 generally.** And: **do not delete a guest access on the strength of an instrument that reports its own non-execution the same way it reports the guest's** — diagnose probe/emission behaviour first, then use the guest's own scalar state. | #2481 |
| GTA V `0x413cf9200` sees **stale root descriptors because source `0x413cf6100` was only a conservative dependency-graph producer, or capture replay lost overlapping aliases** | **Falsified by executing the producer and inspecting replay pointers.** Recovered `0x413cf6100` changes 2,128 dwords across 1,088 records of the shared 64-byte-stride arena, including every candidate root; its six decoded arena stores touch only record offsets +0/+4/+8/+12. Replay binds later root/subrange resources into the same mutable captured instance, so ordered execution propagates those writes correctly. What is stale is the **inspect-time resource dump**, which is a pre-dispatch snapshot; failed operations also never become graph consumers. Therefore `0x413cf9200` must inspect its current 224-byte root at command-ordered realization, not infer from the historical dump—but there is no producer-order or aliasing defect to fix. | #2481 |
| Modelling a register as holding **no value** is enough to contain it: an opaque token with no `sreg` entry, plus an `operand_bits` barrier, cannot leak into the data domain | **Falsified twice, by two different mechanisms, and both are structural rather than incidental.** (1) **Every CFG merge reads scalars through a `sget` lambda that renders an absent scalar as `uconst(0)` and stores the phi back into `rs.sreg`** (`rdna2_emit_cfg.cpp`, five sites), so a token established inside a branch arm comes out of the merge as an ordinary *tracked* zero -- `OpPhi %uint_0 ... -> OpIAdd` -- and, because the barrier is `token AND untracked`, the phi also disarms the barrier for the rest of the shader. The CFG dispatcher does the same one indirection further away: it stores an absent scalar into its Function variable **as zero** and reloads it as tracked data at every block entry. (2) **Dropping the register to plain untracked at the join does not fix it**, which is the trap worth remembering: an untracked ordinary SGPR is not an error in this model -- `operand_bits`' SGPR case ends in `return b.uconst(0)` with `ok` left true -- so "untracked" fabricates the same zero *silently*. Only the token itself is loud. So a value-absence marker needs a join rule that carries the marker (union, not intersection), a reject where a phi operand must exist anyway (loop headers), and the same treatment at the dispatcher's spill/reload. | #3133, #3136 |
| The CFG-dispatcher half of that containment cannot be pinned by a regression test. The dispatcher only ever *rejects*, and any program that reaches it also reaches the structured emitter, whose `join_entry_m0` supplies the same verdict -- so an arm built on it would pass with the dispatcher re-arm deleted, which is what the 324-test suite did when a reviewer disabled it in isolation | **Falsified -- one dispatcher entry has no fallback, and that is enough to isolate it.** `emit_body`'s `portable_compute_cfg_readlane` route calls `emit_cfg_state_machine` and `return false`s on a reject with no second attempt, so for a portable compute program carrying one forward SCC if plus a `v_readlane_b32` the dispatcher is the ONLY emitter: a compile can only be it accepting and a reject can only be it rejecting. `tests/gpu/recompiler/test_entry_m0_dispatcher.cpp` puts a save in one block and the read in the next on that route. With the re-arm, the program rejects and `last_terminal_reject_reason` reads `cfg-recompile-reject mode=unresolved-operand pc=6` -- a tag only `emit_cfg_state_machine` writes, naming the read rather than the save. With the two re-arm lines deleted (build rc=0) it COMPILES, no reason is recorded at all, and every other registered test stays green -- so the arm is the only thing in the tree that sees this. Three controls sit beside it, and the load-bearing one is the minimal pair: the same program with the save's DESTINATION changed to an unread register still compiles, which is what separates a register-precise re-arm from a dispatcher that refuses any stream containing a save. | #3203 |
| A **direct** user-data descriptor staged into a scratch SGPR range with `s_mov_b32` is unresolvable, so any title exhibiting the pattern is blocked by that gap | **The gap was real and is now closed; the inference from it was not.** `sreg_ud_alias` gives a direct descriptor the copy-provenance that `sreg_srt` already gave an indirect one, and on Earthion (`PPSA28061`) the recompiler now independently recovers `ud_alias=s9` — the exact `offset_dw` the front half declares. The draw is **still** declined, because the resource at that offset was dropped upstream by the degenerate-T# guard (`alias_res=null`, `(1 res)`), and the composite does not change (the route's three stable states are pixel-identical and no new content appears; a whole-run frame-identity count is **not** a valid oracle on this title — two runs of the same binary differ in 7 of 28 frames because a narration cycle's phase is nondeterministic). Two independent causes held one symptom in place. **Clearing a provenance decline is not evidence that anything draws: measure the composite (non-black percentage AND bounding box) separately, in both arms.** The consumption rule also must NOT re-check `!sreg_written` on the origin — a copy captures bits, and this shader reuses five of the origin words for its second descriptor before the sample; a synthesized copy arm that omits that reuse passes against the wrong rule. | #1773, #1590 |
| Nikoderiko's dropped 3D world is a **descriptor-provenance gap**: its 8-SGPR `SRSRC` range is *computed inside the shader*, so `sreg_range_written` is set and the user-data fallback is skipped | **Falsified**, and #1607 was retitled so nobody starts from it. Disassembly of 27 dumped failing stages shows every failing vertex stage uses the **canonical bindless-dynamic vertex fetch** — an ordinary table read whose index is wave-uniform and whose base is a user-data pointer, i.e. the exact shape `resolve_dynamic_fetch` exists to handle. `sreg_range_written` is a *symptom* of that legitimate reload, not its cause. **The three-step ladder is not the defect**; two upstream defects starve it. | #1607 |
| The Oregon Trail's black frame is a shader/recompiler gap | **Falsified.** On the default path there are **zero** `[recompile-reject]` lines: the draws are rejected earlier by the no-effect gate (`gpu_execute.hpp:968`), because the guest genuinely programs `CB_COLOR_CONTROL.MODE=0` on them. The MIMG gap that appears when the gate is bypassed with `PROSPER_FORCE_COLORWRITE=1` (#1634) is real and must be fixed, but the frame stays byte-identically black with the gate bypassed, so it is **not** established as the root cause. | #1606, #1634 |
| Syberia's black menu scene is caused by its 10 failing compute dispatches | **Not the cause.** The lit composite is already correct at draw 465, after all ten have been skipped. They remain FATAL gaps per `CLAUDE.md` and are tracked on #1628 with exact `pc`/`fmt`/`op` values. | #1619, #1628 |
| The recompiler's **R11G11B10 pack/unpack has an encoding defect that AMD hardware masks**, since its all-2048-code exactness sweep fails on lavapipe and passes on RADV | **Falsified by counting.** The earlier report used `std::mismatch`, which shows only the *first* difference while the assertion compares the whole vector — so it could not distinguish 1 failing texel from 92. A full census gives **92 differing texels, `all-finite=0`, `any-sNaN=92`, `quiet-model-exact=92`, `canon-model-exact=0`**: every difference carries a signalling NaN and **no all-finite texel differs**, so the finite encoding is exact. The cause is NaN *payload* quieting in `unpack_ufloat`'s `UnpackHalf2x16` (a real f16→f32 conversion; x86 `VCVTPH2PS` sets the quiet bit, RADV preserves the payload) — implementation-defined, and both conformant. Independently derivable offline: 1031 is odd mod 2¹¹ so the generator's G field is a bijection over all 2048 codes, and exactly 92 carry an sNaN with minimum 139. The assertion, not the recompiler, was wrong. | #1681, #1689 |
| Fragment **loop/EXEC control-flow lowering is wrong**, since 7 rendered-pixel assertions fail on lavapipe and pass on RADV | **Falsified — no pixel comparison ever happened.** Every failing draw read back the BLUE *clear*: the backend correctly refused it (`[render] skip draw: fragment shader requires subgroup size 64 (device range 8..8)`, 15 times across the two tests, identically on Mesa 25.2.8 and 26.1.4). The recompiler lowers a wave-wide EXECZ/VCCZ test to a native subgroup vote requiring exactly 64 lanes, and llvmpipe is fixed at 8. The assertions simply lacked the `supports_fragment_wave64_vote` gate their neighbours in the same files already used. | #1681, #1689 |
| The 16-bit **f16 -> u16/i16 converts rely on undefined behaviour**, because they hand an out-of-range float to `OpConvertFToU`/`OpConvertFToS` and clamp the *integer* result afterwards | **Withdrawn — the premise is false, and it was raised in a code review rather than measured.** `b.cvt_f2u` and `b.cvt_f2i` are not thin conversion wrappers: they are prosper's own saturating helpers (`rdna2_to_spirv.cpp`, #135/#686). `cvt_f2u` selects NaN to `0.0`, clamps to `[0, 4294967040]`, and only then converts, so the conversion operand is provably representable; `cvt_f2i` never emits `OpConvertFToS` at all — it converts the bounded magnitude through the unsigned path and restores the sign. So clamping the integer result is defined on every conforming implementation, and adding an outer float-domain clamp is a **behavioural no-op** that duplicates the NaN test and both rails on a hot path. Worked through for every input an f16 source can hold — finite, +/-0, +/-Inf, NaN — old and new forms agree on all of them. **Do not re-add the outer clamp**; the call site is the VOP1 `0x50`-`0x53` block in `src/gpu/recompiler/rdna2_to_spirv.cpp`, which carries a note explaining why it is unnecessary. | #2013, PR #2067 |
| **VOP3P `NEG_LO`/`NEG_HI` on a packed 16-bit INTEGER source is a two's-complement negation** (the reading an integer opcode invites) | **Falsified by the four live sites.** Every one of them — three shaders, two opcodes (`v_pk_add_u16`, `v_pk_max_i16`) — negates the *same* literal `0x00007fff` with `NEG_LO[0]` and `NEG_HI[0]` set together. Two's complement makes that `-32767`, which turns a `v_pk_max_i16` sitting beside a `v_min_i16 2` and a `v_pk_min_u16 1` into a no-op, and turns a gather base into nonsense. A sign-bit (bit 15) flip makes it `0xffff = -1`, which yields the `(x-1, y-1)` base of a 2x2 quad whose per-lane offsets are `+0/+2`, and a `max(x, -1)` beside `min(x, 1)`. It is also the same physical operation the packed **f16** path already models with `fneg` — a shared negate sitting between the operand read and the ALU. Implemented as the sign-bit flip; kernel 32r6 in `test_rdna2_to_spirv` asserts the exact word and fails under the two's-complement reading. `CONFIDENCE: MED` — the derivation above is live evidence; no published statement confirms it, and the discriminating input for any future contradiction is a negated source whose selected half is not `0x7fff`. **This was briefly raised to HIGH and then restored to MED — do not raise it again on the LLVM citation.** Two items were offered as published support in review of #2115 and neither holds: LLVM's AMDGPU modifier reference says of `neg_lo`/`neg_hi` *"This modifier is valid for floating-point operands only"*, which is a **restriction** and if anything cuts against applying the bit to an integer opcode — precisely the unpublished step MED hedges; and the "LLVM folds `xor 0x80008000` into `neg_lo`/`neg_hi` for integer packed ops" claim traces to llvm-project PR **#130234**, merged and **reverted the same day**, covering the **dot-product** family rather than `v_pk_add_u16`/`v_pk_max_i16`. Under this charter's evidence hierarchy that is a single secondary implementation — tier 4, hypothesis only. | #2013, PR #2115, PR #2123 |
| An inline **FLOAT** constant's contribution to the HIGH half of a packed 16-bit operand is **generation-dependent** — gfx11 duplicates it into `[31:16]`, gfx10.3 and older read zero | **The value is right; the generational premise is false, and that is the part worth recording.** Zero is correct on gfx10.3, so the packed-*integer* path (`half ? 0 : bits`) was right and the three f16 paths that replicated the constant and called the half select "a no-op for it" were wrong — fixed for `v_pk_*_f16`, `v_fma_mix_*` and the scalar f16 VOP3 `min3/max3/med3/fma` family. But **there is no gfx10-vs-gfx11 split to model**: `llvm-mc` folds `0x00003c00` to inline `1.0` and leaves `0x3c003c00` a 32-bit literal **identically on gfx1010, gfx1030, gfx1100 and gfx1200**. The fold is value-preserving, so inline `1.0` *is* `0x00003c00` in a packed f16 operand on every one of them. The integer side rules out "the assembler just never folds packed literals": `0x00000001` folds to `1` and `0xffffffff` to `-1`, while `0x00010001` does not. The real rule is simply that the packed operand **is the inline constant's whole 32-bit value** — which is why `-1` legitimately reads `0xffff` in *both* halves while `1.0` reads `0.0h` in the high one, and why the int and float cases never needed different code. The ISA states no replication rule and is consistent with this (VOP3P pseudocode reads `S0[31:16]`; the inline table gives a half constant as a 16-bit pattern). **Do not add a gfx-version predicate here.** Kernels `32r14a`-`32r14e` in `test_rdna2_to_spirv` pin the high-half value of each fixed path and fail under the select-ignoring reading. **Scope: this settles the rule for VOP3/VOP3P `OPSEL` only, and every OPSEL site now shares the helper. Six SDWA sites still make the select a no-op for an inline constant — SDWA's sub-dword select is a different encoding this evidence does not reach, tracked on #2191.** `CONFIDENCE: HIGH`. | #2119, PR #2177 |
| The RDNA 2 ISA guide's **summary tables 58 and 60** give the VOPC opcode layout — so `v_cmpx_*_f32` is at `0x50`-`0x5f` and `v_cmpx_*_i16` at `0xb0`-`0xb7` | **Falsified — those two tables are stale GCN-era boilerplate and contradict the same document.** They also list `V_CMPS`/`V_CMPSX` opcodes, which do not exist on RDNA2 at all, and they place compares in `0x40`-`0x7f`, where llvm-mc reports **64 invalid encodings**. The authority is the per-opcode **table 61**, which agrees with hardware: disassembling all 256 VOPC e32 encodings with `llvm-mc -mcpu=gfx1030` gives `v_cmpx` at `0x10`-`0x1f`, `0x30`-`0x3f`, `0x90`-`0x9f`, `0xb0`-`0xbe`, `0xd0`-`0xdf`, `0xf0`-`0xff`, each exactly its `v_cmp` counterpart **+ 0x10**. The `0xaf`/`0xbf` holes are confirmed twice: llvm-mc rejects both, and table 61 **skips opcodes 175 and 191** (running 174 -> 176 and 190 -> 192). The map now lives once, at `vopc_is_cmpx` in `rdna2_decode.cpp`, with boundary assertions on both sides of every window in `test_rdna2_decode`. **Do not "correct" it back to tables 58/60.** **The windows were duplicated in FOUR places, and the copies were not merely untidy — one was a silent miscompile.** `dead_wave_mask_writes` excludes cmpx from VCC definitions (a cmpx writes EXEC and has no VCC destination), but its private copy listed three of six windows, and the decoder gives every VOPC e32 `dst = 106` (VCC_LO) — so an unrecognised cmpx recorded a **phantom VCC definition**, a preceding live `s_and_b64`/`s_andn2_b64 vcc` looked overwritten before use, was classified dead and **elided**, and the real consumer read stale VCC with no diagnostic. Kernel `32r13v` reproduces it (`0x22220000` before, `0x11110000` after) with `32r13w` as a control on an already-covered window. A fourth copy in `gpu_executor.cpp`'s `changes_exec` under-reported EXEC writers to a dominance proof. All four now call the one predicate. | #2120, PR #2181 |
| Sonic Racing: CrossWorlds' failing vertex fetch is a **runtime-SELECTED descriptor the fold cannot model** — `s_load_dwordx4` a V#, then `s_cselect_b32` word 3 on a null-pointer test immediately before the fetch, so on one arm the descriptor "is in no table" | **Falsified by measuring values instead of reading opcodes.** The fold already models that entire idiom — `s_bfe_u64`, `s_or_b32`, `s_cmp_eq_u32` and `s_cselect_b32` are all implemented, the last explicitly as *"the vertex-fetch format patch's tail"*. It never reaches them: `PROSPER_DYNTRACE_FAIL=1` shows `s[18:19]` reading back **null** out of a constant buffer the fold reads correctly and in range, so the dereference fails and both the V#'s SOFFSET and its word-3 select are unknown from there down. Real guest data, correctly read, containing a null bindless-table pointer — not a modelling gap. **The generalisable trap: a fold that stops early looks IDENTICAL whether it ran out of modelled opcodes or out of readable data.** The `[recompile-reject]` / `[*-unresolved]` line is the same either way, and a disassembler can only answer the first question. When the chain involves memory, reach for the instrument that reports the chain — `PROSPER_DYNTRACE_FAIL=1` prints `base_ok`/`soff_ok`/`unreadable` per step — before hypothesising about opcodes. The static recon that produced this hypothesis marked its mechanics `CONFIDENCE: HIGH` (they held) and its inference `MED` (it did not), and named "which arm is taken" as the first thing to measure; measuring it is what killed the hypothesis. | #2013, #2131, #2132 |
| *Sonic Frontiers*' stage compute programs rejecting on `s_or_b32 vcc_lo, scc, vcc_hi` (`886a6bfd`) are stopped by the **deliberate 64-bit-mask SCC poison** in `rdna2_emit_alu.cpp` — a preceding wave-mask op wrote `SCC=(mask != 0)`, a cross-lane reduction the per-invocation model cannot form | **Falsified by reading `rs.scc` at the rejecting instruction on a live routed arm.** At every one of the five sites `rs.scc` is a **live SSA id** (456 / 605 / 718 / 1478 on four programs) and *both* operands already carry scalar data — nothing is poisoned, and the SCC there was written two instructions earlier by an ordinary `s_cmp_eq_u32 <imm>, sN`. The gate that fails is the Wave64 placeholder guard: `scalar_presence_has_no_placeholders=0` and the pc absent from `vcc_b32_scalar_result_pcs`. The lost fact is **three hops upstream**: the same physical pair is written 12 dwords (9 instructions) earlier by `s_or_b32 vcc_lo, vcc_hi, scc`, which — precisely *because* both dwords are proved scalar — is classified `b32_vcc_complete_scalar_pair` and therefore a mask write, and the SOP2 MUST transfer's `mask_write < 0` term then charged that dual-domain write an SCC loss. That made the following `s_cselect_b32 vcc_hi,1,0`'s `valid_scc_read` false, VCC_HI lost its scalar-word fact, and the consumer became uncertifiable. Removing only that term clears the SCC site on **all five**; **two** of them then execute (measured as skipped=0 at an unchanged 32-program census header) and the other **three** stop at unrelated FATAL gaps — `image_get_resinfo` (MIMG) and a `v_cvt_u32_f32_sdwa`-family VOP1, one of them 331 instructions further on. The poison is untouched and still rejects a genuine B64-mask SCC (regression arms in `test_rdna2_to_spirv`). **The composite did not change**: a live 3840x2160 arm on the same route still shows the HUD-only frame at the same bounding box. **A reject PC names where a fact was consumed, not where it was lost** — the #2481 row above, re-derived on a different title, in a different dataflow. | #2790 |
| *Sonic Frontiers*' three **scene-target-width** stage programs are stopped by `s_cbranch_execz` and `image_load_mip` — the two encodings their `[recompile-reject]` lines name (#2790's handoff) | **Half falsified, and the half that survives is not what the reject lines said.** Both reject PCs were reported by the STRAIGHT-LINE emitter, which those programs only reach after two earlier routes decline. Live, with `PROSPER_DBG_PROGRAM` pointed at each address: the compact structurizer declines `backward else` (`role=route-decline`, ordinary route selection — see the `structured emission stopped` row above), the CFG dispatcher is then the intended route, and *it* declined `wave64-ambiguous-mask-read` at pc481 / pc481 / pc471. The cause was one missing entry in `scalar_alu_source_words`: **V_LSHL_ADD_U32 fell to the VOP3 fail-closed `return 2` default**, so `v_lshl_add_u32 v7, v6, 2, vcc_lo` — a 32-bit read of VCC_LO used as ordinary scalar scratch, byte-identical in all three — was charged the whole pair and demanded VCC_HI, a live mask half at that join. With the entry added, that decline occurs **0 times** in the routed run, all three reach the dispatcher body, and **all three converge on ONE remaining gap**: `image_load_mip` on a 12-mip `2D_ARRAY` resource addressed `dim:2D` (`shape=0 proven_zero_mip=0 img_dim=5/1 samples=1 mips=12 mip_tail=0 compressed=0`). So **`s_cbranch_execz` is dead as a lever for this title** — the dispatcher lowers that control flow once the MUST analysis passes — and `image_load_mip` is the single live one (#2818). The census did not move (`262144 dispatch decisions over 32 program(s)`, 14 listed, all `executed=0`, before and after) and **the composite did not change**. Third instance on this title of "a reject PC names where a fact was consumed, not where it was lost" (#2481, #2801). | #2790 |
| The stages failing #305's user-data condition indicate a recompiler defect | **No — the recompiler is behaving correctly and fail-visibly.** When the seeded user-data block is wrong, the const-fold refuses to invent a descriptor (`SOFFSET untracked -> fetch left unresolved (not folded to 0)`) and the draw is skipped rather than drawing garbage. The defect is upstream, in graphics register state — see [`RESOURCE_BINDING.md`](RESOURCE_BINDING.md) § `Ruled out`. | #305, #1607 |
| A fragment module whose **only** wave-width reason is `kFragmentWaveReasonWaveAny` is width-agnostic, so on a device that cannot supply the guest wave (NVIDIA 32..32, llvmpipe 8..8) the draw can be recovered by pinning `requiredSubgroupSize` to the device width instead of skipping it | **Falsified three times; do not attempt a fourth.** Two distinct defects, and the second survives a correct fix for the first. (1) **The reason bit does not mean "guard".** It records that a vote was *emitted*, not that its result is only branched on. `fragment_wave_any`'s result is assigned to `rs.scc` at `rdna2_emit_alu.cpp:615`, `:1069`, `:1627`, `:2558` and `rdna2_emit_cfg.cpp:4847`, and SCC is then consumed as scalar **data** by `s_cselect_b32` (`:2241`), `s_cselect_b64` (`:1764`/`:1766`) and the `s_addc`/`s_subb` carry-in (`:2211`/`:2220`); of eleven `fragment_wave_any` call sites, three are guards and seven consume the vote as a guest scalar. The `#2418` re-arm at `:1069` says so in its own comment: *"a REDUCE in #2410's taxonomy [...] must not be relaxed"*. (2) **Even a guard-only predicate is insufficient**, which is the half that keeps being missed: #2410 added a dedicated `WaveAnyReduce` bit so the predicate was guard-only *by construction*, and it was still unsound — a guarded block that writes emulated scalar state, restores EXEC, then reads that scalar past the merge leaves the non-entering half-wave **stale**, which is *DOLL*'s FXAA shape (#273), a real shipped shader. Safety needs "the guarded block has no scalar side effect observable after the merge", a property of the *block* that the reason-bit mechanism cannot express: **the gate is the wrong shape, not merely mis-tuned.** #2414 adds that the three candidate guard sites all resolve a wave-wide scalar branch condition, so there is no guard among them anyway. **The defence that catches this** is `test_recompiled_fragment.cpp`'s fail-visible arms — `skipped_wave64_draw` at lines 116, 511, 549, 568, 601, 631, 662, 722 and 772 assert the BLUE *clear* wherever `supports_fragment_wave64_vote` is false, so any relaxation makes those fixtures render and the arms go red (`backedge-break`, EXECZ branch, `mask-compare`, `direct-break`, `divergent-loop`, `VCCZ-loop`). #2404 was rejected in review, #2410 failed CI twice, #2984 was rejected in review. **The route that does work** is composing the two 32-wide subgroup votes into one 64-lane answer (#2414 option 1, #2429) — real wave64 emulation, independent of what the device exposes. Guard-vs-reduce can be settled per shader with `gpu_replay --dump-shader <draw>:fs` plus `spirv-dis`. | #2404, #2410, #2414, #2984 |
| Stray's rejected `IMAGE_LOAD_MIP` NSA-2D fragment op (#3134) can be admitted the same way `IMAGE_STORE_MIP`'s NSA-2D sibling was — widen `rdna2_mimg_zero_mip_shape()` to recognize the packet and let the existing "prove the mip VGPR is zero, then discard it" fast path handle it | **Half right, half falsified — the packet recognition generalizes, the fast path does not apply.** The address decode *is* exactly the sibling shape, byte-verified via llvm-mc gfx1030 (`image_load_mip v[5:7], [v0, v42, v5], s[32:39] dmask:0x7 dim:SQ_RSRC_IMG_2D`), and the widened `rdna2_mimg_zero_mip_shape()` correctly reports `shape=1`/`mip_vgpr=5` for it. But on the live F9 capture (`s14072.prgcap`, fragment `0x30be800000`, pc=134) the resource it addresses genuinely has **six real mip levels** and the mip VGPR is **not provably zero** (`[mimg-mip] ... shape=1 proven_zero_mip=0 img_dim=1/1 mips=6 mip_tail=1 dataformat=7(Uint16) ncomp=2`) — unlike every previously-evidenced `IMAGE_LOAD_MIP`/`IMAGE_STORE_MIP` shape, which all address genuinely single-materialized-mip resources. This is the *identical* root cause #2818 already found for *Sonic Frontiers*' compute-side `IMAGE_LOAD_MIP` (12-mip 2D_ARRAY, `mip_tail=0`) — the guest's own per-level bytes are never uploaded; prosper's graphics texture path only *generates* a mip chain via a `vkCmdBlitImage` downsample cascade, gated to `VK_FORMAT_R8G8B8A8_UNORM` (`tests/fixtures/render_runner.h`, `generated_mip_format_supported`), which this `R16G16_UINT`-shaped resource cannot even use — an **integer** format has no meaningful linear-filtered downsample, and Vulkan does not advertise `SAMPLED_IMAGE_FILTER_LINEAR` for pure-integer formats regardless. So the shape-widening alone changes the diagnostic (`shape=0`->`shape=1`) but not the outcome: the emitter's `res->declared_mip_levels != 1u` / `!res->proven_zero_mip` gates correctly keep rejecting it, and per #2818's own rule — forcing `Lod=0` or reading a blit-synthesized chain would both render a **wrong** image, worse than the current visible reject. A faithful fix needs the shared guest-authored multi-level upload infrastructure #2818 scoped and parked (steps 2-3: real `mipLevels` on the image, a per-level copy from the guest's own tiled bytes, and an `OpImageFetch` with an explicit `Lod` carrying the mip VGPR) — out of scope for a single-title NSA-shape fix. | #3134, #2818 |
| In **Wave64**, a scalar overwrite of the physical VCC pair is what leaves `RegState::vcc` absent, so a program whose entry VCC is missing can be built by writing `vcc_lo` as data | **False, and it cost a whole fixture iteration on #3231.** `record_scalar_write`'s `if (reg == 106) rs.vcc = 0;` sits inside a loop guarded by `rs.sreg_bool_b32.contains(reg)`, and `sreg_bool_b32` is the **Wave32** one-word-mask domain (`b.allow_b32_masks = wave_size == 32`) — empty in Wave64. So `s_mov_b32 vcc_lo, 5`, an SMEM load into `s[106:107]`, and SOP2 arithmetic into a VCC word all leave `rs.vcc` untouched in a Wave64 compute program; the data/mask question is answered downstream by the block-entry `wave64_b64_mask_in` / `wave64_scalar_word_in` MUST sets instead. Where Wave64 actually loses the value is on the way **OUT of a dispatcher region**: `emit_cfg_state_machine` rebuilds its caller's `RegState` with `load_state()`, which republishes VCC only when the terminal block's `wave64_b64_mask_in` still contains s106. That is why *Astro Bot*'s `0x500571000` reaches its final barrier phase with no entry VCC — two earlier phases each exited through the dispatcher — and why a synthetic reproduction needs a **preceding dispatcher region**, not a scalar write. `tests/gpu/recompiler/test_entry_vcc_dead.cpp` is built that way. | #3231 |
| The `IMAGE_LOAD_MIP` NSA packet is a **different operation** from the consecutive-VGPR form, so lowering it needs its own emitter path | **Falsified — it is the same operation in a different ADDRESS ENCODING, and the expected-address-count rule was the defect.** RDNA 2 MIMG's NSA field (dword0[2:1]) says only how many EXTRA dwords carry addresses: addr0 stays in VADDR and every later address names an arbitrary VGPR, one per byte, low byte first. Assembling the shapes by hand through llvm-mc gfx1030 shows the address VECTOR is identical either way — `image_load_mip v[5:7], [v0, v42, v5], s[32:39] dmask:0x7 dim:SQ_RSRC_IMG_2D` and `image_load_mip v[5:7], v[0:2], …` are both `[x, y, mip]`, and the 2D_ARRAY forms are both `[x, y, slice, mip]`. The emitter's coordinate gather already read NSA generically; the only thing that did not was `rdna2_mimg_dynamic_mip_shape`, which derived the mip register as `VADDR + n` and so had to decline rather than name the wrong VGPR. It now reads the encoding, and the NSA form lowers through exactly the #3048 path the consecutive form does — with an executed arm whose mip lives in **v7** while **v2** (the register the old arithmetic would have named) holds zero, so a consecutive reading fetches level zero and reddens. Separately, `shader_resource_mip_chain_plan` refused any resource whose SELECTED level is packed in the shared mip tail; that refused the whole small-texture class rather than a corner (a 32x32 four-byte pyramid is smaller than one 64 KiB macroblock, so **every** level of it is in the tail), and it is now admitted after proving the plan's in-block coordinates equal the ones the descriptor decode published. | #3134 |
| A `cfg-recompile-reject mode=unresolved-operand` naming a VOP3 opcode means that opcode's lowering is missing — so `op=0x346` on Stray's compute `0x300e390000` is an unimplemented `v_lshl_add_u32` | **Falsified.** The emitter has existed at `rdna2_emit_alu.cpp:4844` throughout, and the opcode is already on `rdna2_cfg_support.hpp:1304`'s B32 source-width allowlist; the mode field was telling the truth. The unresolvable operand was `s14`, on which the CFG dispatcher's #3133 entry-M0 re-arm had stamped a token at the **entry block** because its MAY set was whole-stream — for a `s_mov_b32 s14, m0` 153 dwords **later** in the same program, which s14 also serves as workgroup-id X. So the reject PC named the consumer and the cause was ahead of it in program order, the reverse of the usual direction. Reproduced offline with `shader_inspect` and fixed by an entry-rooted forward MAY dataflow. | #3308, `docs/STRAY_STATUS.md` |

**Genuinely open on the recompiler side** (do not confuse with the above): the const-fold is
**path-insensitive**. A PC reachable only via a taken branch inherits the scalar state left by the
mutually-exclusive fall-through arm, so `resolve_dynamic_fetch` can apply a load that provably cannot
execute on that path. Affects 1 of 13 traced Nikoderiko vertex stages. The proposed generic contract:
a PC not reachable by fall-through begins a block whose scalar state is its branch predecessors'
state; with exactly one recorded forward branch targeting it, adopt that branch's captured state;
with more than one, or with a backward edge, **invalidate** every scalar value, descriptor snapshot
and provenance tag rather than continuing with a provably impossible state. The second half is a
tightening — today the fold silently believes a wrong-arm value, which is the invented-descriptor
failure mode in disguise. Carries cross-title regression risk; needs snapshot coverage. (#1607)
