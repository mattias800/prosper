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

## Ruled out

Cross-title falsifications where the **recompiler was blamed and exonerated**. One line per dead
hypothesis, the evidence that killed it, and where that evidence lives. Do not re-derive these
without contradictory new evidence.

| Hypothesis | Verdict and evidence | Source |
|---|---|---|
| GTA V's counted **`structured emission stopped` sites are an independent CFG family** that needs 28 separate structurizer fixes | **Falsified by program-tagged terminals and offline retries.** The message is a wrapper emitted after compact structured emission has already stopped at an earlier instruction/resource rejection. In the phase-anchored 28-tuple census every wrapper's `next-pc` matched the same invocation's earlier terminal PC; later exact fixes at `0x413cf6100`, `0x413cf5400`, `0x413e19200`, `0x413e1ac00`, and `0x413cf9200` removed the wrapper without any structurizer change. `0x413ce2a00` is the complementary positive control: compact route selection declines on a bottom-tested EXEC loop, but the generic dispatcher compiles it successfully, so its `backward else` line is route-selection noise rather than a skip. Attribute only program-tagged terminal records; stderr from concurrent shader compilations interleaves. | #2481 |
| GTA V `0x413dc6700` should receive a **fixed traversal-loop cap** to prevent the recurring RADV device loss | **Falsified as guest semantics.** Its sole backedge is the pc88..97 parent-link traversal and exits when `(parent_word >> 3) & 0x07ffffff` becomes zero; the shader contains no intrinsic numeric trip bound. Captured healthy `0x413ce6000 -> 0x413dc3400 -> 11 x 0x413dc6700` chains replay with an acyclic parent graph (observed maximum depth 6). The live guilty recovery follows a skipped/rejected `0x413ce6000` producer state, so a cap would conceal bad upstream data and truncate a valid deeper graph rather than implement RDNA2. Repair the producer/resource gap; keep the consumer loop exact. | #2481 |
| GTA V `0x413ce6000`'s **tag-7 selector records can be ignored** by the selected-SBUFFER domain classifier, because pc75 `V_CMPX_NE_U32 7,v0` removes them from EXEC | **Falsified by following the complete mask lifetime; reverted before commit.** pc69 saves the entering EXEC mask into `s[2:3]`; pc75 narrows EXEC only for the expensive pc77..138 calculation; the nested masks are restored at pc124/pc138; **pc139 `S_OR_B64 EXEC,s[2:3],EXEC` re-enables the entering lanes** before the descriptor waterfall, and pc142's CMPX-NE compares constant one against zero, so it does not remove them again. The focused regression mutating a pc70 record from tag 7 to tag 6 passed 1/1 — it proved the local classifier, not the shader's semantics — and a live route then rejected a fresh invalid record-4 state, which was the premise failing rather than the proof being incomplete. **A CMPX that narrows EXEC establishes nothing about a later consumer until its restore has been located.** | #2481 |
| GTA V `0x413ce6000` is blocked by its **`[compute-struct-reject]` control-flow terminal**, so clearing that terminal is on the path to its dispatches executing | **Falsified by measuring the emitted module rather than the diagnostic.** The program compiles **today**: `shader_inspect --stage compute` reports `status=ok spirv_dwords=13757`, and it emits the **byte-identical 13757 dwords with and without** a fix that clears the terminal entirely (`cf_rejected` 1→0, `structured_ifs` 0→6, `exact_wave` 0→1). The reject carries `role=route-decline` — the compact route declines and the generic dispatcher compiles it, exactly as the `structured emission stopped` row above says; that row was re-derived the hard way for this program. Its real blocker is **resource resolution**: `table_dependent=18`, every discovered resource carries `addr=0 size=0 srt=ffffffff sgpr=ffffffff`, and the live diagnostic is `[gta-selected-sbuffer] reject=selected-vsharp`. **Do not chase a compute CFG terminal without first checking whether the module is emitted anyway** — `spirv_dwords` answers it in one run. | #2481 |
| GTA V `0x413ce6000`'s dispatch state discriminates its failures — in particular `user_sgprs[10]`, which is 0 in all 94 failed realizations of the compute-only capture | **Falsified by finding a successful dispatch.** Two captures (`codex-ce6000-at1`, `codex-gta5-413dc6700-at1`) retain a **realized** ce6000 dispatch, and its entry SGPRs are `9cc963c0 00000020 00000000 001c0000 00000000 00005204 00000000 00080000 00000000 00005204 00000000` — **`s1..s10` byte-identical to the failures**, only the base pointer differs. `s10 == 0` in the success too, so it separates nothing and **no dispatch proof can be built on it**. The 94/94 was near-tautological besides: that population is `failure_diagnostics`, which is selected on the failure under investigation, so a healthy dispatch cannot appear in it. The discriminator is the **memory content** at `s[0:1]+0xa8`, and it is nondeterministic — identical routed runs give 30% and 61% success (#2516). | #2481, #2516 |
| GTA V `0x413ce6000`'s `[62,180)` interval — containing source pc70 and consumers pc153/156/158 — is **scalar-dead when `user_sgprs[10] == 0`**, because pc3 `S_LSHR_B32 s106,s10,2` yields zero, pc28 `S_CMP_EQ_U32 0,s106` sets SCC, pc35 `S_CBRANCH_SCC0 pc62` is not taken, and pc61 skips the interval | **Falsified by external disassembly. Two mnemonics in that chain are wrong, and correcting either one inverts the conclusion — the interval EXECUTES.** `llvm-mc -arch=amdgcn -mcpu=gfx1030 -disassemble` on the raw words gives `8f6a820a` → **`s_lshl_b32 vcc_lo, s10, 2`** (shift **left**, and the destination is VCC_LO) and `bf076a80` → **`s_cmp_lg_u32 0, vcc_lo`** (**not**-equal, opcode `0x07`; `0x06` is the equal form). So with `s10 == 0`: `vcc_lo = 0 << 2 = 0`, `SCC = (0 != 0) = false`, and pc35 `s_cbranch_scc0` is **TAKEN** to pc62. The dead interval is the other arm, pc36..61, which contains none of those four PCs. Nothing writes `s106` between pc3 and pc28 (every destination in that window enumerated with its width) and nothing writes SCC between pc28 and pc35 (`s_mov_b32` ×4, `s_bitset1_b32` ×2), so the chain is otherwise exactly as described. **Consequences: a dispatch proof pruning `[62,180)` would delete executing work behind a rigorous-looking object; pc70/153/156/158 are live and the current rejection is correct rather than a false positive; and the row below on raw selector-4 presence is NO LONGER SUPERSEDED — that hypothesis is open again.** The correct predicate is also `(s10 << 2) == 0`, not `>> 2`: these agree only at `s10 == 0` and a `s10 = 4` mutation arm cannot separate them (`4<<2` and `4>>2` are both nonzero) — use `s10 = 1` or `s10 = 0x40000000`. **The generalisable rule: prosper's own tables are downstream of the decoder that produced the listing you are reading, so they cannot check it. Disassemble the raw word with `llvm-mc` before recording a mnemonic in this table.** Three hand-derived semantics on this one program have now failed this way (pc75 tag-7, the probe zero, this); each was read rather than evaluated. | #2481, #2511 |
| A temporary two-bit `OpAtomicOr` probe emitted immediately before `0x413ce6000` pc153 **wrote `00000000`, therefore no surviving wave reaches pc153 in that dispatch** | **Conclusion true, derivation void — and that combination is why this row exists.** pc153 genuinely is not reached **in that failed 2,064-thread dispatch**, but the probe could not establish it: a zero from an opt-in *translated* diagnostic cannot distinguish "the guest never reached the site" from "the emitted diagnostic path did not execute", and the reading was withdrawn before the real cause (the scalar-dead `[62,180)` interval, row above) was found. The zero came first and was then interpreted together with the pc75 narrowing as support for the false tag-7 predicate. **Scope stays on the dispatch: ordinary valid `0x413ce6000` variants do execute that region, so nothing here licenses deleting pc153 generally.** And: **do not delete a guest access on the strength of an instrument that reports its own non-execution the same way it reports the guest's** — diagnose probe/emission behaviour first, then use the guest's own scalar state. | #2481 |
| GTA V `0x413cf9200` sees **stale root descriptors because source `0x413cf6100` was only a conservative dependency-graph producer, or capture replay lost overlapping aliases** | **Falsified by executing the producer and inspecting replay pointers.** Recovered `0x413cf6100` changes 2,128 dwords across 1,088 records of the shared 64-byte-stride arena, including every candidate root; its six decoded arena stores touch only record offsets +0/+4/+8/+12. Replay binds later root/subrange resources into the same mutable captured instance, so ordered execution propagates those writes correctly. What is stale is the **inspect-time resource dump**, which is a pre-dispatch snapshot; failed operations also never become graph consumers. Therefore `0x413cf9200` must inspect its current 224-byte root at command-ordered realization, not infer from the historical dump—but there is no producer-order or aliasing defect to fix. | #2481 |
| Nikoderiko's dropped 3D world is a **descriptor-provenance gap**: its 8-SGPR `SRSRC` range is *computed inside the shader*, so `sreg_range_written` is set and the user-data fallback is skipped | **Falsified**, and #1607 was retitled so nobody starts from it. Disassembly of 27 dumped failing stages shows every failing vertex stage uses the **canonical bindless-dynamic vertex fetch** — an ordinary table read whose index is wave-uniform and whose base is a user-data pointer, i.e. the exact shape `resolve_dynamic_fetch` exists to handle. `sreg_range_written` is a *symptom* of that legitimate reload, not its cause. **The three-step ladder is not the defect**; two upstream defects starve it. | #1607 |
| The Oregon Trail's black frame is a shader/recompiler gap | **Falsified.** On the default path there are **zero** `[recompile-reject]` lines: the draws are rejected earlier by the no-effect gate (`gpu_execute.hpp:968`), because the guest genuinely programs `CB_COLOR_CONTROL.MODE=0` on them. The MIMG gap that appears when the gate is bypassed with `PROSPER_FORCE_COLORWRITE=1` (#1634) is real and must be fixed, but the frame stays byte-identically black with the gate bypassed, so it is **not** established as the root cause. | #1606, #1634 |
| Syberia's black menu scene is caused by its 10 failing compute dispatches | **Not the cause.** The lit composite is already correct at draw 465, after all ten have been skipped. They remain FATAL gaps per `CLAUDE.md` and are tracked on #1628 with exact `pc`/`fmt`/`op` values. | #1619, #1628 |
| The recompiler's **R11G11B10 pack/unpack has an encoding defect that AMD hardware masks**, since its all-2048-code exactness sweep fails on lavapipe and passes on RADV | **Falsified by counting.** The earlier report used `std::mismatch`, which shows only the *first* difference while the assertion compares the whole vector — so it could not distinguish 1 failing texel from 92. A full census gives **92 differing texels, `all-finite=0`, `any-sNaN=92`, `quiet-model-exact=92`, `canon-model-exact=0`**: every difference carries a signalling NaN and **no all-finite texel differs**, so the finite encoding is exact. The cause is NaN *payload* quieting in `unpack_ufloat`'s `UnpackHalf2x16` (a real f16→f32 conversion; x86 `VCVTPH2PS` sets the quiet bit, RADV preserves the payload) — implementation-defined, and both conformant. Independently derivable offline: 1031 is odd mod 2¹¹ so the generator's G field is a bijection over all 2048 codes, and exactly 92 carry an sNaN with minimum 139. The assertion, not the recompiler, was wrong. | #1681, #1689 |
| Fragment **loop/EXEC control-flow lowering is wrong**, since 7 rendered-pixel assertions fail on lavapipe and pass on RADV | **Falsified — no pixel comparison ever happened.** Every failing draw read back the BLUE *clear*: the backend correctly refused it (`[render] skip draw: fragment shader requires subgroup size 64 (device range 8..8)`, 15 times across the two tests, identically on Mesa 25.2.8 and 26.1.4). The recompiler lowers a wave-wide EXECZ/VCCZ test to a native subgroup vote requiring exactly 64 lanes, and llvmpipe is fixed at 8. The assertions simply lacked the `supports_fragment_wave64_vote` gate their neighbours in the same files already used. | #1681, #1689 |
| The 16-bit **f16 -> u16/i16 converts rely on undefined behaviour**, because they hand an out-of-range float to `OpConvertFToU`/`OpConvertFToS` and clamp the *integer* result afterwards | **Withdrawn — the premise is false, and it was raised in a code review rather than measured.** `b.cvt_f2u` and `b.cvt_f2i` are not thin conversion wrappers: they are prosper's own saturating helpers (`rdna2_to_spirv.cpp`, #135/#686). `cvt_f2u` selects NaN to `0.0`, clamps to `[0, 4294967040]`, and only then converts, so the conversion operand is provably representable; `cvt_f2i` never emits `OpConvertFToS` at all — it converts the bounded magnitude through the unsigned path and restores the sign. So clamping the integer result is defined on every conforming implementation, and adding an outer float-domain clamp is a **behavioural no-op** that duplicates the NaN test and both rails on a hot path. Worked through for every input an f16 source can hold — finite, +/-0, +/-Inf, NaN — old and new forms agree on all of them. **Do not re-add the outer clamp**; the call site is the VOP1 `0x50`-`0x53` block in `src/gpu/rdna2_to_spirv.cpp`, which carries a note explaining why it is unnecessary. | #2013, PR #2067 |
| **VOP3P `NEG_LO`/`NEG_HI` on a packed 16-bit INTEGER source is a two's-complement negation** (the reading an integer opcode invites) | **Falsified by the four live sites.** Every one of them — three shaders, two opcodes (`v_pk_add_u16`, `v_pk_max_i16`) — negates the *same* literal `0x00007fff` with `NEG_LO[0]` and `NEG_HI[0]` set together. Two's complement makes that `-32767`, which turns a `v_pk_max_i16` sitting beside a `v_min_i16 2` and a `v_pk_min_u16 1` into a no-op, and turns a gather base into nonsense. A sign-bit (bit 15) flip makes it `0xffff = -1`, which yields the `(x-1, y-1)` base of a 2x2 quad whose per-lane offsets are `+0/+2`, and a `max(x, -1)` beside `min(x, 1)`. It is also the same physical operation the packed **f16** path already models with `fneg` — a shared negate sitting between the operand read and the ALU. Implemented as the sign-bit flip; kernel 32r6 in `test_rdna2_to_spirv` asserts the exact word and fails under the two's-complement reading. `CONFIDENCE: MED` — the derivation above is live evidence; no published statement confirms it, and the discriminating input for any future contradiction is a negated source whose selected half is not `0x7fff`. **This was briefly raised to HIGH and then restored to MED — do not raise it again on the LLVM citation.** Two items were offered as published support in review of #2115 and neither holds: LLVM's AMDGPU modifier reference says of `neg_lo`/`neg_hi` *"This modifier is valid for floating-point operands only"*, which is a **restriction** and if anything cuts against applying the bit to an integer opcode — precisely the unpublished step MED hedges; and the "LLVM folds `xor 0x80008000` into `neg_lo`/`neg_hi` for integer packed ops" claim traces to llvm-project PR **#130234**, merged and **reverted the same day**, covering the **dot-product** family rather than `v_pk_add_u16`/`v_pk_max_i16`. Under this charter's evidence hierarchy that is a single secondary implementation — tier 4, hypothesis only. | #2013, PR #2115, PR #2123 |
| An inline **FLOAT** constant's contribution to the HIGH half of a packed 16-bit operand is **generation-dependent** — gfx11 duplicates it into `[31:16]`, gfx10.3 and older read zero | **The value is right; the generational premise is false, and that is the part worth recording.** Zero is correct on gfx10.3, so the packed-*integer* path (`half ? 0 : bits`) was right and the three f16 paths that replicated the constant and called the half select "a no-op for it" were wrong — fixed for `v_pk_*_f16`, `v_fma_mix_*` and the scalar f16 VOP3 `min3/max3/med3/fma` family. But **there is no gfx10-vs-gfx11 split to model**: `llvm-mc` folds `0x00003c00` to inline `1.0` and leaves `0x3c003c00` a 32-bit literal **identically on gfx1010, gfx1030, gfx1100 and gfx1200**. The fold is value-preserving, so inline `1.0` *is* `0x00003c00` in a packed f16 operand on every one of them. The integer side rules out "the assembler just never folds packed literals": `0x00000001` folds to `1` and `0xffffffff` to `-1`, while `0x00010001` does not. The real rule is simply that the packed operand **is the inline constant's whole 32-bit value** — which is why `-1` legitimately reads `0xffff` in *both* halves while `1.0` reads `0.0h` in the high one, and why the int and float cases never needed different code. The ISA states no replication rule and is consistent with this (VOP3P pseudocode reads `S0[31:16]`; the inline table gives a half constant as a 16-bit pattern). **Do not add a gfx-version predicate here.** Kernels `32r14a`-`32r14e` in `test_rdna2_to_spirv` pin the high-half value of each fixed path and fail under the select-ignoring reading. **Scope: this settles the rule for VOP3/VOP3P `OPSEL` only, and every OPSEL site now shares the helper. Six SDWA sites still make the select a no-op for an inline constant — SDWA's sub-dword select is a different encoding this evidence does not reach, tracked on #2191.** `CONFIDENCE: HIGH`. | #2119, PR #2177 |
| The RDNA 2 ISA guide's **summary tables 58 and 60** give the VOPC opcode layout — so `v_cmpx_*_f32` is at `0x50`-`0x5f` and `v_cmpx_*_i16` at `0xb0`-`0xb7` | **Falsified — those two tables are stale GCN-era boilerplate and contradict the same document.** They also list `V_CMPS`/`V_CMPSX` opcodes, which do not exist on RDNA2 at all, and they place compares in `0x40`-`0x7f`, where llvm-mc reports **64 invalid encodings**. The authority is the per-opcode **table 61**, which agrees with hardware: disassembling all 256 VOPC e32 encodings with `llvm-mc -mcpu=gfx1030` gives `v_cmpx` at `0x10`-`0x1f`, `0x30`-`0x3f`, `0x90`-`0x9f`, `0xb0`-`0xbe`, `0xd0`-`0xdf`, `0xf0`-`0xff`, each exactly its `v_cmp` counterpart **+ 0x10**. The `0xaf`/`0xbf` holes are confirmed twice: llvm-mc rejects both, and table 61 **skips opcodes 175 and 191** (running 174 -> 176 and 190 -> 192). The map now lives once, at `vopc_is_cmpx` in `rdna2_decode.cpp`, with boundary assertions on both sides of every window in `test_rdna2_decode`. **Do not "correct" it back to tables 58/60.** **The windows were duplicated in FOUR places, and the copies were not merely untidy — one was a silent miscompile.** `dead_wave_mask_writes` excludes cmpx from VCC definitions (a cmpx writes EXEC and has no VCC destination), but its private copy listed three of six windows, and the decoder gives every VOPC e32 `dst = 106` (VCC_LO) — so an unrecognised cmpx recorded a **phantom VCC definition**, a preceding live `s_and_b64`/`s_andn2_b64 vcc` looked overwritten before use, was classified dead and **elided**, and the real consumer read stale VCC with no diagnostic. Kernel `32r13v` reproduces it (`0x22220000` before, `0x11110000` after) with `32r13w` as a control on an already-covered window. A fourth copy in `gpu_executor.cpp`'s `changes_exec` under-reported EXEC writers to a dominance proof. All four now call the one predicate. | #2120, PR #2181 |
| Sonic Racing: CrossWorlds' failing vertex fetch is a **runtime-SELECTED descriptor the fold cannot model** — `s_load_dwordx4` a V#, then `s_cselect_b32` word 3 on a null-pointer test immediately before the fetch, so on one arm the descriptor "is in no table" | **Falsified by measuring values instead of reading opcodes.** The fold already models that entire idiom — `s_bfe_u64`, `s_or_b32`, `s_cmp_eq_u32` and `s_cselect_b32` are all implemented, the last explicitly as *"the vertex-fetch format patch's tail"*. It never reaches them: `PROSPER_DYNTRACE_FAIL=1` shows `s[18:19]` reading back **null** out of a constant buffer the fold reads correctly and in range, so the dereference fails and both the V#'s SOFFSET and its word-3 select are unknown from there down. Real guest data, correctly read, containing a null bindless-table pointer — not a modelling gap. **The generalisable trap: a fold that stops early looks IDENTICAL whether it ran out of modelled opcodes or out of readable data.** The `[recompile-reject]` / `[*-unresolved]` line is the same either way, and a disassembler can only answer the first question. When the chain involves memory, reach for the instrument that reports the chain — `PROSPER_DYNTRACE_FAIL=1` prints `base_ok`/`soff_ok`/`unreadable` per step — before hypothesising about opcodes. The static recon that produced this hypothesis marked its mechanics `CONFIDENCE: HIGH` (they held) and its inference `MED` (it did not), and named "which arm is taken" as the first thing to measure; measuring it is what killed the hypothesis. | #2013, #2131, #2132 |
| The stages failing #305's user-data condition indicate a recompiler defect | **No — the recompiler is behaving correctly and fail-visibly.** When the seeded user-data block is wrong, the const-fold refuses to invent a descriptor (`SOFFSET untracked -> fetch left unresolved (not folded to 0)`) and the draw is skipped rather than drawing garbage. The defect is upstream, in graphics register state — see [`RESOURCE_BINDING.md`](RESOURCE_BINDING.md) § `Ruled out`. | #305, #1607 |

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
