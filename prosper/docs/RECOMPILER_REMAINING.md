# RDNA2→SPIR-V recompiler — remaining work

**Date:** 2026-07-06. **Status: ~93.0% instruction coverage in-context; 34 of 41 shaders fully recompile —
at the practical ceiling for THIS title.** Every bring-up-critical class is covered (position/blit/clear,
textured incl. 3D sampling + NSA, interpolated, image-copy 1D/2D/3D + arrayed + NSA + MSAA + MSAA_ARRAY,
integer divide/modulo), plus the big structural features: counted-loop reconstruction (`OpLoopMerge`/
`OpPhi`), divergent-if handling (EXEC-predicated linearization incl. provably-dead scalar writes),
64-bit scalar (`s_bfe_u64` via Int64), and NGG vertex shaders. All paths are `spirv-val`-gated
(`tools/spv_validate`, permanent ctest) and, where a Vulkan harness exists, execution-differential-tested.

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

**The recompiler is DONE for this title.** Reaching the game's actual pixels is now gated on the GPU-
execution / render-loop frontier (see `RENDER_LOOP.md`), not the recompiler.

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
