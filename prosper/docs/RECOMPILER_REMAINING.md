# RDNA2→SPIR-V recompiler — remaining work (the last 11 of 41 shaders)

**Date:** 2026-07-05. **Status:** the recompiler is at a well-tested plateau — **30 of 41** real game
shaders fully recompile in-context (~86% instruction coverage), every **bring-up-critical class** covered
(position/blit/clear, textured, interpolated, image-copy incl. 1D/2D/3D + arrayed + NSA, integer
divide/modulo). All paths are `spirv-val`-gated (`tools/spv_validate`, permanent ctest) and, where a Vulkan
harness exists, execution-differential-tested.

The remaining 11 shaders each need **one or more genuinely deep features** — verified below by
disassembly. **None is completable by a single bounded opcode/feature add**, which is why coverage has
plateaued: the cheap, safe, data-driven wins are exhausted. These are also all **advanced compute/geometry
effects — none is needed for the first frame** (the critical path to on-screen graphics is the GfxDevice
boot wall; see `GFXDEVICE_BRINGUP_PROBLEM.md`).

## What each remaining shader needs

| Shaders | Class | Blocking features (all required together) |
|---|---|---|
| **031, 033** | counted-loop MSAA resolve | **loop reconstruction** (`s_cmp`→`s_cbranch_scc0` exit + backward `s_branch`; SGPR loop counter `s10`; loop-carried FP accumulators `v6–v10`) **+ 2D_MSAA** image dim |
| **032, 034** | loop + arrayed/MSAA sampling | loop reconstruction + **2D_ARRAY / 2D_MSAA_ARRAY** image dims (arrayed non-MSAA storage is done; MSAA + the loop are not) |
| **004, 025, 040** | NGG vertex/primitive | **NGG preamble**: `s_sendmsg(GS_ALLOC_REQ)`, `exp prim`, and wave-packing EXEC setup (`s_lshr_b64 exec,-1,vcc`) — modellable as per-invocation no-ops, but needs care to prove correctness |
| **006, 030, 037, 038** | large / wave-level | loop reconstruction **+ wave-level ops**: `s_bfe_u64`/`s_lshr_b64` writing **EXEC** (wave-lane setup), **cross-lane** `v_mbcnt_lo/hi` (active-lane count), `v_readlane`; plus a long ALU tail. Hardest; multiple features each. |

## The big lever: loop reconstruction (unblocks the most, but is architectural)

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
2. **Loop reconstruction** — as a deliberate generality investment (also the gate for many UE4/other-title
   shaders), when prioritized.
3. NGG preamble; MSAA/MSAA-array image dims — smaller, each unblocking 2–3 of the remaining shaders once
   loops land.

Marginal instruction-coverage ops (e.g. `v_cndmask_b32_e64`, `s_bfe_u64`) can still be added safely but
**complete no additional shader** on their own, so they are deferred in favour of the above.
