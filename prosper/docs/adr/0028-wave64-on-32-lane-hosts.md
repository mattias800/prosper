---
kind: adr
status: proposed
date: 2026-10-07
---

# ADR 0028: Run guest Wave64 programs on 32-lane hosts by proof, exact emulation or visible refusal

## Context

PS5 shaders are compiled for 64-lane waves. A host whose subgroups are narrower than 64 lanes
(every NVIDIA GPU at 32; AMD under drivers that do not expose a 64-lane fragment subgroup; Intel at
8/16/32; lavapipe at 8) cannot run a program whose result depends on that width. What prosper does
today, read from the code on `origin/main`:

- **Fragment.** `lower_fragment_votes` (`src/gpu/recompiler/spirv_fragment_vote_lowering.hpp`) runs
  under `FragmentWavePolicy::ProvenVotes` and admits a program only when every vote carries a
  certificate: a predicate uniform across the logical guest wave, a pure dead user graph, or a
  neutral one-arm selection. Anything else is refused, the draw is dropped, and
  `src/diagnostics/perf/wave64_refusal.hpp` prints one `[wave64-unsupported]
  refusal=fragment/subgroup-contract` line per program (with `lowering=unproved-vote` and the
  failing vote's source word), plus a start-up `[render] WARNING` stating that draws are dropped.
- **Compute.** `frontends/shared/live/live_compute.cpp` (the `subgroup-contract-absent` and
  `subgroup-too-narrow` declines) skips any dispatch whose required subgroup size, or whose lane
  operations, the host cannot satisfy, counted as `compute/subgroup-contract`. There is no
  emulation of a 64-lane wave on a narrower compute subgroup.
- **Owned logical-Wave64 graphics draws** already exist: `src/gpu/execute/owned_graphics_wave_draw.cpp`
  (#4270, #4421, #4555) runs owned per-wave graphics draws. The fragment route below (#4384) builds
  on that path rather than beside it.
- **Recompile failures** for either stage are reported separately as `fragment/recompile` and
  `compute/recompile`; those are prosper defects, not host limits, and are out of scope here.
- **One approximation on the default path, narrowed by #4714.** A scalar DATA word that may be the
  structured emitter's fabricated zero must not become a lane-mask bit. #4711 guarded the fragment
  pair projection; #4714 puts one predicate (`scalar_words_projectable`) in front of the pair
  projection and of every Wave64 compute/fragment op that writes EXEC/VCC from scalar data
  (`s_cselect_b64/b32`, `s_pack_*`, `s_lshl/lshr_b64`, `s_bfe_u64`, `s_bitreplicate`, `s_mov`,
  `s_and_saveexec`) on both stages. **Not covered:** the Wave32 mask forms (`allow_b32_masks`) and the
  merged-NGG shell's VCC reconstruction, which rely on a CFG definiteness proof rather than the
  mark. That residue is still an exception to "the default never approximates".

This already satisfies `FAIL-1` for refusals: nothing refused is skipped silently. What it does not
do is run the programs. Evidence from *Assassin's Creed Black Flag Resynced* `PPSA28183` on
Windows with an RTX 4070 SUPER, as recorded: `docs/games/AC_BLACK_FLAG_STATUS.md` records thirteen
fragment programs refused by the recompiler and two by the backend (`subgroup-contract`,
`unproved-vote`), eighteen refused compute programs, and one start-up fragment draw (`0x407edfaf00`)
refused by the `unproved-vote` contract; the 2026-10-07 comment on #4131 (`main` `f2e5124a6`,
Windows) records three fragment `subgroup-contract` refusals. The shape of the `unproved-vote` vote
-- a `v_cmp` whose only consumer is EXEC feeding an `s_cbranch_execz` skip -- was read while
drafting this ADR and is not yet in the status doc; it is recorded there with migration step 1.
**Expectation, not measurement:** on RADV with 64-lane subgroups the `subgroup-contract` programs
should run natively. No Black Flag RADV run is recorded, and the `fragment/recompile` refusals would
fail there too.

The owner's draft PR #4384 (currently titled "gpu: checkpoint general logical64 fragment helper
launch") opens the exact fragment route on top of the owned-wave path: collect the fragment quads,
run the original pixel shader as an owned logical-64 wave in a compute-style pass, and replay its
outputs. It currently admits only straight-line, quad-local programs, and its review is open. This
ADR does not compete with it; it names it as the fragment route and sets the policy that ties all
routes together.

Reference designs, read for verification only (no code taken; paths relative to each project).
Citations are pinned to the commits read (all 2026-10-07): AnyPS5 `400ba7f5ec01`, shadPS4
`afde63182a38`, sharpemu `b02c06181a29`. Older checkouts differ; shadPS4 before that range may lack
`lower_wave64_pass.cpp`.

- **AnyPS5** emits each instruction once per half-lane context when `state.laneCount == 2`, sharing
  values proven uniform (`core/shader/recompiler/SpirvBackend/src/SpirvFlowEmitter.cpp:734-735`,
  `SpirvModuleEmitter.cpp:46`). That makes one host invocation carry two guest lanes, exact in
  divergent control flow.
- **shadPS4** lowers Wave64 ballot-style operations through workgroup shared memory, for compute
  only and only in blocks proven uniform; it warns otherwise
  (`src/shader_recompiler/ir/passes/lower_wave64_pass.cpp:106`, `:117-124`, `:151-169`). Fragment
  receives no such lowering.
- **sharpemu** plans a half-mask exchange with shared-memory barriers
  (`src/SharpEmu.ShaderCompiler/Ir/Gen5Wave64HalfMaskAnalysis.cs`, `Gen5Wave64HalfMaskPlan`).
- **vkd3d-proton** (from its documented behaviour, not re-read here) fails pipeline creation when a
  required wave size of 64 is unsupported, with no emulation.

No single implementation covers all the cases below: AnyPS5 does two-lanes-per-invocation,
shadPS4 and sharpemu do workgroup exchange for compute only, and prosper does proof plus refusal.

## Decision

One policy for every guest Wave64 program on a host that cannot offer a native 64-lane subgroup
for that stage. Let W be the host subgroup width for the stage (W < 64; 32 on NVIDIA, as low as 8
on Intel and lavapipe). Each program takes the first route that applies, chosen per shader by
analysis, and every route is logged by name. The default never approximates (the Wave32 residue
named under #4714 above is a defect to close, not a precedent).

1. **Native.** A host that offers a required 64-lane subgroup for the stage runs the program
   natively, always. Nothing below applies to it; AMD/RADV hosts see no change.
2. **Proven width-independent: one lane per invocation, zero cost.** Programs with no cross-lane
   operation, or whose cross-lane operations are proven independent of wave width, run as plain
   one-lane-per-invocation code. The existing `ProvenVotes` certificates are this route; they are
   extended with cheap exact cases. The first candidate is *an any-vote whose only consumer is EXEC
   guarding an `s_cbranch_execz` skip* (the Black Flag case). **Masking alone does not make that
   vote neutral.** VALU work in the skipped region is EXEC-masked, but SALU and SMEM instructions
   in it execute whenever any lane of the 64-lane guest wave is active and do not execute when the
   whole wave skips. Forcing the vote TRUE runs that scalar work where hardware skipped it;
   evaluating it per W-lane host subgroup lets one subgroup skip scalar writes that another, and the
   hardware, performed. So the certificate is the existing one-arm condition
   (`spirv_fragment_vote_lowering.hpp`: TRUE control only "after proving every extra operation
   UB-free and every live merge export stable and unchanged under P=false"), made concrete for
   this shape. The region, entered with an empty EXEC, must have:
   - **no scalar live-out**: no SGPR, SCC, VCC, M0, or EXEC-derived scalar value written in the
     region and read after the merge;
   - **no scalar memory effect**: no `s_store`/`s_buffer_store`/`s_atomic`, and no scalar load whose
     address or result is meaningful only when lanes are active and is read after the merge;
   - **no wave-level side effect**: no `s_sendmsg`, `s_barrier`, `s_setreg`, `s_sleep`/trap or other
     operation that is observable or UB under the forced arm;
   - **no scalar-dependent exit**: no branch out of the region other than to the merge.
   The conditions are checked on the guest program (the 64-lane wave is the semantic target), not
   on the host lowering.

   **This route is where most of the performance is, so it is grown before any emulation.**
   Compiled code uses cross-lane operations mostly as optimisations, not as 64-lane semantics.
   With uniformity analysis (natural on `GPU-3`'s SSA IR), these become exact one-lane code:
   - a `readfirstlane`/`readlane` of a value proven wave-uniform is a move;
   - a ballot whose result only feeds a popcount for a count, or `mbcnt` used only to hand out
     unique slots whose consumer is proven order-insensitive (an append buffer, a counter
     atomic), is a workgroup-scope prefix or reduction;
   - a vote over a wave-uniform predicate is that predicate.

   Each rewrite needs the same proof standard as a certificate: a positive arm and a mutation arm.
   A rewrite that projects a 64-bit scalar mask pair onto the invocation's own bit must also prove
   neither half is a fabricated zero (the #4711 mark, consulted for Wave64 in both stages by `scalar_words_projectable` since #4714).
   A program it fully rewrites never reaches routes 3-5.
3. **Compute, cross-lane operations only in uniform control flow: workgroup exchange.** One lane
   per invocation; ballot, vote, readlane, `mbcnt` and DPP/permutes that cross a W-lane boundary
   are lowered to an exchange across the 64 / W host subgroups of one workgroup, through workgroup
   memory with barriers (the shadPS4/sharpemu shape). Near-native parallelism; cost is shared
   memory and barriers at each exchange.
4. **Compute, cross-lane operations inside divergent control flow: N lanes per invocation.** Each
   host invocation executes N = 64 / W guest lanes (two on a 32-lane host, four on 16, eight on 8;
   the AnyPS5 shape generalised), exact under divergence. It divides the thread count by N and
   multiplies per-thread registers by about N, so occupancy and latency hiding drop; it is used
   only for programs that need it. A configuration whose register cost cannot be met (likely at
   N = 8) falls to route 6.
5. **Fragment, not proven.** Fragment invocations cannot be re-grouped by prosper and have no
   workgroup memory, so routes 3 and 4 do not apply to the fragment stage directly. Two ways in:
   - **Full-screen passes promoted to compute.** A draw proven to cover its target exactly once
     per pixel can run as a compute dispatch, which then takes route 3 or 4. That needs a single
     full-target triangle or rect with no blending, no depth/stencil test or write, no discard
     feeding a later pass and no MSAA. It writes the same texels, and post-processing chains,
     common in UE4 and Anvil titles, are mostly this shape.
   - **Everything else: the owned-wave route of #4384**, as it grows to admit more program shapes.
6. **Everything else is refused visibly**, exactly as today: draw dropped or dispatch skipped, one
   `[wave64-unsupported]` line naming the stage, program and refusal class, and the counters in
   `perf_alarms`.

An **approximate narrow-subgroup vote** (evaluating a guest vote over the host subgroup) may exist
only as an explicit, logged, default-OFF *guest-behaviour selector* under `CFG-1`, with an issue
whose resolution settles its default and deletes the switch. A run with it set is not evidence
(`VER-4`).

Performance claims are qualitative until measured. **No route becomes the default for a class of
shaders until a same-binary A/B on real titles** (`.claude/skills/perf-change/`) shows its cost,
the control arm being the route switched off (i.e. today's refusal or the cheaper route).

Adds spec rule `GPU-5` (`GPU-4` is proposed by ADR 0027, #4716).

## Consequences

- **Cost by route.** Native and proven: none. Workgroup exchange: shared memory plus one barrier
  per exchange, and a workgroup size constraint (the 64 guest lanes must sit in one workgroup).
  N lanes per invocation: 1/N of the host threads, about N times the registers per thread, possibly
  spills; correct but slowest. #4384's owned wave: an extra quad collection and replay pass per
  admitted draw.
- **Tests.** Every proof extension gets a positive and a mutation arm (a vote that does reach a
  live output must still be refused). The `s_cbranch_execz` certificate's mutation arms include,
  by name, a skipped region with an **SGPR live-out read after the merge** and one with a **scalar
  memory store**; both must be refused, and the positive arm is a hand-built region with VALU work
  only. Each compute route gets execution tests whose expected values are produced by the same
  program run at native 64-lane width on RADV, compared bit for bit. Lavapipe in CI (W = 8) and a
  32-lane host (W = 32) are different configurations of the emulated route (N = 8 and N = 2), so a
  green CI run does not cover the 32-lane case. A hand-built divergent ballot is the positive
  control for route 4, built outside the analysis that selects it.
- **Diagnostics.** The `[wave64-unsupported]` line gains a `route=` field for admitted programs, so
  a run shows which route each Wave64 program took and what is still refused.
- **AMD hosts:** nothing changes; route 1 short-circuits before any analysis.
- **Enforcement:** the refusal counters (`runtime:dropped-draws`, `runtime:skipped-dispatches`)
  and review; no tool can check that an approximation is not on by default except review against
  the switch registry.

## Alternatives considered

- **Approximate narrow-subgroup votes by default.** Makes more frames draw, but silently wrong
  whenever the subgroups disagree, which violates `FAIL-1` and turns a visible drop into an
  invisible defect.
- **Refuse forever.** Today's state; correct but leaves every NVIDIA user without the refused draws
  and dispatches, and the refusals are in shipped titles' main passes.
- **Require AMD hardware.** Abandons every user on a narrower host.
- **N lanes per invocation for every compute program.** Exact and simple, but pays the occupancy
  cost on programs that need nothing; the per-shader choice confines it.
- **Workgroup exchange for every compute program.** Cannot express divergent cross-lane operations
  exactly, which is why shadPS4 restricts it to uniform blocks.

## Migration order

1. The `s_cbranch_execz` certificate under the scalar-effect conditions in route 2, with the
   SGPR-live-out and scalar-store mutation arms (Black Flag's `unproved-vote` draw), and the vote's
   shape recorded in `AC_BLACK_FLAG_STATUS.md`. See *Approval* for when it may land.
2. Add the `route=` field and per-route counters, then census which refused programs the route-2
   rewrites (uniform `readlane`, compaction ballots, uniform votes) would admit.
3. Compute workgroup exchange for uniform control flow, with RADV-native comparison tests.
4. N-lanes-per-invocation for the divergent remainder.
5. Full-screen fragment-to-compute promotion, gated on its coverage proof and compared texel for
   texel against the fragment path where the latter runs (AMD).
6. Grow #4384's admitted shapes, on the owner's schedule.
7. Only after A/B data: decide per-class defaults; settle or delete any approximation selector.

## Open questions

- Whether routes 3 and 4 share one IR lowering with a per-program parameter, or are two passes;
  `GPU-3` (SSA IR) may make the former natural.
- How #4384's owned wave and `owned_graphics_wave_draw.cpp` relate to route 4: the logical-64 wave
  in a compute-style pass may be the same N-lane machinery, in which case fragment and compute
  converge on one implementation.
- Whether workgroup exchange is admissible when the guest's own LDS use leaves too little shared
  memory, or must fall to route 4.

## Approval

Requires the project owner's acceptance, and the owner's agreement that #4384 is the fragment
route this policy names. Acceptance unblocks migration steps 1-5. Step 1 may land before
acceptance only as its own PR that the owner reviews on its own terms, as a proof extension under
today's `ProvenVotes` policy; it does not land on the strength of this proposal.
