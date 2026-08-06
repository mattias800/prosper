# Messenger black-render investigation

This is the canonical completed record for the Messenger gameplay-render failure (#300 / #522).
The formerly invisible save-game list (#299), missing foreground (#530), and boot-to-intro regression (#540)
are fixed and closed. GitHub issues own live findings; this document preserves the evidence boundary that
connected each visible failure to a specific replay, renderer contract, and regression.

## Current conclusion (2026-07-11)

Immutable capture/replay (#514), the color-disabled depth/stencil-pass fix (#520, merged as `376a801`),
and resource-producer provenance (#524) established this causal chain:

1. The retained current-master capsule contains 46 draws. Replaying draws 0:30 produces the real first-level
   landscape with 56,036 RGB-nonblack pixels (BMP SHA-256 prefix `f821d2c28aac4cb8`). Geometry, vertex fetch,
   asset textures, depth/stencil masking, and scene render-target population are working for this frame.
2. Full-screen post draw 31 consumes that healthy 1920x1080 scene RT at PS binding 34, a white 1x1 RGBA8
   texture at binding 35, and a 1024x32 RGBA16F texture at binding 36. The last resource's captured backing
   has only 6 nonzero bytes out of 524,288 (hash `4384046b4840fbaa`). The draw produces zero RGB-nonblack
   pixels, and downstream composition preserves the black result.
3. Binding 32 contains `(1/1024, 1/32, 31, 1)`. The fragment shader (SPIR-V hash `1ad0babe1990d5c0`)
   samples binding 36 at adjacent blue slices and interpolates. Together with the texture dimensions, this
   identifies binding 36 as a flattened 32x32x32 color-grading LUT.
4. `PROSPER_TESTLUT32=1` replaces only that decoded resource with an identity 32-cubed LUT. Replaying through
   draw 31 then restores the complete landscape through the original post shader: all 57,600 output pixels
   have nonzero RGB, output hash `f962051005b617ef`, and BMP SHA-256
   `fdfc5d0f2fc208403be1be86c483d94622ec1d19cbb2594b65711b4f8a71fb75`. This proves the source RT,
   fullscreen geometry, descriptor sampling, 1x1 input, and grading shader are operational.
5. Retained color-target history finds the direct writer of that 1024x32 resource in the same submit: raw
   draw 40, immediately before the grading consumer. Re-realizing the producer reports one unsupported pixel
   instruction, VOP1 opcode `0x0e`, `V_CVT_OFF_F32_I4`. The compute-producer hypothesis is falsified for this
   resource: retained per-dispatch state contains no matching direct 1024x32 image.
6. The producer also exposes a second renderer contract error. Its `CB_COLOR0_ATTRIB2` declares 1024x32, but
   the live renderer previously gave every target the scaled VideoOut extent. Decoding and carrying the native
   target dimensions lets small lookup surfaces render at their actual resolution while large scene/scanout
   surfaces retain the configured render scale.
7. Implementing `V_CVT_OFF_F32_I4` (signed low nibble to f32, scaled by 1/16) and the target-extent fix makes
   the unmodified producer write 32,734/32,768 RGB-nonblack LUT texels. The original grading consumer hits that
   cached target and produces 56,008/57,600 RGB-nonblack scene pixels. The selected VideoOut front buffer then
   contains roughly 37,100-37,500 nonblack pixels and visibly shows the first-level landscape and dialogue UI.
   No `PROSPER_TESTLUT32` or other synthetic resource override is active.
8. The earlier 256x16 pixel-art palette is a transitional pass, not the persistent failing shader. A repaired
   hardware-breakpoint trace (#525) observes `PaletteSwapImageEffect.FadeBlackToGameCoroutine.MoveNext`
   progress through all eight entries and complete. Managed `Material.SetTexture`, its native `SetTextureImpl`
   calls, eight distinct native texture pointers, and the corresponding live descriptor addresses all change.
   That transition eventually produces the expected colored scene before the grading pass erases it.

The original black first-level root cause was therefore a missing shader instruction plus loss of per-target dimensions,
not depth, a stuck palette fade, compute LUT baking, or detiling of the 256x16 palette. The fix merged as #528
(`e5fce22`) after a clean exact-route run without the identity-LUT substitution: the LUT, grading output, and
selected VideoOut front buffer remained nonblack across consecutive flips. #300 and #522 are closed.

The hardware oracle then exposed two independent follow-up defects hidden by the black frame. #534
(`3941533`) fixes reversed Vulkan front-face enum values; retaining culling with the corrected winding restores
the foreground canopy/tree, rock slopes, player platform, waterline structures, and right-side terrain that
were absent from the first visible frame. #541 (`ded4a60`) separates persistent D32S8 layout initialization
from logical depth and stencil validity; earlier stencil-only `ALWAYS`/read-only use no longer makes untouched
depth contents valid, so the later intro `GEQUAL` draw initializes reverse-Z depth correctly instead of loading
the logo fallback and rendering black.

Final validation used a fresh save, the recorded gamepad route, no rendering substitutions, and 180 normal
screenshots at one-second intervals at native 1920×1080. The sequence covers startup through the first-level
dialogue with the full hardware-reference composition, and the user confirmed the graphics look correct.
#530 and #540 are closed. Timing of the fade remains an observation, not a diagnosed emulator defect: the
screenshot tool samples the current buffer on wall time and can repeat frames under synchronous rendering.

The old bindless vertex-fetch frontier is solved, and `NEXT_STEP_VERTEX_FETCH.md` is historical. Earlier
#300 claims about texture decode, alpha, transform collapse, render-target propagation, and a missing
vertex-color binding were made on changing revisions and were later overturned. Do not combine them into a
new proof without reproducing them against the retained capsule or a fully recorded current-master run.

## Ruled out

One line per dead hypothesis, the evidence that killed it, and where that evidence lives. **Do not
restart any of these without contradictory new evidence** — this is the list `CLAUDE.md` points at.

| Hypothesis for the black first level | Verdict and evidence | Source |
|---|---|---|
| Depth / stencil state | **Not the cause.** Replaying draws 0:30 of the retained capsule already produces the real first-level landscape (56,036 RGB-nonblack px); the frame is lost later, at the grading pass. The separate D32S8 validity defect (#541) was a *boot-progression* fix, not the black-level cause. | #300 / #522, #520 |
| Vertex fetch / bindless-dynamic descriptor resolution | **Solved and closed.** Both stages recompile and dynamic V#/T#/S# resolve on master. `NEXT_STEP_VERTEX_FETCH.md` is a historical bring-up record and carries its own superseded banner — do not start from it. | #514, #515 |
| Geometry, transform collapse, or a missing vertex-colour binding | **Overturned.** These #300-era claims were made on changing revisions and were later contradicted; geometry, vertex fetch and asset textures are demonstrably working in the same capsule that renders black. Do not recombine them into a new proof without reproducing them against the retained capsule or a fully recorded current-master run. | #300 |
| The 256x16 pixel-art palette fade is stuck | **Falsified.** A repaired hardware-breakpoint trace observes `PaletteSwapImageEffect.FadeBlackToGameCoroutine.MoveNext` progress through all eight entries and complete, with eight distinct native texture pointers and matching live descriptor addresses. | #525 |
| Detiling of the 256x16 palette | **Not the cause.** That pass is transitional; it produces the expected coloured scene, which the grading pass then erases. | #525 |
| A compute dispatch bakes the 1024x32 grading LUT | **Falsified.** Retained per-dispatch state contains no matching direct 1024x32 image; the writer is raw *draw* 40 in the same submit, immediately before the grading consumer. | #524 |
| Texture decode / alpha / render-target propagation | **Overturned**, same provenance as the geometry claims above. | #300 |
| IL2CPP's incremental Boehm GC (bdwgc) collects live scene objects, so no scene geometry is ever submitted | **Falsified.** The premise was that the title runs a composite chain over an empty scene. It does not: the real cause was the LUT producer's missing `V_CVT_OFF_F32_I4` plus lost per-target dimensions (row below), and the title now completes its first level under the `messenger-scene` guard. The collector is not among tracker #1865's current blockers, and the collector is not among tracker #1865's current blockers. The same claim for `PPSA02664` is separately falsified in `PPSA02664_BLACK_WORLD.md` — the world renders and is then painted over by a full-frame mask fill (#1578). `CUTSCENE_GC_ABORT.md` records a *separate* blocker — the `ABORT("Unexpected state")` on the level1 intro load — which this row does **not** retire: it presents as open, with open leads, and nothing here establishes that it no longer reproduces. | #229 |

**The actual root cause**, for contrast: a missing recompiler instruction (`V_CVT_OFF_F32_I4`, VOP1
`0x0e`, #526) in the LUT producer plus loss of per-target dimensions (`CB_COLOR0_ATTRIB2`, #527),
fixed together by #528 (`e5fce22`); then two follow-ups the black frame had been hiding — reversed
`VkFrontFace` translation (#534) and depth/stencil validity tracked independently in persistent D32S8
surfaces (#541). The formerly invisible save-game list was #299.

## Evidence policy

Every new finding posted to #300 or #522 must include:

- exact git commit and whether local changes were present;
- title dump ID, savedata directory/seed, exact input route, and relevant environment flags;
- pad-read, flip, submit, pass, and draw identifiers as applicable;
- hashes of shader code, resource table, referenced bytes, and output image/pass;
- the observation, the narrower conclusion it supports, and alternatives it does not distinguish.

A diagnostic override proves only the boundary it changes. Call something a root cause only when a real fix
changes the same failing replay/live state and a regression fails before and passes after it.

## Completed merge validation

The #528 merge gate completed all of the following:

1. Ran the focused recompiler, command-processor, render-state, and capture tests plus the full CTest suite.
2. Repeated the exact saved gameplay route from a clean build with no `PROSPER_TESTLUT32` override. The targeted
   `PROSPER_RENDER_RESOURCE_DIM=1024x32` run is valid producer/consumer proof, but normal gameplay must not depend
   on that diagnostic selection flag.
3. Confirmed the front buffer, not only an intermediate target, contains the landscape for multiple consecutive
   flips; the revision, route, environment, and representative RGB-nonblack counts are recorded on #522.
4. Kept the identity-LUT override and retained provenance tools diagnostic-only. They are useful for future
   first-bad-contract investigations but are not part of normal title behavior.

## Tooling

### Immutable GPU capture/replay (#514, complete)

`.prgcap` v3 records ordered draws, target transitions, native target dimensions, shaders, resource tables,
and owned backing bytes,
then reproduces the live hash without Unity. Use `gpu_replay --inspect-only`, draw-range isolation, resource
dumping, and shader dumping for offline work. Captures contain game data and remain gitignored.

### Strict shader/resource contract (#515)

`PROSPER_DESCRIPTOR_VALIDATE=warn` reflects the statically used SPIR-V descriptor interface and reports
set/binding, descriptor class, stage visibility, runtime provenance, and provable buffer ranges.
`PROSPER_DESCRIPTOR_VALIDATE=strict` rejects malformed, missing, ambiguous, wrong-type, or undersized bindings
before Vulkan submission. `poison` continues with NaN-like buffers or magenta/cyan textures at invalid bindings.
Use `gpu_replay --validate CAPTURE.prgcap` for the same deterministic check without creating Vulkan objects.
An all-zero address/size buffer descriptor is an intentional hardware null/zero-read binding, distinct from an
absent table entry. Graphics storage images remain a separate backend limitation tracked by #374.

### Resource producer provenance (#524)

`PROSPER_PROVENANCE_DIM=WxH` retains matching color-target writers across submits and reports the last writer
when a draw samples a matching image. `PROSPER_COMPUTELOG[_DIM]` records dispatch program/resource state. These
tools found Messenger's direct raw-draw producer and falsified the compute hypothesis; extend the same structured
history to DMA/COPY/WRITE and CPU-visible operations as future titles require.

### Structured per-draw probes

On a replayed draw, expose vertex position/color/UV, raw texture samples, constant-buffer factors,
pre-blend fragment output, and final attachment output in a standard report. Existing switches such as
`PROSPER_RENDER_TESTPS`, `PROSPER_TESTTEX`, `PROSPER_CBUFLOG`, and draw isolation are useful primitives.

**First triage: `PROSPER_DRAW_STATS=1` (the per-draw "fragment funnel").** Before even the visual filmstrip,
run `PROSPER_DRAW_STATS=1 gpu_replay <capsule> out.bmp`: it wraps every draw in pipeline-statistics + occlusion
queries and prints one line per draw classifying **where its pixels vanished** — `GEOMETRY-VANISH` (all
primitives clipped/degenerate/off-screen → a vertex/fetch/transform bug), `NO-RASTER` (cull/scissor/zero-area),
`TEST-KILLED` (depth/stencil rejected every sample), or `passed-samples` (colour/stencil written). This is
objective, needs no oracle, and turns "why did this draw render nothing" into a glance instead of a manual
bisection (it localised GTA V's menu black-wedge defect to a single `GEOMETRY-VANISH` mask draw in one run).
See `tools/gpu_replay/README.md`. Use it to pick the suspect draw, *then* the filmstrip below to see it.

**When the funnel says `GEOMETRY-VANISH`, follow up with `PROSPER_GEOM_PROBE=N`** (per-draw geometry probe):
it captures draw N's post-transform clip-space vertices via transform feedback and reports where they landed
(clip-space bbox, on-screen/clipped counts, `w<=0`, NaN, all-collapsed). That distinguishes *why* the geometry
vanished — degenerate (fetch returned zero/constant), behind camera (`w<=0`), or shifted off-screen (bbox
outside `[-1,1]`). It localized GTA V #1231's vanished mask draw to an off-screen transform (bbox `x[-4.86,0]`,
all verts outside the clip cube) vs a working mask on-screen. Read it with the funnel — the funnel owns the
"rasterizes?" verdict, the probe owns *where the vertices are*.

**Then localize visually with `gpu_replay --draw-steps` (visual bisection).** Before probing individual draws,
run `gpu_replay --draw-steps PREFIX --draw-steps-target WxH <capsule>` on the scene: it dumps a numbered BMP
filmstrip of the composite building up per operation-step (plus a `[draw-step]` log with `visible-px`/`hash`),
so the exact step where the black/wrong composition first appears is found by scrubbing — no oracle needed.
Narrow with `--draw-steps-every 1` around the divergence, then use `--through-operation`/`--draw` for the
precise draw and `--graph` + `--dump-resource` for the surface it samples and its writer. See
`tools/gpu_replay/README.md` ("Isolation and extraction"). This turns "which of N draws collapsed the frame"
into a scrub-and-look instead of a guess, and is the recommended first step of this whole procedure.

## Decision boundary

Do not start another depth/stencil, vertex-fetch, geometry, palette-fade, compute, tiling, or whole-stack rewrite
from the historical #300 comments — see `## Ruled out` above for the evidence against each. The producer, missing
opcode, target extent, consumer, and front-buffer result are now connected by one live trace. Preserve that chain as the regression boundary and use the same provenance-first
method for the next title failure. Dead Cells startup is stable and its Evil Empire splash is captured
(#539/#545 closed); active work has moved to practical software-render progression capture (#549), reusable
checkpoints/snapshots (#302/#248), late offline-service contracts (#552), and existing 3D Unity/Unreal workloads.
