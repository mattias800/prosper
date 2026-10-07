---
kind: status
status: current
---

# Kena: Bridge of Spirits (`PPSA01802`) — status

Unreal Engine 4 (Ember Lab), one 28.5 GB `kena-ps5.pak` (no IoStore), Wwise, SDK `0x03000000`. Tracker
[#3787](https://github.com/mattias800/prosper/issues/3787). Brought up on Windows 11 / RTX 4090;
Linux/AMD title-menu investigations are recorded below.

## The sun adds nothing: the lighting-channel texture is all zero (2026-10-07, #4703)

**Read this first.** Measured on Linux/RADV with `prosper-app` in a visible window, `PROSPER_NULL_PAGE=1`,
`scripts/kena/linux-reach-level-load.pad`, at the title menu. The builds were `main` `156714914` and the
depth-bounds branch below. The instruments were RenderDoc captures of three guest frames
(`PROSPER_RENDERDOC_AT_PAD_FLIP=430`, `PROSPER_GPU_LABELS=1`) and two short register probes that were
never committed. The F9 bundle still aborts on this title (#3807: binding 32, address
`0xf00000000000`, declared `0xffffffff`), so there is no `.prgbundle` to replay.

- **The directional light contributes exactly 0.** The deferred sun pass (program `0x500aed0000`)
  adds 0.0000 luminance to the scene colour in the regions its own shadow mask calls lit, shadowed
  and partial (12,742 / 60,371 / 5,427 samples). RenderDoc pixel history at a sunlit path pixel
  shows the draw passing with `shaderOut [0,0,0,0]` in all three frames. The lit scene is ambient
  light only, which is why the world looks flat and grey next to the oracle (#3781).
- **Why: the lighting-channel test fails at every pixel.** At pc 43–47 the light ANDs its
  lighting-channel mask with an `image_load` of UE4's lighting-channel texture, then kills every lane
  whose result is 0. That texture is 0 at 100% of view pixels.
  - Its writer (program `0x5008fa0000`, the stencil-to-lighting-channels copy) renders an
    `R16_UINT` target (`CB_COLOR0_INFO=0x00050408`) through a UINT16_ABGR compressed export
    (`SPI_SHADER_COL_FORMAT=0x7`).
  - prosper unpacks every compressed export as FP16 and has no `R16_UINT` colour target, so the
    value 1 is stored as FP16 bits `0x0001` = 6e-8 in an `R8G8B8A8_UNORM` image, which rounds to 0.
  - This is a general gap, filed as **#4703** with the evidence and a suggested fix.
- **The shadow-cascade projections covered every pixel (fixed in this change).** UE4 restricts each
  CSM cascade's projection to its depth slice with the depth-bounds test
  (`DB_DEPTH_CONTROL=0x8`, bounds [0.0005, 0.0016], [0.0015, 0.0065] and [0.0059, 1.0]). The local
  light volumes use it too (`0x6a`), and so does a near/far pair split at depth 0.00125.
  - prosper never decoded `DEPTH_BOUNDS_ENABLE`, and bound no depth attachment for a draw that only
    bounds-tests. Each cascade therefore wrote the whole mask, and the last one drawn won.
  - With the test implemented, the three cascades partition the screen in the capture.
  - The title frame changes in 0.12% of pixels: the sun the mask feeds still adds 0 (above).
- **Not yet localised:** the light shafts, the foliage and the grade.
  - The shafts plausibly need the sun. Auto-exposure brightening a sunless scene would also wash it
    out. Neither link is measured.
  - The GBuffer has no grass or fern geometry, and the ground under it is the landscape's dark
    albedo. The grass pass has not been found.
- **Instrument note.** The RenderDoc run (and only it) refused three fragment programs
  (`0x505b7f0000`, `0x5040eb0000`, `0x5040890000`: `shader-recompile/fragment:12046` dropped draws).
  The default run on the same binary refused none in 46 windows. The RenderDoc frame's final image
  matches the default run's F9 screenshot, so the measurements above stand. The cause of the
  difference is unknown.

## No refused fragment programs remain (2026-10-07, #4680)

Measured on Linux/RADV with `prosper-app` in a visible window, `PROSPER_NULL_PAGE=1`,
`PROSPER_DBG=1`, `PROSPER_DROPPED_DRAW_CENSUS=1`, `scripts/kena/linux-reach-level-load.pad`, 260 s.
Before is `main` `a6453028c`; after is the #4680 branch.

| | before | after |
|---|---|---|
| `dropped-draws` reasons | `volume-multi-target:7943`, `ngg-subgroup:3666`, `shader-recompile/fragment:717` | `volume-multi-target:6045`, `ngg-subgroup:2790` |
| `unsupported-wave64-shaders` alarm | 47 of 48 windows, `fragment/recompile:717` | did not fire |
| refused programs (`refused_shaders_*/index.txt`) | 2 fragment | none |

The `shader-recompile/fragment` count was **two unrelated causes**, not the one program
(`0x505c3d0000`) the entry below names. Program addresses are run-local.

- **AGC helper rectangles with an inherited pixel shader.** These were most of the count. Every
  draw had the helper's vertex program (`es=0x5007020000`). The helper binds no pixel shader, so it
  runs with whatever the previous draw left bound, and the refused program changed from run to run
  (295 dwords in one run, 182 in the next).
  - The stale pixel user data held `00000092 00fff000 05000000` at `s[24:27]`, which the fold
    cannot publish as a V#.
  - Kena's helpers have no depth/stencil effect (`DB_DEPTH_CONTROL=0x70`). Their colour is
    suppressed (#4610), so they were "no effect" anyway; the decision just came after the pixel
    shader compile.
  - The executor now decides it before any shader work. The census reports these draws as
    `no-effect(agc-helper)`.
- **A counted loop counting in VCC_HI** (140 dwords). It appears after New Game. Its header is
  `s_cmp_lt_i32 vcc_hi, s34`, so VCC_HI is the induction variable. The counted-loop emitter now
  closes VCC's mask phi with a placeholder when the mask is provably unread from the header. The
  program compiles live (5,442 words, wave64).
- **The title-menu picture does not change.** An F9 frame at 222 s matches
  `assets/screenshots/kena-title-menu-world.webp`, apart from the falling leaf. That is expected:
  the helper draws had no effect to lose, and the loop program draws after the menu. The
  foliage, light shafts and grading are still missing. The other two drop classes,
  volume-multi-target and ngg-subgroup, went to zero with #4643 (next entry) without restoring them.

## The translucency-lighting volumes draw; the fog and the foliage are not theirs (2026-10-07, #4643)

**Read this first.** Measured on Linux/RADV with `prosper-app` in a visible window, default launch
(`PROSPER_NULL_PAGE=1`, `scripts/kena/linux-reach-level-load.pad`, 260 s), branch
`fix/issue-4643-multitarget-volume` against `main` at `a6453028`.

- **What #4643 was.** Kena's translucency-lighting injection writes two 64³ RGBA16F volumes from one
  layered pass: one in MRT0, one in MRT1. The backend refused every volume pass with more than one
  colour target, so all of its draws were dropped. The merged-NGG producer in the same passes was
  dropped too (`ngg-backend-readback-split`), because a split of an MRT pass carried its slots by
  readback. The backend now attaches every slot's own volume through a 2D-array view of its slice
  range (`tests/fixtures/render_volume_slots.h`). A split of such a pass carries every slot by LOAD,
  with no readback.
- **Result.** Dropped draws over 260 s, from the `[perf-alarm] summary` breakdown:

  | | `main` | #4643 |
  |---|---|---|
  | `backend/volume-multi-target` | 9,984 | **0** |
  | `backend/ngg-subgroup` | 4,608 | **0** |
  | `shader-recompile/fragment` | 915 | 713 |

  The `PROSPER_DUMP_PERSISTENT=ms:215000` census lists six populated 64³ volumes
  (`0x50132f0000` … `0x50146f0000`, every texel non-zero, no non-finite value). `main` lists one,
  `0x5013ef0000`, whose bytes differ in 1,641,440 of 2,097,152 bytes between the two runs.
- **The picture barely changes.** F9 frames at 225 s differ in 0.7% of pixels by more than 30/765,
  around candle flames, lanterns and particles. Part of that is animation. The fog, the washed-out
  grading, the missing foliage and the missing light shafts are unchanged. So neither drop class was
  the cause of those defects (see `## Ruled out`). What remained in the drop census was the refused
  Wave64 fragment program, since removed by #4680 (entry above).
- **Cost.** The four-volume clear (`0x5007bc0000`) now publishes four claimed volumes per frame
  instead of one. Surface readbacks rise from 2 to 5 per flip, each a fenced CPU wait. #4652 already
  tracks avoiding that wait; this makes it four times as valuable on Kena.
- **Rung.** Still 2.

## The title-menu world renders on `main` (2026-10-07, #3835)

**Read this first.** Measured on Linux/RADV with `prosper-app` in a visible window, default launch
(`PROSPER_NULL_PAGE=1`, no input), `main` at `97917c5f`.

- **The shrine and forest draw behind the title menu.** An F9 screenshot at 235 s shows the Kena
  wordmark, the `New Game / Load Game / Options` menu and `Version 1.16` over the 3D scene, 95% of
  the frame non-black. A RenderDoc capture of three guest frames at 215 s shows the same scene in
  all three. In those three frames, no 1600×900 RGBA16F target holds a NaN or Inf, and no draw
  reads a fragment input that its vertex program does not write (every draw from the scene pass to
  the scanout was checked). `tools/evidence/prerender_check.py` finds no dump asset that explains
  the frame.
- **The NaN half image came from AGC's helper rectangle.** #3835's 2026-10-06 chain was measured on
  `feat/ngg-live-p5` and `fix/issue-4625-compute-renderer-volume`. Neither branch contains #4610,
  which landed on `main` the same day.
  - The half target's writer was ES `0x3007060000`, a 112-byte vertex program that exports only
    the primitive and the position. That is Kena's compile of AGC's helper rectangle, whose words
    are `kKenaRect` in `tests/gpu/execute/test_efc_helper_program.cpp`.
  - The pixel shader on that draw is the one the previous draw left bound. It is
    `0x30096e0000`/`0x50096a0000`, a 132-dword program that reads five flat PARAM inputs and
    samples one 32×32 texture four times. Its own vertex program, `0x5009660000`, exports
    PARAM0–6.
  - That helper is a metadata operation (here AGC's "Decompress Htile") that writes no colour, so
    the inherited pixel shader's output must not reach the target. #4610 recognises the helper by
    its program family and draws it with no colour write. Live, it reports
    `helper rectangle recognised by family: vs=0x5007020000 … op=decompress-htile`.
- **The A/B.** Both arms used one scratch binary: `main` plus a local switch that skips only
  the helper's colour suppression. The switch was never committed. Each arm took one F9 frame at
  235 s:

  | arm | runs | F9 frame non-black |
  |---|---|---|
  | suppression on (`main`'s behaviour) | 2 (one on this binary, one on the `main` build) | 95.2%, 95.1%: world and menu |
  | suppression skipped | 2 | 0.0%, 0.0%: the whole frame is black, menu text included |

  With suppression skipped, a draw-program census shows the helper drawing with several inherited
  pixel shaders. Two of those draws target the scanout at `0xbfc0000000`.
- **What is still wrong.** Compared with the 1.04 oracle (#3781), the grass, ferns and flowers in
  the foreground are missing, so the ground is black. The light shafts are missing, and the scene
  is foggier and less saturated.
  - The run's `dropped-draws` alarm fires in 42 of 43 windows. Its reasons are
    `backend/volume-multi-target:7722` (#4643), `backend/ngg-subgroup:3564` and
    `shader-recompile/fragment:595`. The fragment count is one refused Wave64 program,
    `0x505c3d0000`. (Superseded by the entry above: it was two causes, and both are fixed.)
  - The volume passes are a likely cause of the fog and the missing light shafts, and the NGG
    subgroup drops of the missing foliage. Neither link has been checked per draw. **Superseded by
    the #4643 entry above: both drop classes are now zero and none of those defects moved.**
  - The refused fragment program was logged on a draw whose vertex program is the helper
    (`es=0x5007020000`), so some of those 595 uses may be helper draws that write nothing anyway.
- **The lit scene filling only the top-left two thirds of its target is not a defect.** The scene
  target is 3200×1800, and the complete view covers about 2133×1200 of it. The next pass writes a
  full-frame 3200×1800 image from it, and the final 3840×2160 frame is correct. This is UE4's
  screen-percentage rendering followed by an upscale.
- **Rung.** The title screen's 3D layer now renders, which is still rung 2. Gameplay was not
  reached on Linux.
  - `scripts/kena/reach-first-gameplay-prompt.pad` does not carry over. Linux polls the pad
    quickly during boot, so all seven of its pad-read anchors fire within the first 40 s, long
    before the menu appears at about 200 s.
  - The new `scripts/kena/linux-reach-level-load.pad` uses seconds anchors held for 1 s. It reaches
    "Choose Difficulty", seen on screen at pad flip 1000, and then the first level load.
  - **The level load then loses the Vulkan device, in 2 of 2 runs**, at about 500 s and 640 s.
    RADV reports `context is guilty of a hard recovery`. The first failing submit is compute
    `0x500a380000`, dispatch 22. That only names the submit that saw the loss, not the work that
    hung. In those runs, 13 vertex, 4 vertex-main, 9–14 fragment and 4 compute programs were
    refused. The next run should arm `PROSPER_GPU_BREADCRUMBS=1` or `RADV_DEBUG=hang`. **This is
    the gameplay blocker on Linux.**

## The volume clear runs; the next blocker is #3835's NaN half image (2026-10-06, #4625)

**The 2026-10-07 entry above supersedes this one's conclusion.** The NaN half image below was AGC's
helper rectangle drawing with a pixel shader the previous draw left bound. These runs were on
branches without #4610, and `main` no longer writes colour for that draw. Measured on Linux/RADV,
stacked on #3135 P5 (`feat/ngg-live-p5`). The runs were `prosper-app` default launches
(`PROSPER_NULL_PAGE=1`, no input, 260 s) and one `tools/screenshot` run (26 samples, 10 s apart).

- **What #4625 was.** Compute `0x5007bc0000` is a clear: four 64×64×64 RGBA16F storage volumes, 16³
  groups of 4×4×4 threads, all writes zero. Once the NGG producer drew into one of the volumes, the
  renderer claimed it and the clear was skipped on every frame. The fix publishes a claimed volume
  to guest memory before a compute binding reads guest bytes. It reads the retained image back,
  tiles it in the producer's native layout, and releases the claim
  (`src/gpu/execute/renderer_volume_publication`).
- **Result.** The `skipped-dispatches` alarm went from 41 of 45 windows (`backend-declined:336`) to
  1 of 33, 1 of 39 and 1 of 45 windows, with `backend-declined:1` in each run. That one skip is a
  different program: `0x5008be0000` binds a 512³ R32 volume, which exceeds the 512 MiB backend
  bound. **The menu world is still black.** The menu text appears in 23 of 26 samples, and every
  sample is black behind it (`tools/screenshot`).
- **Where the picture is lost.** These measurements come from a RenderDoc capture of one menu
  frame on this build, and they reproduce #3835's 2026-09-25 chain:
  - The lit scene target is present. It is 3200×1800 R11G11B10, 100% non-zero, mean 0.12, and
    shows the shrine and forest.
  - The 1600×900 RGBA16F half target is the problem. The pass that writes it (#3835's PS with five
    unexported PARAM inputs) outputs NaN at 100% of pixels. PixelHistory shows `shaderOut = NaN`,
    so blending is not the cause.
  - Compute `0x500a1b0000` combines the lit scene with that NaN image. The R11 scene colour it
    writes is 49% non-finite and 51% zero.
  - The 3200×1800 RGBA16F scene image that compute `0x500b220000` writes back is all zero at its
    next reader. The R11 image that reader writes is zero too.
  - The final compositor samples zero scene and zero bloom, so it outputs zero. Its 32³ LUT is
    populated (99.8% non-zero, mean 0.74).
- **Also seen, not investigated:** the lit scene target looks tiled. The complete view fills about
  the top-left two-thirds, and the right and bottom bands repeat parts of it.

## The merged-NGG LUT producer runs (2026-10-06, #3135 P5)

**Read this first.** Measured on Linux/RADV with `prosper-app` and the default launch
(`PROSPER_NULL_PAGE=1`, no input, `PROSPER_DBG=1`), 260 s, on branch `feat/ngg-live-p5`. The
merged ES+GS LUT producer (`es 0x5009440000`, chain `0x5009470000`) now runs through the
subgroup shell instead of being dropped.

| | before (same branch, producer refused) | after |
|---|---|---|
| refused vertex programs | 1 (the merged chain) | **0** |
| `dropped-draws` alarm | 46 of 48 windows; `shader-recompile/vertex:16874` | **did not fire** |
| `skipped-dispatches` alarm | 1 of 48 windows; `backend-declined:1` | 44 of 45 windows; `backend-declined:398` (#4625) |
| 32³ LUT `0x509cff0000` | never written | **131,072 bytes, 90,212 non-zero** |
| title-menu world | black | **still black** |

- **The LUT matches #3857's.** #3857's layered fallback read back 131,072 bytes with 90,212
  non-zero, byte-identical across its two arms. This path reads back the same size and the same
  non-zero count (`PROSPER_DUMP_PERSISTENT=ms:190000 PROSPER_DUMP_PERSISTENT_EXTENT=32x32`, which
  now reads volumes). The count was compared, not the bytes: #3857 kept no copy of its LUT.
- **Two things had to change for the chain to run live.**
  - The resource table is folded over the LINKED program. The prolog's own analysis stops at
    its `s_setpc` link, so it could not prove the raw register-offset V# loads at pc 33 and 66.
    On the linked program they are entry- and register-proven.
  - The shell's guest bindings are the ones it accesses. The direct V# at s8 is declared but
    never read.
- **The world is still black, so something downstream is also wrong.** One candidate: compute
  program `0x5007bc0000` binds the now renderer-written 64³ volume `0x5013f30000` as a storage
  image, and every such dispatch is skipped (#4625). Before P5 it ran over guest bytes no prosper
  draw had written.

## Linux/AMD shader refusals (2026-10-06)

**Read this first.** Counts are distinct refused programs (vertex / fragment / compute), all on Linux/RADV with `prosper-app`, a default launch, `PROSPER_NULL_PAGE=1`, no input and `PROSPER_DBG=1`. Runs are 260 s unless noted, and each row is its own run or runs, measured on that fix's branch. The counts vary from run to run with what the title streams: the vertex column reads 6 or 8 for the same code. The title picture is unchanged, the menu over a black world, which is the zero colour LUT (#3135).

| build | refused programs | what changed |
|---|---|---|
| main `9b223e64` (1 run, 240 s) | 26 / 25 / 5 | — |
| #4576 (3 runs, 240 s) | 6 / 6 / 2, 6 / 6 / 2, 11 / 12 / 4 | `sceAgcCbBranch` targets used to be folded at record time and paired with the previous fold's pipeline; the branch now runs in-stream |
| #4584 (1 clean run) | 8 / 0 / 2 | a V# read at two PCs under a clashed key left the second consumer without a resource (`unresolved-cbuf`). #4588 closes the unclashed variant and was not measured on Kena |
| #4587 (1 run, without #4584) | 6 / 5 / 2 | an `s_load_dwordx2` index pair as the source of a raw-wide register offset; vertex dropped draws per 5 s window fell from 4,294 to about 800 |
| main `b295a17a`, all of the above (2 runs) | 6 / 2 / 2, 6 / 1 / 2 | the fragment refusals are a new class, a base-0 T# in a direct sharp slot (#4592), absent from the earlier runs' scenes |

What remains (each run's refused shaders are dumped under `PROSPER_CAPTURE_DIR/refused_shaders_*`):
- **Fragment:** #4592.
- **Vertex**, diagnosed 2026-10-06; shares are of about 650 dropped vertex draws per 5 s:
  - **About 61%:** the merged ES+GS NGG chain (a 102-dword prolog linked to a 413-dword main). It is the same program as #3857's strip-layer producer, and it is refused at the main's `v_mbcnt` over a ballot mask, with LDS and `GS_ALLOC_REQ`. No proof rule fixes it; it needs the merged-NGG launch (#3135).
  - **About 28%:** a 1,526-dword program (three copies). Its register-offset wide load's pointer pair is rewritten after the load, which the whole-program pointer-stability rule refuses.
  - **About 11%:** two large NGG programs (pc 326/330). They hit the same pointer rule, plus the x1/x2 source proof refusing any branch before the source.
- **Compute**, diagnosed 2026-10-06:
  - **388 dwords:** a counted-loop route claims the program and refuses on a prelude EXEC do-while instead of falling back to the general CFG route.
  - **2,060 dwords:** three stacked CFG-dispatcher gaps (entry-M0 save over an ambiguous pair; a spill slot classified mask program-wide; a mask reassembled from two spilled halves). With all three applied in a scratch build, it compiles and passes `spirv-val`.

## Handoff to Linux/AMD (2026-10-05)

**The sections below predate the 2026-10-06 entry above.** Measured on Windows/RTX 4090 with the normal
`prosper-app` at 25% volume, one 150 s title-screen run per arm (needs `PROSPER_NULL_PAGE=1`). The
title menu still renders over a black world.

- **The largest dropped-draw cause is now vertex recompile, not Wave64.** After #4010's proven-vote
  lowering, fragment `subgroup-contract` refusals fell from 37,118 to 11,636 uses over comparable runs.
  `shader-recompile/vertex` became the top reason, at 26,211 draws on the pre-#4425 run.
- **#4425 (main `0ebb929b`) removed two phantom raw-wide refusals.** One counted image reads as guest
  writes; the other stopped the VCC walk at any VALU. Distinct refused NGG vertex programs went from
  45 to 23. The world is still black.
- **The remaining 23 are tracked in #4427.** 10 are blocked by the forward-only raw-wide proof (loops);
  13 pass every code-side proof but are refused live for an untraced reason. Triage with
  `shader_inspect <refused vs .bin> --raw-wide-proof`.
- **Linux/AMD goes first** (maintainer decision), because native Wave64 removes the NVIDIA emulation
  confound. The Windows-only Wave64 draft (#4384) is parked at `d23d963e`.
- **Open alongside:** #4426 (`PROSPER_DBG=1` stalls Kena at startup on Windows) and #4419
  (`DB_DEPTH_CONTROL` colour-on-depth bits are not applied).

## Current state — rung 2, gameplay reached with the world absent (2026-09-23)

On `main` (which has both crash fixes, [#3817](https://github.com/mattias800/prosper/pull/3817) and
[#3814](https://github.com/mattias800/prosper/pull/3814)), `screenshot.exe` with the route
`prosper/scripts/kena/reach-first-gameplay-prompt.pad` reaches, in order:

| t (one 540 s run) | what renders |
| --- | --- |
| 0-70 s | black (the Ember Lab logo movie is decoded and delivered to the guest but never visible — open) |
| ~80 s | **main menu** — `New Game / Load Game / Options`, `Version 1.16`, over black |
| ~160 s | **Choose Difficulty** |
| ~170-220 s | **brightness calibration** (the map art and slider render) |
| ~240 s | loading spinner — the first level loads |
| ~270-330 s | **intro narration** (*"Unique wooden masks are carved to honor the dead…"*), legible, with the blue *Spirit Guides* highlight |
| ~410-540 s | the first gameplay prompt, *"Press … to Pulse"* — **over a black world** |

The guest is in the game loop and asking for gameplay input, but the scene behind the prompt does not draw
(6,817 non-black pixels of 8.3 M per sample, all of them the prompt). Under the ladder that is **rung 2**, not rung 3.

**Likely cause of the black world, not yet established per draw:** the title's fragment shaders need wave64.
The run logs `[render] skip draw=... fragment shader requires subgroup size 64 (device range 32..32 ...)` for 402
distinct shaders (400 `why=0x2 wave-any`, two `lane-id`), and an NVIDIA device is 32-wide, so every one of those
draws is skipped — the cross-title gap [#2147](https://github.com/mattias800/prosper/issues/2147). The main menu's
missing animated background is probably the same thing. Confirming it needs either a 64-wide (AMD) run or a
census of which draws in a gameplay frame are skipped; neither has been done.

Every menu frame also alternates with a fully black composited frame in about half the samples; not investigated.

### Linux/AMD title-menu boundary (2026-09-24)

The live menu still shows its wordmark and UI over black. A retained 1600×900 RGBA16F target on the normal
render path contains nonzero scene data (one observed target has every RGB channel at half-float maximum), while
the later 3200×1800 compositor output is black. Other same-extent retained images contain recognizable forest
content. The saturated target is not by itself evidence of the correct scene: raw half-float values and converted
BMP previews have different meanings. The ordered submit capsule is explicitly **unverified** because it was not
matched to its own presented frame; its draw/dispatch lineage is useful for localization, not a visual oracle.

In that capsule, the final compositor samples a zero 32³ LUT at PS binding 39. The immediately preceding LUT
producer is unrealized because its merged-NGG vertex chain hits the existing wave/mesh gate
[#3135](https://github.com/mattias800/prosper/issues/3135). A corrected **one-draw-only** live bracket
confirms the boundary on current main: the final compositor's retained scene and half-resolution inputs have
visible content, its bound LUT has 0/131,072 nonzero decoded bytes, and the executed draw turns its 3200×1800
target black. Replacing only that draw's LUT sample with a constant or its final output with its existing scene
sample makes the menu endpoint nonblack. Those are diagnostic interventions, not rendering fixes: the constant
produces an almost white image, and the direct sample is a distorted gradient. The earlier saved-module override
changed over a hundred draws and its black result was not an isolated LUT experiment.

A fresh same-binary v58 ordered capture of the missing producer records **4 vertices × 32 instances**. The
old capture omitted failed-draw instance count and now reports it as unknown when opened by current replay.
The producer's terminal PARAM output is transformed after shared-memory reads, so prosper's proven no-GS
per-vertex shortcut rejects it. A mesh/workgroup execution path must preserve its cross-lane and LDS behavior;
lifting the vertex `v_mbcnt` gate would not do that. The ordered captures remain unverified against their own
presented frame. The live same-draw bracket establishes the bound zero LUT and black output independently.

The compile-only four-wave NGG probe now offers a raw VGPR value plus execution-hit trace at a selected linked
instruction. On a **speculative** one-subgroup input for this producer, it exposed a bounded DPP reduction bug
in the probe's native Wave64 dispatcher: a `BOUND_CTRL` marker was passed as part of the lane shift, so the
reduction never collected neighboring lanes. With that corrected, the same input changes from zero to 32
nonzero raw PRIM records. A byte-matched two-instruction synthetic test guards the reduction and wave isolation.
The referenced vertex slots clarify those records: their packed indices cover 96 distinct slots, with
nondegenerate clip-space triangles and two candidate triangles on each of layers 0–15. Supplying instance
indices 16–31 to a second otherwise-identical offline subgroup produces the corresponding layers 16–31;
changing only the proposed primitive IDs leaves the readback byte-identical. The earlier count of just six
POS1.z values examined only lanes carrying PRIM records and missed the vertex records they reference.
These are still **supplied** launch inputs, not validated hardware subgroups, primitives, pixels, or a title
fix. The captured blobs are callback-time copies and the live renderer still rejects the producer. #3135
remains open.

The LUT fragment program has a separate missing input. A live exact-program trace records
`SPI_PS_INPUT_ENA=SPI_PS_INPUT_ADDR=0x2020`, enabling fields 5 and 13. Its first vector instruction,
`v_bfe_u32 v3, v2, 16, 11`, extracts the array index from the ancillary VGPR (v2 after field 5's
two VGPRs). The fragment recompiler previously reserved that register but left it at zero, and a
private execution of the exact live fragment module with supplied full-screen geometry produced
32 byte-identical LUT slices. A controlled fragment `Layer` input and view-base specialization
produced 32 distinct slices in that fixture; neither result establishes the correct guest LUT.

The four eight-slice batches were **fixture-only**, not production behavior. Live Kena passes one
32-slice view to the backend, which refuses mesh draws exceeding this device's
`maxMeshOutputLayers`; live draw construction does not yet wire the guest NGG chain into MeshEXT.
Partitioning will need a proved mapping for guest workgroups, primitive layer output and side
effects. The guest view's starting slice and an internal host batch offset must remain distinct;
the fixture's base values cannot be copied into production without that contract.

A targeted live state log for that same producer records `SPI_SHADER_PGM_RSRC2_GS=0x008b0000`
(2,176 LDS dwords, 8.5 KiB), `VGT_GS_ONCHIP_CNTL=0x10020040`
(64 ES vertices, 64 GS primitives and 64 instanced primitives per subgroup using the
[gfx10 register layout](https://chromium.googlesource.com/chromiumos/third_party/mesa/+/refs/heads/stabilize-13982.70.B-chromeos-amd/src/amd/registers/gfx10.json)),
`GE_NGG_SUBGRP_CNTL=1`, `VGT_GS_MAX_VERT_OUT=3`,
`CB_COLOR0_VIEW=0x00040000`, and `CB_COLOR0_ATTRIB3=0x4606c01f`. The latter's
`MIP0_DEPTH=31` encodes 32 slices and its resource-type field is 2; the immediately following
consumer binds the same address as a 32³ texture. PR #3842 added a generic retained 3D color-target
path; the shader and draw launch still need to produce valid layered output for it. The slice-max
register is recorded raw rather than interpreted as a layer count until its boundary convention is
established. On this AMD host, a standalone mesh draw rendered 32 layers in four eight-layer batches
with Khronos validation loaded and no reported errors. This proves only that the host API route exists:
the guest wave/primitive mapping and correct LUT values still need implementation and game verification.

An exact-program-and-output-shape live witness on the 32×32, 32-instance LUT draw reported the same
state eight times: `SPI_SHADER_PGM_RSRC1_GS=0x622c0047`, `RSRC2_GS=0x008b0000`,
`VGT_GS_ONCHIP_CNTL=0x10020040`, `VGT_ESGS_RING_ITEMSIZE=4`, and `GE_NGG_SUBGRP_CNTL=1`.
The `GS_VGPR_COMP_CNT` and `ES_VGPR_COMP_CNT` fields are both 3, so the hardware loads v0–v3 and
v5–v8. `VGT_SHADER_STAGES_EN=0x2030` has `GS_W32_EN=0`, selecting Wave64. The separately guessed
packed GS offsets use units of four, matching this programmed ring item size; this does not validate
the exact lane allocation. An independent review found the first logger wrongly read `GE_CNTL` and
`VGT_PRIMITIVE_TYPE` from the context file: both are **UCONFIG** registers. A corrected live run
records `GE_CNTL=0x00008040` (both programmed group-size fields are 64),
`VGT_PRIMITIVE_TYPE=6` (the capture reports Vulkan triangle strip),
`GE_MAX_OUTPUT_PER_SUBGROUP=0x000000c0` (192), `VGT_GS_MAX_VERT_OUT=3`, and raw
`VGT_GS_OUT_PRIM_TYPE=2`, all present. These are programmed bounds and types, not proof of the
actual subgroup split. The original program-only diagnostic filled its eight-line cap on other
draws using the same shader, which is why the extent/instance filter is needed.

An offline raster check follows the 9-bit indices of every PRIM record in the two **supplied**
16-instance groups. It finds two opposite-winding, full-screen triangles on each candidate layer
0–31, with no uncovered 32×32 pixel sample centers and PARAM.xy matching the normalized screen
coordinates at every referenced vertex. The captured failed draw has `raster=0/0/0` (no culling,
CCW front, fill), so the opposite windings do not by themselves discard half the lookup. A
wrong packed-offset input fails this checker with mixed-layer primitives. This is candidate
geometry only: the actual hardware launch, fragment output, and live title pixels are still
unverified. The raster-field print is included for failed draws in `gpu_replay --inspect-only`,
which previously printed it only for realized draws (#3135).

The input-default probe logged changes to earlier fragment programs `0x3007c60000` and `0x3007c80000`
paired with vertex program `0x3007060000`; it did **not** change the final compositor's input wiring.
A separate pair trace of `0x3007060000` with fragment `0x30096e0000` found five guest-programmed
`SPI_PS_INPUT_CNTL_0..4=0` values and header-derived constant defaults. The guest writes these registers
through indirect **physical** offsets (17,420 observed writes, none through the virtual AGC bank).
That vertex program exports PRIM/POS0 without PARAM exports; it is a fused front shader, with no hidden
linked back/chain body. This traced pair is also **not** the final compositor. The captured compositor
vertex SPIR-V stores Locations 0–3. Trace the exact earlier pass changed by the probe and establish its
PS5 input contract before changing generic interpolation semantics.

## Reproduction

```bash
PROSPER_NULL_PAGE=1 PROSPER_GUEST_ARGS= \
PROSPER_SAVE0=<FRESH>/save0 PROSPER_SAVEDATA_DIR=<FRESH>/savedata \
PROSPER_PAD_SCRIPT=@scripts/kena/reach-first-gameplay-prompt.pad PROSPER_PAD_SCRIPT_LOG=1 \
  ./build-mingw-app/screenshot.exe <DUMP_ROOT>/PPSA01802-app0 --seconds 10 --count 54 --timeout 560 --out <OUT>
```

`PROSPER_NULL_PAGE=1` and an empty `PROSPER_GUEST_ARGS` are the UE4 recipe (`DRAGON_QUEST_STATUS.md`). The route's
anchors are pad reads — see `prosper/scripts/kena/README.md`. `tools/dump_hygiene.py` on the dump is clean; its
`_original_files/` (the untouched encrypted originals) is retained and never read.

The launch hang before the first frame (no raw scanout either; recorded here as "about 2 in 12", measured
6 of 28 on 2026-10-05) is fixed on Windows — see item 4 below. So is the boot-time crash that took about 1 launch
in 5 down 5-10 s in with `0xC0000005` and nothing in stderr — item 5.

## What was fixed to get here

1. **`sceAvPlayerStartEx` was unregistered** ([#3781](https://github.com/mattias800/prosper/issues/3781), merged
   as [#3788](https://github.com/mattias800/prosper/pull/3788)). The title adds its logo movie with
   `auto_start = 0` and starts it only through this call; the dispatcher's `return 0` answered `SCE_OK` without
   starting anything, and the title waited forever on a black screen. 0 of 2 → 2 of 2 runs reach the menu.
2. **A fixed `sceKernelBatchMap` MAP_DIRECT failed over pages prosper had lazily committed**
   ([#3812](https://github.com/mattias800/prosper/issues/3812), PR [#3814](https://github.com/mattias800/prosper/pull/3814),
   Windows only). Host-side reads first-touched unmapped pages of the guest's own reservation; the VEH replaced
   those placeholder pieces with private memory, and `MapViewOfFile3` cannot place a view over them. The guest got
   ENOMEM and UE4 aborted (`sceKernelBatchMap failed with error code: 0x8002000c`) 50-65 s in, 5 of 6 runs. A fixed
   map now releases the range's guest-owned contents first, as `mmap(MAP_FIXED)` does on Linux.
3. **The recompiler emitted invalid SPIR-V for a fragment `s_bfm_b64`**
   ([#3816](https://github.com/mattias800/prosper/issues/3816), PR [#3817](https://github.com/mattias800/prosper/pull/3817)).
   Its lane test used `linear_localid`, which is id 0 in a fragment module; NVIDIA's driver dereferenced it while
   compiling the pipeline and the process died inside `nvoglv64.dll` on the first level load, 4 of 4 runs.
4. **The launch hang: the host's own allocations sat inside the guest's address window**
   ([#4426](https://github.com/mattias800/prosper/issues/4426), Windows only). Every prosper executable was
   linked with `HIGH_ENTROPY_VA`, under which Windows draws the base of the process's bottom-up allocations
   (stacks, heaps, every later `VirtualAlloc(NULL)`) from the low terabyte — the guest's window. UE4's 512 GiB
   MallocBinned3 arena has to cover `[0x7c00000000, 0xa000000000)` wherever it lands, so a host allocation there
   got it refused with ENOMEM; the title then re-entered `FMemory::GCreateMalloc` with `GMalloc` null and
   blocked forever on that function's own `__cxa_guard`, before its first print. 6 of 28 launches as built,
   0 of 12 with only that header bit cleared in the same binary. Executables that boot a guest are now linked
   without the flag and steer their later allocations into `[4 GiB, 16 GiB)`, best effort
   (`docs/platforms/WINDOWS_PORT_HANDOFF.md` § Gotchas); the refusal is reported (`[memhle] reserve FAILED …`)
   instead of silent. This applies to every UE4 title, not this one.
5. **The boot crash: prosper delivered an APR completion counter below one already delivered**
   ([#4504](https://github.com/mattias800/prosper/issues/4504)). UE4's listener walks `last+1 ..= cnt` and then
   stores `last := cnt` unconditionally, so a late lower counter rewinds it and the next event completes a token
   a second time; its tracking entry is gone and the guest dereferences null (`eboot+0x16bd26b`). Two deferred
   posts could read the ring's high-water mark as N and N+1 and post them in the opposite order. Reading the
   mark and posting it are now one step. Before: 12 of 64 launches died this way (the 64 include launches that
   hung first, so the rate among booted launches was higher). With the fix: 0 of 21 booted launches faulted, and
   a detector built for that measurement (not in the tree) saw no out-of-order post in any of them, against 4 of
   12 with the ordering removed. Shared code: any UE4 title using this listener.

## Ruled out

- **The missing sunlight comes from the shadow projection shadowing every pixel** — false as the
  cause. The projection did cover every pixel (the unimplemented depth-bounds test, fixed with
  #4703's companion change). With the cascades partitioned, the sun pass still adds exactly 0,
  because its lighting-channel test fails (#4703, 2026-10-07).
- **The CSM PCF reads the wrong texels because the gather offsets decode wrongly** — false. The
  recompiled SPIR-V decodes offsets 0x3e00, 0x3e3e, 0x3e02, 62, 2, 0x23e, 0x200 and 0x202 to the
  standard (−2..2) 3×3 PCF grid. The shadow atlas holds casters (cascade 0 median depth 0.468)
  (2026-10-07).
- **The deferred lights skip the scene because the GBuffer shading-model bits are lost** — false.
  GBufferB alpha is 177 (DefaultLit, id 1) on 97% of view pixels at the light draw. The early-out
  is the lighting-channel AND, not the shading model (#4703, 2026-10-07).
- **The `shader-recompile/fragment` drops are one refused Wave64 program that needs a recompiler
  feature** — false. Most were AGC helper rectangles compiling the pixel shader the previous draw
  left bound, against that draw's stale user data. The refused program changed from run to run
  (295 dwords, then 182) and was always on `es=0x5007020000`. The helper has no depth/stencil
  effect, so the draws had nothing to lose. The remainder was a separate counted-loop VCC_HI
  refusal (2026-10-07, #4680).
- **The fog, the washed-out grading, the missing light shafts or the missing foliage come from the
  dropped multi-target volume passes or the dropped merged-NGG producer** — false. With #4643 both
  `backend/volume-multi-target` (9,984 per 260 s on `main`) and `backend/ngg-subgroup` (4,608) are
  zero, six 64³ volumes hold content, and the title-menu frame changes in 0.7% of pixels (candle
  glow and particles). The fog, the grading and the missing foliage are unchanged (2026-10-07, #4643).
- **#3835's all-NaN 1600×900 half image is a float-semantics or interpolant defect in the
  recompiled pixel shader** — false as the cause of the black title world. It was AGC's helper
  rectangle drawn with the pixel shader the previous draw left bound. That shader reads PARAMs the
  helper never exports, and hardware writes no colour for that draw. #4610 suppresses the
  helper's colour. With that suppression skipped in one scratch binary, 2 of 2 frames are black;
  with it on, 2 of 2 show the world. What this leaves open: the value hardware returns for a
  selected PARAM that was never exported. No draw in a current title frame depends on it
  (2026-10-07, #3835, #4610).
- **The lit scene being "tiled" (the view filling the top-left two thirds of its target) corrupts
  the picture** — false. It is UE4's screen percentage: the next pass upscales the view to a
  full-frame 3200×1800 image, and the final frame is correct (2026-10-07, #3835).
- **Kena's black frames from about t=100 s under #4610 (first-gameplay route) are caused by the BOOL64
  polarity** — false. They came from Kena's own compile of AGC's helper rectangle. #1588's exact list did not
  recognise it, so the eliminate-fast-clear pass painted the inherited pixel shader over the finished scanout.
  With the rectangle recognised, the forest loading art renders through 320 s on the same polarity (#4610).
  What this does NOT show: whether the polarity is right for Kena. A recognised helper writes nothing under
  either polarity, so this result is polarity-blind. Kena's predicate words were not traced, and no per-helper
  lever was run on Kena.
- **The skipped translucency-lighting clear (#4625) blacks out the title world** — false. With
  the claimed volume published and the clear running every frame, `skipped-dispatches` fell from
  `backend-declined:336` to `backend-declined:1` (a different 512³ program), and the world is still
  black (2026-10-06, #4625).
- **The scene is never lit, or the black comes from the compositor or its LUT** — false on this
  build. In a RenderDoc capture the lit 3200×1800 R11 scene shows the forest and the compositor's
  LUT is populated. The loss comes earlier: a 1600×900 half target is NaN at every pixel
  (`shaderOut = NaN`, so not blending), and it is mixed into the scene colour (#3835, #4625).
- **The black title-menu world is the zero colour LUT alone** — false. With the merged-NGG
  producer running and the 32³ LUT read back live at 131,072 bytes / 90,212 non-zero (#3857's
  figures), the world is still black (#3135 P5, 2026-10-06).
- **The live chain's prolog-only resource table is enough for the subgroup shell** — false. The
  shell refused at prolog pc 6 (`unresolved-operand`) and, with pc 6 supplied, at pc 33. The
  raw register-offset V# loads at pc 33/66 are provable only on the linked program, which is now
  what the NGG candidate's table is folded over (#3135 P5).
- **The black pre-menu screen is a renderer failure** — false. The composite was black because the logo movie was
  never started (unregistered `sceAvPlayerStartEx`, #3781); registering it reaches the menu with no renderer change.
- **The BatchMap ENOMEM is #2424's too-small free placeholder** — false. `VirtualQuery` on the failing range shows
  `MEM_COMMIT | MEM_PRIVATE` pages in 16 KiB pieces with non-64 KiB allocation bases, and the registry holds
  **0** free placeholders at that moment; it is prosper's own lazy commit, not a placeholder size mismatch (#3812).
- **The guest mapped or touched the failing range before the fatal map** — false. `PROSPER_MEMLOG=1` shows no guest
  map, flexible map or unmap of that range before the fatal MAP_DIRECT, and a RIP-tagged log of every VEH lazy
  commit (5,120+ in 145 s; the first 60 and every 1024th logged, 65 lines) found **all 65** in a host
  system DLL's copy routine and none in guest code (#3812).
- **The level-load crash is an NVIDIA driver bug** — false as the root cause. Of all 11,710 graphics SPIR-V
  modules created in a crashing run, `spirv-val` rejects exactly one — the last one created — and its only
  defect is an operand naming id 0 emitted by prosper's `s_bfm_b64` lowering (#3816). The driver crashing on
  invalid SPIR-V rather than rejecting it is a driver robustness gap, not the cause.
- **`PROSPER_DBG=1` stalls this title at startup** — false. In an interleaved series the stall and the crash
  both occur without it (control 1 stall + 1 crash of 8; `PROSPER_DBG=1` 2 + 2 of 8), and the stall happens
  before any code that reads the variable runs. The three consecutive stalls that prompted #4426 were a small
  sample of two unrelated intermittent defects (items 4 and 5 above).
- **The UE4 arena landing at `0x2000000000` instead of `0x7c00000000` is caused by `PROSPER_DBG`** — false.
  Control runs land at both. It is the reservation's whole-window fallback, taken when a host allocation
  occupies the top band; the same cause as the launch hang, one notch less severe (#4426).
- **Seconds-anchored pad pulses are enough for this title's menus** — false. An 11-press seconds-based route
  delivered 2 presses, because the guest reads the pad about once a second there and a 300 ms pulse mostly falls
  between reads; pad-read anchors (`pN`) deliver every press.
- **Any nonzero 3D LUT makes the title oracle-correct** — false. An exact one-draw constant-LUT control changes
  the live compositor and menu endpoint, but the result is nearly white. Directly outputting the already bound
  scene sample instead gives a distorted gradient. The LUT producer is a real blocked dependency; the scene
  feeding it is also spatially wrong, so both paths need separate repairs (#3135, #3835).
- **The traced earlier fragment inputs are zero because virtual AGC register translation dropped their writes**
  — false for the traced `0x30096e0000` pair. A path-labelled register watch observed 17,420 direct physical
  indirect-array writes to the controls and zero virtual-bank writes (#3835).
- **The traced earlier vertex program's missing PARAM stores are in an unlinked back half** — false for its
  `0x30096e0000` pair. Exact-pair logging names one fused front program, no chain, and no linked second body;
  its decoded entry ends without a PARAM export. This says nothing about the separate final compositor pair (#3835).

## Next

- Confirm that the black gameplay world (and the menu background) is the wave64 skip, on a 64-wide device or with
  a per-draw skip census of a gameplay frame; if it is, the title waits on #2147.
  A Linux lane has since traced the menu scene ([#3813](https://github.com/mattias800/prosper/pull/3813)): the
  forest is present in a retained MRT3 image that a one-block fragment shader samples at binding 37 — so on Linux
  the background scene is rendered somewhere. The Linux title-menu boundary above now narrows the later loss;
  do not assume the Windows black background is solely the wave64 skip. Resolve the generic missing-PARAM
  contract and merged-NGG LUT producer separately, then compare a live default-path image with the oracle.
- Host-side renderer reads of never-mapped guest memory (#3820), and `SCE_KERNEL_MAP_NO_OVERWRITE` (#3819).
- The invisible logo movie: 150 frames decoded and delivered, none visible; delivered at ~2x real time.
- The early-boot hang (about 2 in 12 launches).
- Derive the actual ES/GS subgroup allocation and every per-wave `s3`/v0..v8 value for the 4×32 draw.
  The current source-bounded **candidate** uses two 64-ES/32-GS subgroups and produces two offline triangles
  on each of 32 candidate layers. An output pattern that fits the target is not a hardware launch oracle;
  require a wrong-partition control and independent ABI evidence before live shader admission (#3135, #2072).
- Lower the proved output into MeshEXT with the existing retained 3D target path, preserving the 96 indexed
  vertex slots and 32 primitive records **per candidate subgroup**, then compare the live default-path
  title frame with the oracle. Devices without the optional subgroup/mesh features must continue to decline
  safely. The separate scene/interpolation defect (#3835) may still affect visual correctness.
