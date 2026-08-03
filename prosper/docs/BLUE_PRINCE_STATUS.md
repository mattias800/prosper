# Blue Prince (PPSA25009, Unity/IL2CPP) — status & investigation map

Last revised: 2026-08-03 (snapshot route/reference triage; see Ruled out). Prior history:
#1216 (closed — boot/blank-frame era), #1264 (Day One hold investigation), #1287 (visuals umbrella,
open).

## Ladder position

**Rung 3 — gameplay with real GPU draws, sustained.** The scripted fresh-save route plays the
intro cinematic, loads Day One, walks the manor approach, opens the front door, and explores
Mount Holly's entrance hall/vestibule with the footstep HUD advancing. Materials and the display chain are now hardware-faithful offline (families 1-3
resolved 2026-07-26); rung 4 needs the live oracle side-by-side plus families 4-5.

Reaching the frame loop uses the standard gated switches:

```bash
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_IME_AUTOKEY=1 PROSPER_RENDER=1 \
  ./build-linux/screenshot <root>/PPSA25009-app0 --seconds 3 --count 650 --timeout 2000 --no-manifest
```

`PROSPER_IME_AUTOKEY=1` answers the save-name IME; the route needs no pad script. Timing:
intro cinematic ≈ shots 20–70, Day One yard ≈ 90–115, front door/vestibule ≈ 115+ (plain run);
capture-armed runs pace 2.4–4× slower — budget `--count 730` to pass renderer submit 3300.
The `blue-prince-title` snapshot route guards the title screen. The `blue-prince-hall` route guards
the native 1920×1080 entrance hall with a progressing, cross-run-validated content plateau; its
reviewed live capture is published as `assets/screenshots/blue-prince-hall.png`.

## Landed fixes this arc (evidence on the linked issues/PRs)

- **#1335/#1344** — the Day One frozen/blown frame: ONE stale arena slot folded
  `(PA_SC_SCREEN_SCISSOR_BR, 0)`; degenerate-SCREEN-pair recovery in the scissor combine.
- **#1349/#1350** — `PA_SU_POLY_OFFSET` depth bias decoded + applied (capture v29 retains it;
  `gpu_replay --inspect-only` prints per-draw `bias=`); #1351/#1357 hardening.
- **#1352/#1354** — the per-light shadow-plane destroyer: light-loop rects with
  `DEPTH_CLEAR_ENABLE` + `DB_DEPTH_CONTROL==0` were treated as clears and clobbered the just-
  rendered shadow maps via sampled-bridge recency; clears now act only through the enabled write
  path. Restored real per-texel shadow structure (marble floor, sconce pools, stained-glass
  door detail). Stencil twin: #1355/#1359; initial-value contract: #1361/#1362.
- **#1353/#1363** — shadow casters decoded `(Z_READ=P, Z_WRITE=0)` from another stale arena slot;
  lone-zero DB base halves now recover from their partner (GFX10 equal-bases contract), unifying
  the persistent-DS identities to `(P,P)`.
- **#1364/#1368** — the apparent family-level arena problem was an unwritten-output problem, not
  an undocumented mixed-record format. The two suspicious stretches are exactly the 32-record
  blocks advertised after calls to the Cobra-discovered `dbOlWdppb4o` interpolant helper. Before
  #1368 that NID was a success stub, so the blocks retained stack contents: import and eboot return
  addresses, heap pointers, and host pointers split into `(offset,value)` dwords. #1368 recovered
  the SDK alias and initializes every advertised `SPI_PS_INPUT_CNTL_0..31` record; its focused test
  poisons all 32 outputs first so a partial write cannot pass. #1411 separately resolves the other
  live shape of that alias, Dragon Quest VII's virtual `0x10000000+n` offsets, at fold time. Two
  fresh-save, scale-1 current-master startup routes ran through Blue Prince's intro for 380+ seconds
  without a `[dbbase-clobber]` event; the quiet run produced 76/76 captures with 76 distinct source
  frames.

- **#1427/#1429** — the silent 1 MiB buffer-upload clamp that erased 44 of 248 entrance-hall
  draws (see family 5). Uploads now follow the descriptor's declared range; any short upload is
  reported (`[buffer-truncated]`), and `PROSPER_MAX_BUFFER_UPLOAD_MB=N` lowers the ceiling so one
  build reproduces the collapse and the fix back to back.

- **#1691/#1703** — the decoded-texture identity map was scoped to one graphics span, and this title
  cuts every frame into 21-22 spans at its interleaved dispatches, so each span re-resolved every
  identity from scratch. It is now scoped to the submit, with cross-span reuse gated on the ordered
  in-submit write journal plus a re-check that the entry's source range is still the range the
  binding resolves. Two reversed-order routed A/B pairs on an otherwise idle box: persistent-cache
  resolutions **76.7 -> 41.3 per submit (-46 %)**, reproducible to the digit, and heavy-scene frame
  time **246.6 -> 228.4 ms** and **248.1 -> 228.8 ms** (~4.04 -> ~4.38 submits/s).
  **The frame-time gain does not come from the texture terms.** Per-submit, across the two pairs:

  | term | pair 1 (span -> submit) | pair 2 (span -> submit) |
  |---|---|---|
  | `build_resources` | 47.60 -> 47.32 ms (**-1.0 %**) | 47.28 -> 48.31 ms (**+0.9 %**) |
  | `texture` (within it) | 7.31 -> 7.17 ms | 7.15 -> 7.34 ms |
  | `backend` | 165.91 -> 165.50 ms | 170.3 -> 167.3 ms |
  | **`pass_control`** | **35.20 -> 15.78 ms (-19.4)** | **28.76 -> 14.82 ms (-13.9)** |

  The sign **flips** between pairs in the term the change touches while the magnitude stays
  consistent in a term it does not — that pattern is why the mechanism is recorded as open rather
  than settled, and why the numbers are here instead of the word "unexplained". Either it is a real
  second-order effect or `pass_control` is misattributing; both are worth knowing, and the next
  person should be able to tell them apart without re-running the route. The plausible candidate is
  allocator behaviour (the span-scoped arm rebuilds the identity map 16.7x per submit while the same
  pass loop allocates an 8.3 MiB frame buffer per pass), but it has **not** been measured — do not
  promote it to an explanation. Host cache footprint is byte-identical between arms (16 scratch
  slots, 111 persistent entries, 574.3 MiB), so it is not a working-set effect.

  **This does not address the reported ~2 fps.** At the time this landed, #1284 decomposed the same
  title's submit as 263.33 ms = `draw_setup.resources` 95.42 + `readback` 79.16 + `gpu_wait` 29.29 +
  `record_upload` 22.58; this change moved a ~7.5 % term and left all four intact. Note also that the
  issue's headline **15.2x is a replay figure** — the persistent decode cache is disabled by
  construction in replay — and live only ~77 of ~5,500 texture references per submit ever reached it.
  **Those four numbers are now superseded**: re-measured 2026-08-02 (see the #1284 bullet below), three
  of the four have collapsed and only `draw_setup` remains. The title is still CPU-bound in the
  frontend, which is the part of this paragraph that held.

- **#1284 (open) — the frame-time term is backend storage-buffer upload, not textures or
  descriptors.** Re-measured on `3a473bca` (post-#1292, post-#1703), 35 peer-free heavy windows:
  the backend submit is **165.18 ms** (was 263.33 in the issue body), and **every term that
  decomposition named has collapsed except `draw_setup`** — `readback` 79.16 -> 14.71,
  `record_upload` 22.58 -> 2.73, `gpu_wait` 29.29 -> 6.71, fence waits 61 -> 16.3, while
  `draw_setup` is 119.87 -> 126.81 at 1.39x the draws. `gpu_device` is **4.31 ms inside 165 ms**, so
  the GPU is 2.6 % busy and this title is bound in prosper's own per-draw CPU path.
  Sub-attributing `draw_setup.resources` (89.66 ms, 54 % of the submit) live gives
  **`res.buffer` 93.7 %**, `res.descriptor` 2.0 %, `res.texture` 1.8 %. The frontend resolves every
  binding to a zero-copy direct guest view (`logical=1,764.4 MiB materialized=0.0 MiB` per submit)
  and the backend copies the deduplicated set into host-visible staging anyway, while
  `backend_buffer_pool` sits pegged at its 256 MiB ceiling with evictions exactly equal to misses.
  Full numbers, method and the four falsified sub-hypotheses (descriptor setup, live texture cost,
  per-draw `getenv`, the #1268 content hash) are in
  `docs/RENDERER_PERFORMANCE_2026_07.md` § *Blue Prince 3D submit decomposition, re-measured*.

  **The immediately actionable part: `PROSPER_BACKEND_BUFFER_POOL_MB` defaults to 256 MiB and this
  title's host-staging working set is 974 MiB**, so the pool thrashes — evictions exactly equal
  misses (~216k each) and one buffer is destroyed for every one created. A four-arm alternating A/B
  (256/2048/256/2048, one binary) takes the submit from 203.06 to 125.98 ms, **−34.8 % normalised
  per draw**, with `res_buffer` −63 % and `cleanup` −80 %; at 2048 MiB the pool settles at 974 MiB /
  503 buffers with **zero** evictions. All arms were contended, so confirm the magnitude on a clean
  box before quoting it. The fix (memory-aware default + LRU eviction) cannot change rendered
  output — the same bytes are uploaded from the same sources; only host staging recycling changes.

## Defect families (#1287) — families 1–3 and 5 RESOLVED 2026-07-26

Families 1–3 and 5 below are **fixed on master**; they are retained as the resolution record so
the evidence is not re-chased. Only family 4 remains open. Both #1427 remainders tracked in #1435
are resolved: #1440 added the ctest-resident truncation guard, and the residual 27 vanished draws
were confirmed as legitimate frustum culls.

1. **RESOLVED — concentric ring/band moiré** radiating from lights. It was never
   light-accumulation math: the ring structure was the *palette/material atlas* sampled through
   wrong coordinates and then modulated by lighting. Three landed fixes compose to the cure —
   **#1411** (Gen5 virtual interpolant registers resolved at fold time; this is the decisive one:
   the palette-UV interpolant reached the FS through a virtual register that resolved wrong, which
   is why the vertex buffer AND the runtime SSBO both audited byte-correct while the fragment
   stage observably received position-like sweeps), plus **#1399** (`IMAGE_SAMPLE_D` honors the
   guest's explicit gradients instead of falling back to implicit LOD — the fine moiré overlay)
   and **#1401**/**#1404** (LINEAR-S# dref lowered as real compare-then-filter 2×2 PCF with the
   decoded address modes, instead of a hard-threshold single tap). Offline proof:
   `lit_frame.prgcap` replayed through `gpu_replay --recompile-raw` renders smooth navy walls,
   real carpet-runner art, and no rings (frame:
   `docs/screenshots/issue-1287-hall-materials-fixed.png`, #1287 comment 2026-07-26).
   The prior "lives in the light-accumulation/falloff math" reading was wrong — the ruled-out list
   (shadow-plane content, caster bias, mips/LOD, sampler filters) was correct but incomplete;
   the interpolant path had not been suspected.
2. **RESOLVED — overbright/white blowout** of lit surfaces. Same root as family 1 (wrong sampled
   material energy, then amplified by the light loop), plus the display-chain half in family 3.
3. **RESOLVED — full-scene magenta flip**: NOT game intent. **#1334/#1382** — CB MODE=RESOLVE
   shared only CPU pixels while the destination inherited `gpu_valid=true` over a never-written
   persistent image; after the #780 CPU-copy discard the compute tonemap imported the stale image
   and read black, the post chain cascaded, and the present fallback published raw FP16 HDR under
   a plain linear clamp — which reads as magenta. The resolve now GPU-copies into the destination.
4. **OPEN — floating black rectangles** at wall-lamp positions; **RGB-noise ball** center-hall.
   Untriaged.
5. **RESOLVED — the "dark slab" was a whole-room geometry loss** (#1427/#1429), not a misplaced
   surface. `build_R` clamped every non-texture buffer upload to 1 MiB *in silence*; a vertex fetch
   indexes anywhere inside its descriptor, so every element past the clamp read zeros, those
   vertices transformed to one clip point, and their primitives died as degenerate. On this scene
   that erased **44 of 248 draws** — the checkered tile floor, the far table with the flower pot,
   most of the room — and what survived read as a sloped plane hovering over the missing floor.
   Buffer size correlates exactly (454 KiB healthy; 1.45-3.0 MiB collapsed; the 1.66 MiB stream
   that straddled the clamp lost precisely the vertices past it). Uploading the declared range
   drops vanished draws to 27/248 and renders the hall at oracle parity
   (`docs/screenshots/issue-1427-oracle-before-after.png`). The old bisection range recorded here
   (operations 1100-1200) was a red herring produced by the isolated-draw depth trap — do not
   resume it. The residual 27/248 `GEOMETRY-VANISH` draws were each checked with
   `PROSPER_GEOM_PROBE`: all wrote their expected finite vertex count, retained real positional
   spread and healthy topology, and landed wholly outside the clip volume or behind the camera.
   None had the pre-fix `unique-pos<=2` collapse signature; the repeated transparent-pass meshes
   reproduced the same off-screen bounds as their opaque passes (#1435). The bouquet's
   confetti-coloured flowers remain part of family 4.

## Capsule & artifact inventory (`~/bp-1264/`, all run-local addresses)

- `hall.prgcap` — entrance hall, submit 3302, pre-v29 (no bias state), exact live parity
  `41cc4a6524492a6a` at capture time; post-#1352 replay `a56192c29c21c375`. THE workhorse for
  offline light-pass iteration (its light draws carry full scissors).
- `hall3.prgcap` — v29 sibling; **bad lighting oracle**: 1299/1533 draws genuinely
  empty-scissored that frame (per-pass vport closes), so its lighting rides RTT seeds.
  **Always check a capture with `gpu_replay --inspect-only | grep scissor` before trusting it
  for lighting work.**
- `dayone_healthy.prgcap`, `f9grabs/dayone_v24.prgcap`, `bp2.prgtl` — Day One era artifacts
  (#1264/#1335 evidence chain).
- `clobberdump.log` — six pre-#1368 arrays around DB-base clobbers; retained as the raw evidence
  that the two 32-record interpolant outputs contained split stack/pointer residue.
- Route scripts: `caphall*.sh` (capture-armed), `hall1.sh`/`hallfix.sh` (plain 120-shot route).

## Diagnostics cheat-sheet (all env-gated, off by default)

- `PROSPER_DBBASETRACE=1` — logs cx writes to the DB Z/STENCIL base registers with fold path;
  dumps the full array on a within-array clobber (`[dbbase-clobber]`). Current-master validation
  should produce no such dump; a recurrence should reopen producer attribution, with another
  unwritten advertised output as one hypothesis.
- `PROSPER_SCISSORLOG=1` — empty-combine dumps; `[scissor] degenerate SCREEN pair recovered` and
  `[render_state] lone-zero DB … recovered` fire on the two landed arena recoveries.
- `PROSPER_DSLOG=1` / `PROSPER_DSBRIDGE_LOG=1` — persistent-DS call keys and sampled-bridge
  HIT/miss (how #1352 was found).
- `PROSPER_FS_TAP=DRAW:PC` on a v19+ capsule — visualize a light-FS intermediate offline; `DRAW`
  is the semantic `draw[ID]` printed by `gpu_replay --inspect-only`, not its compact `item=` offset
  (#1396 fixed the index-space confusion that made the tap silently mutate a different draw).
- `PROSPER_DRAW_STATS=1` + `PROSPER_GEOM_PROBE=DRAW` — the pair that found #1427: draw-stats flags
  `GEOMETRY-VANISH`, and the probe reads back post-transform clip positions
  (`unique-pos=1 (max-mult=19236) -> COLLAPSED` is the fetch-returns-a-constant signature).
- `PROSPER_MAX_BUFFER_UPLOAD_MB=N` (1..64) — lower the buffer-upload ceiling to reproduce a
  truncation collapse on the same build; `[buffer-truncated]` reports every short upload.
- `PROSPER_NO_SUBMIT_TEXTURE_DECODE_SCOPE=1` (#1691) — restore the pre-#1691 span-scoped lifetime of
  the decoded-texture identity map, so one binary A/Bs that change on a routed run.
  `PROSPER_RENDER_TIMING=1` then reports
  `[render-timing] decode_scope decodes=… same_span=… cross_span=… invalidated=… pinned=… scope=…`
  alongside the existing `texture_cache` line.
- `gpu_replay --recompile-raw` (#1416) — re-recompile every retained raw VS/FS with the CURRENT
  recompiler and replay: the ~5 s offline A/B that resolved family 1 against a ~8 min live route.
  Pair with `PROSPER_MIPLOG=1` (per-binding mip eligibility) and `PROSPER_BUFLOG=1` (per-binding
  buffer upload provenance) when a replay disagrees with expectations.
- **When a `--recompile-raw` A/B and a live route disagree, suspect neither first — check host
  load.** The 1080p long-boot routes on this shared box slide past their capture windows above
  load ≈4 and photograph an earlier scene (see #1417); a same-build A/B under identical load is
  the only honest discriminator.
- The #1352-era scissor three-stage traces are parked on branch
  `fix/issue-1335-regindirect-tags`.

## Ruled out (do-not-redo list)

Exonerated with evidence (do not re-chase without new contradictory evidence): present staleness,
capture transposition, texture/descriptor decode for the Day One hold (#1287/#1334/#1335 record);
mips/CBR/checkerboard, shadow acne, caster bias, and sampler filtering for the ring family
(#1287 comment trail, 2026-07-25). The ring family itself is CLOSED (#1411/#1399/#1401/#1404) —
do not reopen a light-accumulation-math hypothesis for it; if rings reappear, check the
interpolant-resolution path first. Likewise the "dark slab"/sloped-floor artifact is CLOSED
(#1427) — missing geometry from a truncated buffer upload, not a misplaced or mis-transformed
surface; if large geometry disappears again, run `PROSPER_DRAW_STATS` before any shading theory.

Also closed: the **27 draws that still vanish at clip in the entrance hall are not a defect**. Each
was individually probed and confirmed as a legitimate frustum cull, and the transparent passes
reproduced the same off-screen bounds as their opaque passes (#1435/#1443). Do not re-open a
vanished-draw triage for that room without a *new* symptom.

Snapshot-guard triage (#1433/#1697) ruled out three tempting repairs and one misleading instrument:

- The `blue-prince-hall` failure is not evidence of a renderer regression and must not be repaired
  by lowering its thresholds. The current evaluator rejects three complete retained healthy runs;
  all eight inspected images from the newest run are the will/phonograph intro, not the hall.
- Neither the old 345--388 second window nor guest present count identifies the intended phase.
  Retained intro and hall evidence overlap around present counts 2900--3300. Use a content-identified
  consecutive plateau across a re-profiled bounded route, not another absolute-time/present window.
- The stored reference really is the pre-#1429 collapsed hall: the exact guard luma/SSIM algorithm
  scores the committed post-#1429 restored-hall image only 0.273488 against it (0.85 is required).
  Conversely, 45 retained intro frames score at most 0.072428 against the restored image. Do not use
  `issue-1287-hall-live-fixed.png` as a replacement seed: it predates the corrected screenshot
  persistence boundary and has transparent pixels with ambiguous provenance. Re-baseline only from
  fresh, composited, cross-run-verified hall evidence.
- Before #1433, `screenshot` premultiplied renderer RGB by render-target alpha for every content
  metric and persisted that alpha in its PNG, even though `prosper-app` blits the same target into a
  `VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR` swapchain. That made visible hall frames look mostly black to
  the evaluator: the 685--755 second hall tail measured 95,949--118,923 visible RGB colours but only
  hundreds to thousands after the erroneous alpha multiplication. Normalize alpha only at the
  rendered screenshot export boundary; raw guest scanout remains literal. Do not calibrate a guard
  against the old alpha-premultiplied numbers.

The repaired guard was then verified from two independent fresh-save runs. All 16 retained opaque
captures were individually inspected: both runs show the reflective checkered floor, round table
and bouquet, both busts, chandelier, lamps and console props, with no alpha holes or checkerboard
artifact. Their hall plateaus contain 8/11 consecutive matches and 7/10 local pixel changes;
opposite-run references validate 11 and 8 matches respectively. Fresh hall frames score only
0.271575--0.356440 against the obsolete collapsed reference, while the two retained intro
negatives score at most 0.082887 against the fresh references (0.85 is required).
