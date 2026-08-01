# Blue Prince (PPSA25009, Unity/IL2CPP) — status & investigation map

Last revised: 2026-07-26 (materials fixed and residual vanished draws triaged; see below). Prior history:
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
The `blue-prince-title` snapshot route guards the title screen.

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
