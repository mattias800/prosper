# Dragon Quest VII Reimagined (`PPSA17942`) — title/name-entry status

**Status as of 2026-07-31.** Current master reaches the localized, animated title screen at native
3840x2160 from a genuinely isolated save. The validated route sends seven Cross pulses at 3, 6, 9,
12, 15, 18, and 30 seconds; the title first appeared at about 34 seconds and remained animated through
the 40-second capture. Bring-up ladder rung: **2 (title screen rendered)**. Gameplay has not been
validated.

An exact-current-master Cross-only continuation with both save roots isolated now independently reaches
the new-save name keyboard without the historical MallocBinned3 failure. After the validated title route,
Cross at 35 seconds left the title, Cross at 55 seconds entered and highlighted `1: Unused`, and one
deliberately delayed Cross at 200 seconds confirmed that slot. The direct frontend reached the keyboard
by 204 seconds; the checked-in unmodified 3840x2160 frame at 222.3 seconds clearly shows the keyboard with
`A` highlighted and twelve empty name positions. No `GameSaveData*.dat` artifact existed when the run was
stopped; only the system and language files were present, so save creation and gameplay remain unproven.
The post-confirmation cadence was roughly 5.5 rendered FPS in this shared-GPU run. Adjacent frames still
intermittently wash white or blue, so this is a progression/control milestone rather than visual-correctness
evidence.

The earlier exact-master replay corrected #1553's temporal/flicker interpretation: Cross at 55 seconds
had already entered and highlighted `1: Unused` with its normal “Which slot would you like to use?” prompt. Circle at 140 seconds then
canceled to the adventure-log list, and Circle at 270 seconds canceled again to the title. Start/Options
at 330 seconds and Circle at 350 seconds did not advance the title. Cross is confirm and Circle is cancel
in this flow; #1553 incorrectly attributed the already-visible post-Cross prompt to the later Circle
press. The Cross-only continuation validates the next confirmation through name entry, but the
name-entry screen is not a gameplay milestone.

The same build can remain black after the startup sequence when run with **no input**. That is authored
Slate state, not a lost final composite: current-master operation-prefix replay shows a coherent
sky/ocean/title scene after draw 92, then draw 94 deliberately covers it with opaque black. Routed input
changes the foreground lifecycle and reveals the title. The remaining visual/performance work is tracked
by #1486; the earlier recompiler gap was fixed by #1483.

> Naming note: this dump is labelled `DOLL` in the older docs (`DOLL_LOADING_PROGRESSION.md`,
> `DOLL_POSTPROCESS_HANDOFF.md`) — it is the same title. #1373's specific root cause (Gen5 virtual
> interpolant registers) was fixed by #1411. The old mostly-white `MENUNAME_Title_001` screenshot is
> historical; the current capture renders the localized Dragon Quest VII logo, sky, and ocean.

## Reproduction recipe

Direct native Vulkan frontend capture, no diagnostic substitution. Run from `prosper/` with unique
save roots under `$HOME`. In the recipe, `<EVIDENCE_ROOT>` and `<FRESH_SAVE_ROOT>` are unique
directories created under `$HOME`; substitute their absolute paths before running the command so
screenshots and both save backends stay outside the worktree and `/tmp`.

```bash
PROSPER_GUEST_FS=1 \
PROSPER_NULL_PAGE=1 \
PROSPER_GUEST_ARGS= \
PROSPER_SAVE0=<FRESH_SAVE_ROOT>/save0 \
PROSPER_SAVEDATA_DIR=<FRESH_SAVE_ROOT>/savedata \
PROSPER_PAD_SCRIPT=@scripts/dragon-quest-vii/reach-title-screen.pad \
PROSPER_PAD_SCRIPT_LOG=1 \
./build-linux/screenshot <DUMP_ROOT>/PPSA17942-app0 \
  --seconds 1 --count 40 --out <EVIDENCE_ROOT>/shots --timeout 300 --require-composited-frame
```

Note `PROSPER_GUEST_ARGS=` (empty) and `PROSPER_NULL_PAGE=1` — the UE4 recipe. The Unity/Messenger
default (`-force-gfx-direct`) is wrong for this title. `tools/screenshot` defaults it, so pass the empty
value explicitly. The reusable route and its input-delivery expectations are documented in
[`scripts/dragon-quest-vii/README.md`](../scripts/dragon-quest-vii/README.md).

The validated run observed all seven Cross/neutral edges. Performance varies with concurrent GPU work;
treat the current title cadence as a ballpark observation, not a benchmark.

### Routed current-master progression

| t | content |
|---|---------|
| 0–33 s | startup logos/movie transitions and post-logo black state, advanced by Cross |
| ~34 s | **localized Dragon Quest VII Reimagined title appears** |
| 34–40 s | animated logo, sky/ocean, birds and water |

### Validated Cross-only continuation to name entry

The title route remains the short, checked-in regression recipe. A separate direct frontend run kept
the title's seven Cross pulses, then sent Cross at 35, 55, and 200 seconds. The long final pause was
intentional: it allowed a clean frame to anchor the selected `1: Unused` prompt before one confirmation.

| t | content |
|---|---------|
| ~76–168 s | `1: Unused` highlighted; “Which slot would you like to use?” |
| 200.046 s | Cross delivered; neutral at 200.453 s |
| 204–252 s | player-name keyboard, intermittently mixed with white/blue washed samples |

The representative image is
[`assets/screenshots/dragon-quest-vii-name-entry.png`](../../assets/screenshots/dragon-quest-vii-name-entry.png),
an unmodified native Linux `tools/screenshot` PNG from 222.3 seconds. The run was stopped after establishing
this state; no character was entered and no gameplay claim is made.

Some one-second samples show a dark/purple background behind the stable logo while adjacent samples
show the expected sky and ocean. That may be an authored transition or a remaining rhythmic background
flicker; settle it with a high-cadence sequence before changing renderer behavior.

### Historical no-input progression

The earlier table below used **no scripted input**. It remains useful for reproducing the authored-black
state, but it is not the title-screen route. Distinct-colour counts are from a 160x90 luminance thumbnail;
timings are from a ~5.8 submit/s run and move ±20 s between runs:

| t | content | distinct colours |
|---|---------|------------------|
| 0–40 s | early boot / publisher frames | 194–244 |
| 40–80 s | Square Enix / ARMOR PROJECT frames | 303 |
| ~85 s | **Unreal Engine splash** (correct) | 223 |
| 125–145 s | **CRIWARE splash** (correct) | 377 |
| ~155 s | transition | 147–171 |
| 162 s → end (350 s+ observed) | **pure black** | **1** |

The workload does not change across that boundary: the guest submits ~88–96 draws and ~28 dispatches
per submit continuously from t≈5 s to the end of the run. The black frames are not an idle or stalled
guest, but they are also not evidence that the rendered scene is missing.

## Historical no-input black-frame investigation

This section preserves the investigation that localized the late no-input composite. Its original
draw-95 conclusion was wrong and is corrected in **What is actually established** below.

Capture one late submit offline and bisect it — this is far faster than re-routing live:

```bash
# 1. record the index, then capture the submit in a second run
PROSPER_GPU_TIMELINE=dq.prgtl ...                                  # find a submit inside the black window
PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT=<N> PROSPER_GPU_TIMELINE_CAPTURE=black.prgcap \
  PROSPER_GPU_CAPTURE_MAX_MB=2048 ...
# 2. reproduce and bisect it deterministically (~30 s per iteration)
gpu_replay black.prgcap out.bmp                                    # reproduces the black frame exactly
gpu_replay --inspect-only black.prgcap
gpu_replay --draw-steps steps/s --draw-steps-every 1 black.prgcap steps.bmp
```

The frame is a complete UE4 pipeline: draws 0–23 depth prepass, 24–35 the 2048×2048 shadow atlas,
39–70 lighting at 3840×2160, 71–89 the bloom down/up pyramid, 90–92 tonemap into the scanout buffer,
94–95 Slate UI quads. 22 of the 96 semantic draws are reported `reason=no-effect`; every one of those
has `cwm=0 depth=1/0/6 stencil=0`, so they are genuinely effect-free occlusion/predication draws and
are **not** the defect.

Per-operation bisect of the captured frame:

Current-master prefix replay corrected the old interpretation:

| operation | semantic draw | result |
|-----------|---------------|--------|
| 120 | 92 | coherent sky/ocean/title scene |
| 121 | 93 | unchanged (`reason=no-effect`) |
| 122 | 94 | first exact opaque-black output |
| 123 | 95 | remains black; empty sample makes this draw a no-op |

So the frame carries real content through draw 92, and **draw 94 is the first black output**.

### Original defect-1 hypothesis — withdrawn

Draws 94 and 95 are the same shader pair (`vs=73a37ede4c0c6c82 fs=71e10841003bafd3`), a 4-vertex
full-screen quad with `blend=1 color=6,7 alpha=1,7` (VK `SRC_ALPHA`/`ONE_MINUS_SRC_ALPHA`). The original
analysis noticed the sampled textures but missed the decisive per-vertex colors:

- draw 94: `addr=…12450000 1x1 dcc=0` — a 1×1 white texture and packed vertex color
  `0xFF000000` (opaque black).
- draw 95: `addr=…98330000 1920x1080 fmt=4 tile=27 dcc=1 nz=0 meta-unique=1 meta-first=0x40`
  — **all-zero guest surface bytes with uniform DCC metadata `0x40`**, tagged `temporal-RTT-seed`,
  with packed vertex color `0xFFFFFFFF` (white).

`0x40` is PAL's `ClearColor0001`, i.e. clear-to-`(0,0,0,1)`; `gfx10_dcc_fast_clear_rgba8` decodes it
correctly to **opaque** black. The original analysis incorrectly attributed the final output to that
sample. In the live run this
is visible as `[render] DCC fast-clear addr=… code=0x40`. Offline the RTT cache hits instead
(`PROSPER_RTTLOG=1` → `sample tex addr=…98330000 1920x1080 -> HIT`) and the injected pixels are equally
empty: dumping the seed with `--dump-rtt-seed` gives **13 non-zero halves out of 12.4 M**.

`PROSPER_RTTLOG=1` with `PROSPER_RTTLOG_MIN_SUBMIT`/`_MAX_SUBMIT` around the transition explains why.
The engine ping-pongs its 4K HDR scene target between two addresses, and both carry real content:

```
47  [rtt] pass target=0x308cfc0000 extent=3840x2160 (5 draws) px_nonzero=32867450
47  [rtt] pass target=0x30769d0000 extent=3840x2160 (5 draws) px_nonzero=32867450
 1  [rtt] pass target=0x3098330000 extent=3840x2160 (5 draws) px_nonzero=8294400   <-- alpha only
```

`8294400 == 3840*2160`: exactly one non-zero byte per pixel, i.e. the RGBA16F alpha half only. So
prosper rendered the same five-draw pass into that address **once** and produced no colour at all,
and the timeline shows the guest had previously rendered a real 24-draw **1920×1080** pass into the same
base (`producer … submit=921 draw=23 target=…98330000/1920x1080 writes=24`). Two things are wrong here
and either could be the fix:

1. `g_rtt` (`frontends/shared/live_renderer.cpp`) is keyed by **base address alone**, so the later
   3840×2160 RGBA16F pass replaced the earlier 1920×1080 surface. A sample of the still-live 1920×1080
   view then gets a foreign extent (nearest-rescaled) instead of its own surface.
2. The one-off 4K pass produced only alpha where the identical pass at the two ping-pong addresses
   produces 32.8 M non-zero bytes, so that pass's own inputs were probably missing.

The historical proposed next step was to capture the producer chain as an ordered bundle
(`PROSPER_GPU_TIMELINE_CAPTURE_BUNDLE` + `PROSPER_GPU_TIMELINE_CAPTURE_DEPTH`) so the 1920×1080 producer
submit and the one-off 4K pass can both be replayed offline, then decide between (1) and (2) with the
producer's own draw steps. Current evidence made that unnecessary for title visibility. Do **not**
"fix" this by suppressing the DCC fast clear or authored draw 94.

### Original defect-2 hypothesis — withdrawn

Operations 118–122 are the scanout buffer (`…9fc0000000`) after the tonemap and the first Slate quad,
and they reach mean 247/255 with the title scene's structure clearly visible (sky gradient, sun disc,
horizon, ocean bands) but blown out.

How much this matters depends on defect 1: draw 95 covers the entire viewport, so if its texture is the
game's real full-screen layer then the over-exposed buffer underneath is never visible and this is not a
separate defect. Settle defect 1 first, then re-measure. Either way one compute program is still skipped
at the title, and it is not one of this frame's 28 dispatches (all 28 are realized) — it belongs to
another submit:

```
[mubuf-unresolved] pc=3 srsrc=s0 srt_tag=NONE 0x0 key_res=null (0 res)
[recompile-reject]  pc=3 words=e0102000,80000100 fmt=12 op=0x4
[compute] skip unsupported program 0x3013e80000            # run-local address
```

The raw shader later proved to be an exact four-instruction, one-dimensional zero-clear kernel, not an
exposure or grading pass. Its decoded output range does not overlap draw 95's foreground surface, and
the earlier title-screen milestone was reached while the dispatch was still skipped. It is not the
title-visibility blocker.

## The flashing white screen and the UI noise block are one defect: `CB_COLOR_CONTROL.MODE=2` (#1588)

**Read this before any further work on this title's composition.** It supersedes the "final Slate quad"
line of investigation below, which was chasing an ordinary draw that is not ordinary.

The project owner played through to the System Settings onboarding screens and reported the screen
flashing white with roughly one frame in ten "decent", plus a block of colour noise where the dialogue
box belongs. Two F9 grabs from that session (3840x2160 composite submits, 97 and 98 draws) show the same
structure, and the cause is visible in capture metadata without replaying anything:

Each submit contains **exactly two `mode=2` draws** — draw-scoped census `mode=0` x7/x8, `mode=1` x88,
**`mode=2` x2**, summing to the 97/98 draws — and they are the only draws covering the two defect regions:

| | scissor | topo | cwm | blend | position in submit |
|---|---|---|---|---|---|
| box | `[700,1560)-[3140,1988)` | 4 (`raw=3`) | f | 0 | mid-frame |
| screen | `[0,0)-[3840,2160)` | 4 (`raw=3`) | f | 0 | **last operation** |

- the white screenshot is 3840x2160 with **one distinct colour**, `(255,255,255)`, at 100% of pixels;
- the noise block's measured bounding box is x 700..3139, y 1560..1987 — **exact on all four edges**
  against the box draw's scissor, and the two grabs are from the same run two minutes apart.

`MODE=2` is `CB_ELIMINATE_FAST_CLEAR`, a colour-block metadata operation. `render_state.cpp` models
DISABLE(0), NORMAL(1), RESOLVE(3) and DCC_DECOMPRESS(6) and lets every other mode **fall through to an
ordinary draw** after a once-per-mode warning (`[gpu] resolve_pipeline_state: unsupported
CB_COLOR_CONTROL.MODE=2 -> ordinary draw fallback`, present in the session log). The only downstream
consumer of `cb_color_mode` is a diagnostic print, so nothing rescues it later.

The draws carry hardware's decompress signature, which is why the fall-through is destructive:

- both bind the **same 486-byte vertex shader that no other draw in the frame uses**, with no vertex
  resources at all (`VS: none`). Disassembled it is the procedural rect: `id & 1`, `id >> 1`,
  `v_cvt_f32_u32`, `fma(2.0, v, -1.0)`, exporting the four clip-space corners;
- both are 3-vertex RectLists (`topo=4 raw=3 indices=0`);
- both **inherit the pixel shader of the draw immediately before them** — correct on hardware, where the
  colour block performs the expansion and the bound PS is irrelevant.

**Why white and not black.** The inherited full-screen shader clamps its sample with
`v_med3_f32(s19, x, 0.25)` where `s19 = 1.0`, scales by `s16 = 1.0`, then linear-to-sRGB encodes. Its
output **floor is sRGB(0.25) ~ 0.54 and it saturates at 1.0** — it can only produce a bright field. Note
that the "alpha is `s18` x clamp = 0, so the draw is a no-op" reasoning recorded further down this
document does **not** transfer: that applies under `SRC_ALPHA` blending, and these draws are `blend=0`.
The two `mode=2` draws also bind different textures, which fits the two appearances: the box one samples a
2048x2048 single-channel R8 glyph atlas (`swz=0004`), the full-screen one a 48x36 texture.

The eliminate pass is emitted per fast-cleared surface, so which regions get overwritten varies frame to
frame — and the title double-buffers its scanout. In the white grab the two buffers' pre-frame RTT seeds
are one correct System Settings screen (7,410 distinct colours) and one **single-colour pure white** plane,
which is the buffer that submit renders into. That is the flashing.

`tests/test_render_state.cpp` currently **asserts the fall-through** for MODE=2 and MODE=5 ("unmodeled CB
modes log once per distinct value while retaining fallback behavior"). A correct fix must update that
block, and it is the natural home for the regression — state this in the PR, because a test that pins the
buggy behaviour makes a correct fix look like a regression to anyone who does not read it.

## Superseded analysis — read this before trusting the sections above

The two "defect" sections above were written on 2026-07-30 from a capsule taken on a **pre-#1483**
build. Measurement has since contradicted parts of them. Keep them for the reproduction recipe and the
diagnostics inventory; do not treat their conclusions as current.

**Withdrawn: "the final Slate quad (draw 95) blacks the screen."** Its fragment shader was dumped
(`PROSPER_SHADER_DUMP_SUCCESS`, raw hash `71e10841003bafd3`, 440 bytes, 76 instructions) and read:

```
pc=0024  image_sample  dmask=0xf -> v0          # samples RGBA
pc=0101  v_med3_f32    v3, -2.0, v0, 1.0        # clamp the sampled value
pc=0104  v_mul_f32     v0, s18, v3              # times a uniform scalar
pc=0105  v_cvt_pkrtz_f16_f32 v0, v2, v0         # alpha = high half of the COMPR export
```

Exported alpha is `s18 * clamp(sampled)`. The surface it samples decodes to all zeros, so
`med3(-2.0, 0, 1.0) == 0`, alpha is 0, and under its `SRC_ALPHA`/`ONE_MINUS_SRC_ALPHA` blend the draw is
a **no-op**. It cannot be blacking the frame.

**Withdrawn: "the 4K composite underneath is over-exposed."** Same provenance, not re-measured on a
current build.

## What is actually established (measured, current)

- **The title screen is reached on current master.** A direct frontend run with unique roots for both
  `PROSPER_SAVE0` and `PROSPER_SAVEDATA_DIR` and
  `reach-title-screen.pad` produced the localized Dragon Quest VII logo over animated sky/ocean from
  about 34 through 40 seconds at native 3840x2160. All seven Cross/neutral transitions were observed by the
  guest. The representative repository screenshot is an unmodified native frontend capture of the same
  validated title state; its earlier capture did not isolate both save roots.
- **Draw 92 is healthy.** Mixed-operation prefix replay through operation 120 produces the coherent
  scene (sky, ocean, islands and ships). A fragment tap at draw 92 also preserves that structured
  content. The old statement that draw 92 leaves alpha only is false.
- **Draw 94 is authored opaque black.** It samples a valid 1x1 white texture, but all four exact
  40-byte Slate vertex records carry packed color `0xFF000000` = `(0,0,0,1)`. The vertex and fragment
  shaders decode and multiply that value normally. Suppressing this draw would discard guest-authored
  UI state.
- **Draw 95 is the foreground layer.** In the no-input capture its sample is empty and its computed
  alpha is zero, so it is a no-op and draw 94 remains visible. In an older visible capture the same
  draw/state samples a non-empty 3840x2160 foreground surface. Routed input changes this lifecycle and
  exposes the title.
- **The remaining zero-clear compute is separate.** Its live direct descriptor decodes to a bounded
  strided buffer range that does not overlap draw 95's foreground texture. Historical title evidence
  also reached the title with this dispatch skipped, so it is not the route blocker.

## Ruled out — eliminated, do not re-run these

- **"The failing draws run with the previous pipeline's *user data*."** This title's own
  `[dynfail]`/`[drawpkt]` evidence founded #305 on that mechanism, and the **user-data half is now
  falsified**: measured on the louder Nikoderiko reproduction, the block is written by the
  *immediately preceding* bind, a handful of packets before the draw. **The PGM half still holds and
  is not ruled out** — every sampled failing draw is the first bind-or-draw event of its own `q3`
  fold and does inherit the previous `q1` submit's program; that inheritance is normal on a shared
  ring, and the defect is that the inherited state is wrong. Also falsified: the
  stale-shader-registry, missing/mis-ordered-bind, user-data-tail-alignment and TYPE-0-data-packet
  (#140) candidates. The full falsification list with its numbers lives in
  [`RESOURCE_BINDING.md`](RESOURCE_BINDING.md) § `Ruled out`; #305 was retitled on 2026-08-01 so its
  title no longer asserts the dead mechanism. The observable condition that *does* hold is that the
  programmed user-data block is **larger** than the bound pipeline's `USER_SGPR` window.
- **The depth-only-pass RTT clobber is not the cause.** It is a real defect (#1510) and it is real for
  other titles too, but fixing it does not fix this. Three variants were built and measured; all three
  made offline replay of a captured title frame produce a `(u, v, u)` colour ramp instead of the previous
  black, i.e. worse. PR #1513 was closed for that reason.
- **Not a colour-disabled scanout pass**: zero scanout passes are affected by that guard.
- **Not a missing or mismatched scanout cache entry**: `scanout=HIT` on every present.
- **Not an RTT-cache bypass of the tonemap source**: the `RTT PATH SKIPPED` diagnostic added by this
  change shows only shadow-atlas mip-tail levels and one 32x32 volume taking the guest-decode path.
- **Not movie/USM playback**: the frame is a fully rendered UE4 scene and no AvPlayer activity
  accompanies the transition.
- **Not the `no-effect` draws**: all 22 have `cwm=0` with no depth or stencil write.
- **The `[dynvb]` guessed vertex format is not the cause of the flashing or the UI noise block.** prosper
  does substitute a format it could not decode on this title (`[dynvb] PS code=… has unknown V# format
  0x0; using Float32x4`, and at one fetch it derives a 1.3 GB range from an all-zero V#), which is a real
  charter violation and belongs to its own issue. But it lands on a scene draw in the 4K HDR pass
  (`indices=9360`, matching the log's `draw_vertices=9360`), and that pass's target seeds to a coherent
  sunset sky/ocean image in the same capture. The two composition defects are `mode=2` draws whose only
  resources are one constant buffer and one texture each. Same for the `[buffer-truncated]` non-pointer
  descriptors (`addr=808080808080 declared=4294967295`, i.e. `0xFFFFFFFF`): real, tracked under the #305
  family, not these defects.
- **"The guest never asks for audio" is false.** A grep for `sceAudioOut`/`sceNgs2` returns nothing in a
  whole session, which reads as "audio was never initialised". This title drives **`sceAudioOut2`**, which
  emits neither name. `prosper-audio: opened port 18` is the tell: `hle_audio.cpp` sets `kMaxPorts = 16`
  and `kA2SinkPortBase = kMaxPorts + 1`, so host sink ports 17..20 are the AudioOut2 contexts. A real
  48 kHz stereo device opened. Whether the guest ever pushes PCM into it is unmeasured — #1692 carries
  the three-way `PROSPER_AUDIOLOG=1` instrument that settles it in one run.

## Open questions for the next investigator

1. Is the dark/purple background visible in some title samples an authored transition or a rhythmic
   renderer flicker? Capture a short high-cadence sequence at the already-routed title and correlate
   only adjacent frames; do not compare equal wall-clock indices from separate boots.
2. What input sequence completes name entry, creates the save, and advances into gameplay, and does the
   old MallocBinned3 content-load failure still reproduce on current master? Extend the checked-in route
   only after `PROSPER_PAD_SCRIPT_LOG=1` proves each transition.
3. The historical `(u, v, u)` ramp from the rejected #1510 experiment remains unexplained but is not
   present in the accepted current-master title capture. Do not revive that patch without a separate
   generic reproducer.

## Historical tooling blocker (resolved on current master)

The handoff build used for the original analysis hit **#1505**: `live_gpu_targets` was disabled by every capture/diagnostic
switch (`live_renderer.cpp`, the `PROSPER_GPU_CAPTURE` / `PROSPER_GPU_TIMELINE_CAPTURE` /
`PROSPER_RTTLOG` / `PROSPER_DUMP_DRAWSTEPS` / `PROSPER_RESOURCE_HASH_DIM` / `PROSPER_TARGET_STEP_HASH_DIM`
list), and the CPU RTT path it falls back to reads out of bounds and SIGSEGVs a few seconds into
rendering. Plain runs are unaffected and hide it.

Add **`PROSPER_NO_RTT_SNAPSHOT_BORROW=1`** to every capture or diagnostic run. That is sufficient for
`PROSPER_RTTLOG` and the timeline capture, but **not** for `PROSPER_TARGET_STEP_HASH_DIM`, which still
faults — there was a second out-of-bounds path on that side.

Do not carry that workaround forward blindly. The accepted current-master direct frontend capture and
the retained current-master capsule inspection both complete without it. If a new diagnostic still
faults, report the exact switch and current revision instead of treating #1505 as an active blanket
blocker.

## Methodology traps this investigation actually fell into

1. **Distinct-colour counts are not a content metric here.** prosper's `(u, v, u)` ramp scores **10,775
   distinct colours** on a 160x90 thumbnail; the genuine title sky/ocean frame scores **3,842**. A smooth
   gradient has a unique value in nearly every pixel, so "more colours" reads as "richer content". Open
   the image. Reserve the metrics for detecting *collapse* (1 colour, `max=0`), which is what the
   `tools/snapshot` guards are calibrated for.
2. **Frame-sequence before/after tables are confounded by run timing jitter.** Which game state each
   sampled frame lands on moves between runs, so a brighter frame at the same index is not evidence.
3. **Submit indices and guest addresses are both run-local.** Correlate only *within* one run. An
   analysis that matched a `PROSPER_RTTLOG` window from one run against addresses from another produced a
   confident and wrong conclusion.
4. **`bash -n` proves syntax, not semantics** — it cannot see an unquoted expansion that word-splits.
5. **`grep -c "^  color-state"` is not a draw census.** `--inspect-only` emits that line for graphics
   *dispatches* as well as draws, so a raw grep over one capsule returned 121 rows for 97 draws and a
   `mode=0` count of 31 where the draw-scoped figure is 7 — a total that does not even sum to the draw
   count, which is the tell. Scope the census to rows under a `draw[` header before quoting it. The
   `mode=2` figure survived only because it was independently confirmed by a lever that iterates draws.
   This is `GAME_COMPAT_ORCHESTRATION.md` trap #16 (one diagnostic label covering two packet kinds).
6. **A `.bmp` and a same-named `.prgbundle` are not necessarily one grab.** The frontend writes the bundle
   through a `.tmp` and the screenshot separately, so killing the app mid-grab leaves a `.bmp` whose
   same-named `.prgbundle` is from a **previous** grab. Two files handed to this investigation as "the same
   scene in two states" were 51 minutes and one boot apart; the screenshot's real partner was the abandoned
   `.tmp` written 0.7 s before it, which loads fine. Analysing the mismatched pair would have produced a
   large artefactual 1920x1080/1519-draw-vs-3840x2160/97-draw "difference" in exactly the comparison that
   was believed to be the highest-value evidence available. **`stat` the artifacts and confirm sub-second
   pairing before treating a screenshot as evidence about a bundle.** Recorded as trap #35 in
   `GAME_COMPAT_ORCHESTRATION.md`.

## What has been fixed

- **#1411** — Gen5 virtual interpolant registers (`0x10000000+n` → `SPI_PS_INPUT_CNTL_0..31`). Fixed
  #1373's "mostly white" title. Necessary, not sufficient.
- **#1483** — `vgpr_write_count()` counted `v_readlane_b32` (VOP3 `0x360`) as a VGPR write although it
  writes an SGPR. The title's grading kernel spills scalars into `v28` and executes
  `v_readlane_b32 s28, v28, 9`; the destination-SGPR/spill-VGPR number collision recorded a phantom
  clobber, so a later legitimate `v28` lane reload rejected as `invalidated-vgpr-lane-slot`, the CFG
  dispatcher bailed out, the straight-line fallback rejected at the kernel's first `s_cbranch_vccz`,
  and the dispatch was skipped. Before the fix two compute programs were skipped at the title; after
  it, one.
- **#1529** — kept live owner-backed RTT snapshots authoritative when DCC metadata made the guest
  decode cache eligible again. This removed the synthetic `(u, v, u)` coordinate-ramp regression from
  retained title-frame replay without special-casing Dragon Quest.

## Known-good diagnostics for this title

| what | how |
|------|-----|
| exact recompiler reject | `PROSPER_DBG=1` → `[recompile-reject]`, `[compute-cfg]`, `[graphics-cfg-reject]` |
| raw failing shader | `PROSPER_SHADER_DUMP=DIR` (**create DIR first** — the writer does not) + `shader_inspect` |
| which draw blacks the frame | offline capsule + `gpu_replay --draw-steps … --draw-steps-every 1` |
| renderer-owned target contents | `PROSPER_RTTLOG=1` with `PROSPER_RTTLOG_MIN_SUBMIT`/`_MAX_SUBMIT` |
| a target's cached bytes | `gpu_replay --dump-rtt-seed ADDR PATH` |
| why a sampled target took the guest-decode path | `PROSPER_RTTLOG=1` -> `RTT PATH SKIPPED (storage=/rtt_on=/volume=/mip_tail=)` |
| what the presented frame actually holds | `PROSPER_DUMP_PERSISTENT=1` -> `[persist] present: … scanout=HIT/MISS rgb_nonblack=N` |
| who writes a surface (colour, compute, DMA, WRITE_DATA) | `PROSPER_PROVENANCE_DIM=WxH` |
| a successful shader's raw RDNA2 + SPIR-V | `PROSPER_SHADER_DUMP_SUCCESS=DIR` (**create DIR first**) + `shader_inspect` |

## Do not restart

- The "mostly white" title composition (#1373) — fixed by #1411. The checked-in screenshot now shows
  the localized title rather than `MENUNAME_Title_001`.
- The premise that no-input black means the title scene failed to render. Use the checked-in route;
  draw 92 already contains the coherent scene and authored draw 94 covers it while that UI state is
  active.
- Everything in the **Ruled out** section above.
- The two withdrawn conclusions in **Superseded analysis** — in particular, do not re-derive "draw 95
  blacks the frame"; its shader has been read and it is a no-op with the texels it receives.
