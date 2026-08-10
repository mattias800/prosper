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

> **Diagnosed and NOT yet fixed. Blocked on #1706 — read that before attempting the fix.**
>
> The obvious fix is to stop the bound pixel shader's export reaching the target for `MODE=2`, and an
> offline A/B (#1695) confirms it restores the correct frame. It is **not safe to key on `MODE` alone**:
> #1706 established that prosper's decoded `CB_COLOR_CONTROL.MODE` is not per-draw-trustworthy. Astro Bot
> submit 6174 has four draws all reporting `mode=2`, of which **two are ordinary shaded draws** — one
> blended into two MRTs with a 3,107-dword VS and a vertex buffer — so suppressing on the mode drops real
> geometry from another title. `gpu_execute.hpp` already works around the same latch for `MODE=6` by
> matching helper-program content. Fix the tracking (#1706), then this fix is correct as written.
>
> Landed meanwhile: `MODE=2` is now named in `pm4_registers.hpp`, and the once-per-mode warning is a
> counted report (powers of two, exact running count, `unmodeled_cb_color_mode_count()`), so per-title
> exposure is measurable instead of being inferred from a line that could not distinguish one occurrence
> from hundreds of thousands.
>
> **`gpu_replay` cannot demonstrate this fix.** Both `--bundle` and `.prgcap` go through
> `materialize_gpu_replay`, which binds the **stored** `ResolvedPipelineState` — the `d.ps = x.ps` line in
> its draw loop (`gpu_capture.cpp`, line 3405 as of `37768edc`; grep the assignment rather than the line
> number, it drifts) — so replay never re-runs `resolve_pipeline_state` and a change there is invisible to it.
> That is why #1695's A/B lever had to sit in `gpu_replay`'s `main()`. Verify in a live run or in
> `tests/test_pipeline_render.cpp`, not by replaying an artifact.

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
ordinary draw**. The only downstream consumer of `cb_color_mode` is a diagnostic print, so nothing
rescues it later.

> **The log line quoted here is historical.** At the time of this investigation the warning read
> `[gpu] resolve_pipeline_state: unsupported CB_COLOR_CONTROL.MODE=2 -> ordinary draw fallback` and
> was deduped once per mode value. **That string no longer exists in the tree** — grepping a fresh
> run's log for it finds nothing. The current report names the mode, says it is still executed as an
> ordinary colour draw, and carries a running count; the exact per-mode total is available from
> `unmodeled_cb_color_mode_count()`. The *behaviour* described above is unchanged.

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

`tests/test_render_state.cpp` used to assert `occurrence_count(..., "MODE=2 ") == 1` under the message
"unmodeled CB modes log once per distinct value while retaining fallback behavior". That asserted the
**log-dedupe mechanism**, never the draw behaviour, and its "retaining fallback behavior" clause was not
tested at all. The history settles the intent: #919 introduced the block over modes **2 and 3** as
stand-ins for "not implemented yet", and #1238 then implemented MODE=3, moved the placeholder to **5**,
and gave 3 its own behavioural assertion. So it was an accidental pin, not a contract — the same
migration is due for 2. The block now asserts the counted-report contract and deliberately does **not**
pin what these modes do to the draw, so the eventual fix will not read as a regression.

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

- **Audio is now audible (#1700 fixed the discard; the measurement below is the historical
  pre-fix state).** The 12-channel MAIN port is folded to the host bed and its context opens a
  sink. On the same `reach-title-screen.pad` route, ctx0 now reports
  `mixed=188 skip-fmt=0 sink=open port=17`, `BED LIFE: nonzero=12,636,416/14,041,088 peak=0.38978
  rms=0.03533`, and the captured stereo bed is 147 s of music-like content (55.8 % of its energy
  below 200 Hz, 38.6 % from 200 Hz–2 kHz, 5.4 % from 2–8 kHz, 0.2 % above; `audio_analyze.py`
  reports `CLEAN`, `dup-grains=0.0%`).
  **What the bed actually contains — measured, and it is not what a 12-channel count suggests.**
  `PROSPER_AUDIO_LAYOUT=1` over 19,962,368 frames: all of the content is in **ch0/ch1**
  (rms 3.1e-2 / 4.3e-2, peak 0.407, correlation +0.46 — a real decorrelated stereo pair). ch2 and
  ch4..ch7 hold only a ~1e-9 residue, and **ch3 and ch8..ch11 are exactly zero**. So the title
  writes a stereo mix into a 12-channel container and never writes ch3 or ch8..ch11 at all — the
  LFE and height positions under the assumed order, so this describes which INDICES are dead
  without resting on the mapping it is used to support. Within the residue, ch4/ch6 correlate +0.96 and ch5/ch7 +0.95 while ch4/ch5 is -0.04, so
  the surround tier pairs even=left / odd=right; ch2 correlates near-equally with both groups
  (centre-like). Reproduced identically on a second independent 150 s run.
  **The 8-channel port (ctx1/port2) remains exactly zero** — 0 non-zero of 55,175,168 samples —
  even though prosper forwards it. That is a separate open question (#1721), not this fix.
  **Rung 4 for audio: the project owner confirmed by ear that the music plays and sounds right.**
  That establishes real audio reaching the device at sane levels through the guest's own path. It
  establishes **nothing about the channel order** — with ten of the twelve channels empty, every
  mapping that routes ch0/ch1 to the two sides produces a bed that differs only by the ~1e-9 residue
  on ch2 and ch4..ch7 — about 150 dB below the content, and inaudible by any measure. Not
  "bit-identical": those channels are a residue, not exactly zero, and this document insists
  elsewhere that those are different findings. A left/right swap
  is inaudible without a reference in any case. Do not cite the listening test as layout evidence.
- **Historical pre-fix measurement**, with `PROSPER_AUDIO_FLOW=1` (see `AUDIO.md`) on a
  `reach-title-screen.pad` run of the direct SDL3 frontend. The title creates **two** AudioOut2 contexts, each with one MAIN
  (`type=0x0`) port, and they carry opposite content. All figures below are the never-reset `LIFE:`
  run totals from the **final report line of a single run**, not summed interval samples:

  | | ctx0 / port1 | ctx1 / port2 |
  |---|---|---|
  | `data_format` | `0xc00` -> 12ch f32 | `0x800` -> 8ch f32 |
  | pushes / s | 188 | 188 |
  | guest PCM read / s | 48,128 frames, 2,310,144 B | 48,128 frames, 1,540,096 B |
  | mixed to host | **no — discarded, `skip-fmt=188`** | yes, 188 grains/s to sink port 18 |
  | **`LIFE: nonzero`** | **1,782,048 / 18,456,576** | **0 / 15,007,744** |
  | **`LIFE: peak` / `rms`** | **0.38976 / 0.01033** | 0.00000 / 0.00000 |
  | `LIFE: nan` | 0 | 0 |
  | context sink | `never-opened`, `silent-paced=188` | `open`, `BED LIFE: nonzero=0/3,751,936` |

  So the audio the title actually renders is in the **12-channel port**, at healthy levels — and
  prosper's push mix loop rejects `channels > 8` and throws away all 2.31 MB/s of it, 188 times a
  second. Because port1 is ctx0's only port, `have_pcm` is never true for that context, so it
  **never opens a host sink at all**. The 8-channel port that prosper *does* mix and forward to the
  real device is the one that is exactly zero-filled — which is why the device opens, plays, and is
  silent. `nan=0` on both ports rules out a NaN/decode artifact. This was a defect in prosper, not
  upstream in the title, and #1700 fixed it.
- **A declared channel count is not a description of the content.** The `0xc00` decode and the
  `2,310,144 B/s` figure above are both *derived from the same channel count*, so neither
  corroborates it, and "12 channels" reads as "a 7.1.4 bed with height" when the measured bed is
  stereo with ten of its twelve channels carrying nothing audible. Measure the channels before designing a fold for them; see
  instrument-trap 43.
- **Read the `LIFE:` totals, not a single interval.** This finding was initially called the opposite
  ("the guest submits only silence") from one report line in which port1 showed `nonzero=0/577536`.
  Playback has gaps, so any one interval can land in one; in the same run the port's run total is
  1.78 M non-zero samples at peak 0.38976. Every port and context line now carries a never-reset
  total for exactly this reason. See instrument-trap 39.

## Windows pre-lift baseline (2026-08-10, master `a6043524`)

Recorded **before** the runtime-selected-descriptor lift (#2412) begins landing its layers, so that a
cross-title regression on this title is detectable afterwards. The lift changes device creation, reflection,
SPIR-V emission and the executor, so every title is in its blast radius even though DQ is not the title it
targets.

Six runs, `screenshot --seconds 60 --count 1`, `PROSPER_NULL_PAGE=1 PROSPER_GUEST_ARGS= `
`PROSPER_NO_RTT_SNAPSHOT_BORROW=1 PROSPER_GFXLOG=1 PROSPER_DBG=1`, no input route.
Windows 11, RTX 4090 driver 32.0.16.1047, MinGW-w64 UCRT gcc 16.1.0, Vulkan SDK 1.4.350.0.

| run | frames | `[render] item` draws | wave64 shader lines | `recompile-reject` | composite crc |
| --- | --- | --- | --- | --- | --- |
| base1 | 862 | 3497 | 4 | 0 | `666f7b3f` |
| base2 | 769 | 3114 | 4 | 0 | `666f7b3f` |
| base5 | 189 | 1926 | 21 | 0 | `666f7b3f` |
| base3 | 0 | 0 | 0 | 0 | `08ed2210` |
| base4 | 0 | 0 | 0 | 0 | `08ed2210` |
| base6 | 0 | 0 | 0 | 0 | `08ed2210` |

### Assert these — they held on every run

- **`recompile-reject` count is 0.** Every run, rendered or stalled. This title has no recompiler rejects at
  all, which is what distinguishes it from GTA V's 951 (all `unresolved-operand`; run-local like every
  other count here — the same class measured 920–933 across the Linux lane's runs, so do not quote 951
  as a constant) and is why the descriptor
  lift is **not** expected to change DQ's composite. A non-zero count after the lift is a real regression.
- **`crc=666f7b3f` on every run that rendered at least one frame**, and `crc=08ed2210` on every run that
  rendered none — 3/3 each way, no overlap. So the crc **also diagnoses which mode a run was in**, which is
  useful because the two are otherwise easy to confuse.
- Boot reaches guest execution (`File root is /app0/`) on 6 of 6.

### Do NOT assert these — they are phase-dependent and will cry wolf

**Frames (189–862), draws (1926–3497), and wave64 shader lines (4–21) all vary by more than 2x between runs
of the same binary.** They are not noise around a value; they depend on how far the boot got inside a fixed
wall-clock window, and a 60 s window lands in different phases on different runs.

**Every figure in this table is Windows/NVIDIA (RTX 4090), and the wave64 column is a device property, not
a title property.** Whether a wave64 fragment shader is skipped is decided by a **seven-way disjunction over
adapter properties** (`render_runner.h:4487`–`:4496`) — subgroup-size control, min and max subgroup size,
`requiredSubgroupSizeStages`, `subgroupSupportedStages`, the subgroup feature set, and internal-GDS use
without fragment stores/atomics. On this project's Linux lane — AMD Radeon 8060S (RADV STRIX_HALO) — **all
seven terms are false**, so that adapter is expected to report **0** skips for the same title:
`subgroupSizeControl=true`, `minSubgroupSize=32`, `maxSubgroupSize=64`, `requiredSubgroupSizeStages` and
`subgroupSupportedStages` both include `FRAGMENT`, `subgroupSupportedOperations` includes
VOTE/ARITHMETIC/SHUFFLE, `fragmentStoresAndAtomics=true`. **That 0 is derived from device properties, not
measured over this route** — no Linux run of this title over this configuration has been taken.

**Surprise is possible in BOTH directions, so re-derive rather than compare.** An adapter that omits
`FRAGMENT` from `requiredSubgroupSizeStages` fires disjunct 4 for *every* wave64 fragment shader and will read
far **above** 4–21; an adapter like the Linux lane's reads 0. Neither is a regression. Run `vulkaninfo`
against the seven properties before concluding anything from this column — citing `maxSubgroupSize` alone is
not enough, because it licenses only "disjunct 3 cannot fire" and says nothing about the other six.

Frames and draws are host-speed-dependent for the same reason the spread above exists — a fixed wall-clock
window reaches a different point on a different machine — so compare against this table only from a
Windows/NVIDIA run, and re-derive it locally otherwise. Cross-platform, only the two assertions above
(`recompile-reject = 0` and the crc dichotomy) carry.

**Draws-per-frame is not an invariant either, and this was nearly recorded as one.** base1 and base2 agreed to
two decimal places (4.06, 4.05), which looked like a stable normalisation that would survive the ±12% spread
in the raw counts. base5 gives **10.19**. The agreement was coincidence — both runs happened to sample the
same phase. This is instrument trap 146 (a ratio over a moving sequence measures *when the window landed*),
committed while assembling a baseline whose purpose is to avoid exactly that.

### The wave64 column counts SHADERS, not skipped draws — do not divide it by the draw count

`[render] skip draw: fragment shader requires subgroup size 64` is emitted inside a dedupe guard keyed on
fragment-shader identity (`render_runner.h:4502`, `logged.insert(shader_key).second`), while the `continue`
that actually drops the draw sits **outside** it at `:4576`. So the line fires **once per distinct shader**
and the skip happens **per draw**: `grep -c` over that message counts shaders, and the number of draws lost
to wave64 is **not measured by any current diagnostic**.

This is recorded because the mistake is easy and was made here: a census on #2448 compared this count against
the `[render] item` draw total and reported *"22 skipped draws of 1,467, i.e. 1.5%"*, concluding that wave64
could not explain a black composite. **Numerator and denominator were different units.** The conclusion may
still be right, but that measurement does not support it, and the honest statement is that the skipped-draw
count is unknown.

Counting skipped draws needs a counter at the `continue`, not at the log line.

### The acceptance rule, which is load-bearing

**A run counts only if it reached guest execution AND rendered at least one frame.** `File root is /app0/`
alone is insufficient: base3/4/6 all satisfy it, render nothing, and return a *different crc*. Scored
naively, the baseline would read "crc may be either value" and a stalled run after the lift would present as a
composite change.

### Stall rate here is 50%, and that is an apparatus figure

3 of 6 runs rendered nothing. That is far worse than the **1 stall in 11** measured on 120 s runs **without**
`GFXLOG`/`DBG` (#2448) — those diagnostics are high-volume and slow the boot, so a 60 s window is much more
likely to close before rendering starts. **Do not quote 50% as a property of the title**; it is the rate for
*this* configuration. The comparable no-diagnostic figure is on #2448.

### How to use this after a lift stage lands

Re-run the same command, discard any run that fails the acceptance rule, and compare only the asserted
invariants. A change in frames, draws or wave64 skips is **not** evidence of anything on its own.

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
   `GAME_COMPAT_ORCHESTRATION.md`. **Fixed at the source by #1693 / #1694**: a grab claims both output names
   when it is armed, from one timestamp and at one collision suffix, so the two artifacts of one press
   always carry the **identical full stem, suffix included**. Pair on the whole stem, never on the
   timestamp — `…-210000-123.bmp` and `…-210000-123-2.prgbundle` are two different grabs despite an
   identical title and millisecond. The `stat` habit still applies to artifacts captured before that
   landed (`captures/frame_grab_001.*` is exactly that shape) and to anything else named by ordinal.
7. **`PROSPER_GFXLOG=1` PERTURBS this title's routed capture — the diagnostic changes what it observes.**
   Measured 2026-08-10 on current master (`08d42aea`), Linux, native 3840x2160, the documented
   `reach-title-screen.pad` route with fresh save roots. Identical command except for the variable:

   | run | frames 33-44 | verdict a naive reader takes |
   | --- | --- | --- |
   | documented recipe (no `GFXLOG`) | title held, mean luminance 173-220, 38 of 45 frames with content | correct |
   | `PROSPER_GFXLOG=1` added | **pure black**, mean 0.00, from frame 33 to 39 | "the title regressed on master" |

   `PROSPER_GFXLOG` emits roughly **160,000 lines** over this route (counted: 159,619 `[render] item`
   lines in one run) and slows it enough to shift a **time-dependent pad script** out of alignment with
   the sampled frames. The presses themselves are delivered on schedule — `[pad-script] elapsed=30.049
   frame=479 buttons=cross` — so `PROSPER_PAD_SCRIPT_LOG` *confirms the route ran* while the game is in a
   different state by the time each screenshot fires. Both halves look healthy; only the composite is
   wrong. **Every routed capture in this project is timing-dependent, so add no logging variable to a
   scripted route without re-establishing the baseline under the same variable.** This nearly produced a
   phantom title-screen regression report on a clean master.
8. **A no-input black frame here is the authored Slate state, and it will invert a cross-platform
   conclusion if taken as data.** Also 2026-08-10: a default 90 s launch is black with
   `pixel-distinct=1`, and it was one step away from being published as "Linux composites black with zero
   wave64 skips, therefore wave64 cannot explain the Windows black composite" — a *falsification* of the
   leading hypothesis (#2448) drawn entirely from a state this document and
   `scripts/dragon-quest-vii/README.md` both already describe as authored. The routed run says the
   opposite: the title renders correctly. **Read the route README before the run, not after it** — the
   warning is in the first paragraph, and the black frame is convincing enough that nothing downstream
   would have questioned it.

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
| whether audio is absent, silent, or dropped by us | `PROSPER_AUDIO_FLOW=1` -> `[audio-flow]` (see `AUDIO.md`); read `nonzero=N/M`, not peak/rms |
| what a multichannel MAIN bed actually carries per channel | `PROSPER_AUDIO_LAYOUT=1` -> `[audio-layout]` (rms/peak/nonzero%/spectral tilt/correlation); `PROSPER_AUDIO_LAYOUT_DUMP=PATH` for the raw capture |

## Do not restart

- The "mostly white" title composition (#1373) — fixed by #1411. The checked-in screenshot now shows
  the localized title rather than `MENUNAME_Title_001`.
- The premise that no-input black means the title scene failed to render. Use the checked-in route;
  draw 92 already contains the coherent scene and authored draw 94 covers it while that UI state is
  active.
- Everything in the **Ruled out** section above.
- The two withdrawn conclusions in **Superseded analysis** — in particular, do not re-derive "draw 95
  blacks the frame"; its shader has been read and it is a no-op with the texels it receives.
