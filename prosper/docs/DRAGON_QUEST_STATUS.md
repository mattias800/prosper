# Dragon Quest VII Reimagined (`PPSA17942`) — title-screen status

**Status as of 2026-07-31.** The game boots, composites its studio splashes correctly, and reaches a
state where it renders a complete UE4 frame (base pass, shadow atlas, lighting, bloom pyramid, tonemap)
every submit — but the presented frame is **pure black**. Bring-up ladder rung: **1 (real graphics:
splashes)**. Rung 2 (title screen rendered) is **not** reached.

The oracle screenshot is attached to #1373. The remaining defects are tracked by #1486; the recompiler
gap this investigation found is #1483.

> Naming note: this dump is labelled `DOLL` in the older docs (`DOLL_LOADING_PROGRESSION.md`,
> `DOLL_POSTPROCESS_HANDOFF.md`) — it is the same title. #1373's specific root cause (Gen5 virtual
> interpolant registers) was fixed by #1411; the title composition was not, so do not treat #1373 as
> "the title works".

## Reproduction recipe

Headless, native Vulkan, no diagnostic substitution. Run from a worktree build:

```bash
distrobox enter ps5ys -- bash -lc "cd <WORKTREE> && \
  env PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_GUEST_ARGS= TMPDIR=<WORKTREE>/build/tmpdir \
  ./build/screenshot <DUMP_ROOT>/PPSA17942-app0 \
  --seconds 9 --count 30 --out DIR --timeout 900"
```

Note `PROSPER_GUEST_ARGS=` (empty) and `PROSPER_NULL_PAGE=1` — the UE4 recipe. The Unity/Messenger
default (`-force-gfx-direct`) is wrong for this title. `tools/screenshot` defaults it, so pass the empty
value explicitly.

Observed progression (distinct-colour count of a 160×90 luminance thumbnail per frame; timings are from
a ~5.8 submit/s run and move ±20 s between runs):

| t | content | distinct colours |
|---|---------|------------------|
| 0–40 s | early boot / publisher frames | 194–244 |
| 40–80 s | Square Enix / ARMOR PROJECT frames | 303 |
| ~85 s | **Unreal Engine splash** (correct) | 223 |
| 125–145 s | **CRIWARE splash** (correct) | 377 |
| ~155 s | transition | 147–171 |
| 162 s → end (350 s+ observed) | **pure black** | **1** |

The workload does not change across that boundary: the guest submits ~88–96 draws and ~28 dispatches
per submit continuously from t≈5 s to the end of the run. The black frames are *not* an idle or stalled
guest.

## What is actually happening in a black frame

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

| operation | mean | distinct colours |
|-----------|------|------------------|
| 95–117 | 92–120 | 2740–3554 |
| 118–122 | **247** | 1322–1354 |
| **123 (draw 95)** | **0** | **1** |

So the frame carries real content all the way to the last draw, and **the final Slate quad (draw 95)
blacks the whole screen**.

### Defect 1 — the final Slate quad samples an empty renderer-owned surface (#1486)

Draws 94 and 95 are the same shader pair (`vs=73a37ede4c0c6c82 fs=71e10841003bafd3`), a 4-vertex
full-screen quad with `blend=1 color=6,7 alpha=1,7` (VK `SRC_ALPHA`/`ONE_MINUS_SRC_ALPHA`). They differ
only in the sampled texture:

- draw 94: `addr=…12450000 1x1 dcc=0` — a 1×1 white texture (Slate's solid-colour stand-in). Harmless.
- draw 95: `addr=…98330000 1920x1080 fmt=4 tile=27 dcc=1 nz=0 meta-unique=1 meta-first=0x40`
  — **all-zero guest surface bytes with uniform DCC metadata `0x40`**, tagged `temporal-RTT-seed`.

`0x40` is PAL's `ClearColor0001`, i.e. clear-to-`(0,0,0,1)`; `gfx10_dcc_fast_clear_rgba8` decodes it
correctly to **opaque** black, which then wipes the frame through the alpha blend. In the live run this
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

Next step: capture the producer chain as an ordered bundle
(`PROSPER_GPU_TIMELINE_CAPTURE_BUNDLE` + `PROSPER_GPU_TIMELINE_CAPTURE_DEPTH`) so the 1920×1080 producer
submit and the one-off 4K pass can both be replayed offline, then decide between (1) and (2) with the
producer's own draw steps. Do **not** "fix" this by suppressing the DCC fast clear: `0x40` decodes
correctly, and the metadata is only stale because prosper never writes DCC keys back.

### Defect 2 — the 4K composite underneath is massively over-exposed (#1486)

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

`(0 res)` means the shader-resource table for that dispatch is **empty** — a user-data/EUD descriptor
resolution gap (#282), not a missing opcode. A skipped exposure/grading producer is the standard cause
of exactly this symptom (see the Messenger LUT precedent in CLAUDE.md), so resolve this dispatch's
descriptors before looking anywhere else for the exposure error.

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

- **The black frame is a faithful presentation, not a renderer loss.** `PROSPER_DUMP_PERSISTENT=1` over a
  full route reports `scanout=HIT` on *every* present — the cache entry is found, extent and format match
  — and `rgb_nonblack=0` throughout the black window. Cross-checked against the passes: the scanout pass
  in that window reports `px_nonzero=8294400`, exactly `3840*2160`, i.e. one non-zero byte per pixel =
  **alpha only**. The presented PNGs measure `max=0` at full resolution. The guest's own composite is
  producing black RGB and prosper presents it correctly. **Search inside the composite, not in
  presentation.**
- **The tonemap source has real content.** The pass writing draw 92's source reports
  `px_nonzero=11321053` of 33,177,600, 200-300 times per run. Draw 92 replaces (`blend=0`) with it.
- **Draw 95's source has no writer at all.** `PROSPER_PROVENANCE_DIM=3840x2160` reports, for every
  address in the rotating family it samples:
  `no recorded color/compute/DMA/WRITE_DATA overlap for [0x…,+0x7e9000)`. Nothing prosper tracks fills
  them, and they miss the RTT cache, so they decode from guest memory. Note `0x7e9000` = 8,294,400 =
  exactly `3840*2160`, i.e. one byte per pixel, which does not match a 4-byte view of that extent and is
  worth checking on its own.

## Eliminated — do not re-run these

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

## Open question for the next investigator

Why does draw 92 — a full-viewport replace (`blend=0`, `mask=0xf`) from a source that demonstrably holds
11.3M non-zero bytes — leave the scanout with alpha only? Everything downstream (draws 94, 95) is then
irrelevant, because the pass as a whole never carries colour.

A secondary, separable question: what actually produces the `(u, v, u)` ramp seen when the #1510 fix is
applied? Measured as `R = B = x/width`, `G = y/height`. It is **not** the diagnostic clear (that is a flat
`(0,0,1,1)` blue, `render_runner.h`), not `PROSPER_TESTTEX` (`(u, v, checker)`, and the var was unset) and
not descriptor poison mode (also unset). Bisecting it offline with
`gpu_replay --draw-steps --draw-steps-every 1` on a captured frame is the cheapest way to identify it.

## Blocker on the tooling you will want to use

**#1505 blocks capture and replay on master.** `live_gpu_targets` is disabled by every capture/diagnostic
switch (`live_renderer.cpp`, the `PROSPER_GPU_CAPTURE` / `PROSPER_GPU_TIMELINE_CAPTURE` /
`PROSPER_RTTLOG` / `PROSPER_DUMP_DRAWSTEPS` / `PROSPER_RESOURCE_HASH_DIM` / `PROSPER_TARGET_STEP_HASH_DIM`
list), and the CPU RTT path it falls back to reads out of bounds and SIGSEGVs a few seconds into
rendering. Plain runs are unaffected and hide it.

Add **`PROSPER_NO_RTT_SNAPSHOT_BORROW=1`** to every capture or diagnostic run. That is sufficient for
`PROSPER_RTTLOG` and the timeline capture, but **not** for `PROSPER_TARGET_STEP_HASH_DIM`, which still
faults — there is a second out-of-bounds path on that side.

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

- The "mostly white" title composition (#1373) — fixed by #1411; the current symptom is black.
- Everything in the **Eliminated** section above.
- The two withdrawn conclusions in **Superseded analysis** — in particular, do not re-derive "draw 95
  blacks the frame"; its shader has been read and it is a no-op with the texels it receives.
