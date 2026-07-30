# Dragon Quest VII Reimagined (`PPSA17942`) — title-screen status

**Status as of 2026-07-30.** The game boots, composites its studio splashes correctly, and reaches a
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

## Do not restart

- The "mostly white" title composition (#1373) — fixed by #1411; the current symptom is black.
- Movie/USM playback as the black-screen cause: the frame is a fully rendered UE4 scene, not a video
  surface, and no AvPlayer activity accompanies the transition.
- The `no-effect` draws in the captured frame: all 22 have `cwm=0` with no depth or stencil write.
