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

## The content oracle for this title: which UE package the guest read

`GAME_COMPAT_ORCHESTRATION.md` records that no aggregate frame metric separates an animating menu
from gameplay, and that the Unity answer is the `Media/levelNN` scene sequence. This is an Unreal
title, so that oracle does not apply. The Unreal equivalent is the **IoStore package stream**,
decoded from a `PROSPER_FILELOG=1` run by `tools/re/iostore_index.py`:

```bash
python3 tools/re/iostore_index.py \
    <DUMP_ROOT>/PPSA17942-app0/doll/content/paks/pakchunk0-ps5.utoc --log <run>.log --maps
```

Why it works here, and why `pak_index.py` does not: `DefaultGame.ini` in this dump sets
`bUseIoStore=True`, so `pakchunk0-ps5.pak` holds **only** configs, fonts, localization and CRI audio
banks — 3,967 entries, **zero** `.umap` — while the 17.8 GB `pakchunk0-ps5.ucas` holds 103,274
packages including all 5,314 maps. The `.utoc` directory index is unencrypted
(`ContainerFlags = 0x08`, Indexed only), so every read offset resolves to a package path.

The map namespace is semantic, which is what makes it self-checking:

| package prefix | count | what it is |
| --- | --- | --- |
| `Map/Product/Title/` | 3 | the title screen — `Title_PL.umap` is the guest's own `GameDefaultMap` in `DefaultEngine.ini` |
| `Map/Product/World/Field/` | 4,729 | towns (`TNNN`), castles (`CNNN`) and dungeons (`DNNN`), each `_M_` (modern) or `_P_` (past) |
| `Map/Product/World/Battle/` | 474 | battle arenas |
| `Map/Product/World/*_PL.umap` | 4 | the per-era persistent levels, e.g. `World_M_PL.umap` |

So "still on the title" and "in a town" are different package prefixes, not different frame
statistics.

**The instrument was validated outside any run**, by computing a known package's physical `.ucas`
offset by hand from the container index and checking that the tool names it back — the discipline
CLAUDE.md requires, because a same-source control tests the discriminator and not the domain:

| package | logical offset | physical offset | resolves |
| --- | --- | --- | --- |
| `Map/Product/Title/Title_PL.umap` | `0x248ab0000` | `0x154ebf1e0` | yes, exact |
| `Map/Product/World/World_M_PL.umap` | `0x249060000` | `0x1550ee210` | yes, exact |
| `Map/Product/World/Field/F001/T001/T001_M_PL.umap` | `0x299be0000` | `0x18cd2ba30` | yes, exact |

Note the 4.1 GB gap between the two offset spaces at the startup map. A chunk's
`FIoOffsetAndLength` is a **logical** (uncompressed) offset; a read syscall carries a **physical**
offset into the packed `.ucas`. Resolving one against the other produces a confident wrong package
name, which is why the tool goes physical -> compression block -> logical -> chunk and its
registered self-test pins exactly that step.

### Residency is not activation — this title front-loads, and the naive form of the oracle fires on a menu

**Do not test "did the guest read a `Map/Product/World/**.umap`".** Measured 2026-08-20 over an
800-sample run: this title reads `Title_PL.umap` at 2 s and then a cluster of
`Map/Product/World/Field/F001/T001/Modern/T001_M_Env_*`, `T001_M_Gmk_GOut`, `T001_M_PartyTalk_GIns`
and `F001_M_Monster_Random_GOut` at **57-58 s** — while the player is still on the
adventure-log/slot-selection screen, roughly ten minutes before any world is entered. Two more
World maps (`D048_M_PL`, `C001_T_M_Npc_GIns`) are read at **0.0 s**, before the title map. A
membership test would have reported "gameplay" on a run that never left the menus, which is exactly
the failure the oracle exists to prevent.

**The form that discriminates is the phase profile, not the membership.** Bytes read per minute,
resolved to the top-level content directory, separate the two states by two orders of magnitude:

| window | Environment | Character | Map/Product | UserInterface |
| --- | --- | --- | --- | --- |
| 0-60 s (boot + front-load) | 133 MB | 118 MB | 11.6 MB | 331 MB |
| 360-420 s (first-run setup menus) | 0.25 MB | 5.1 MB | **0 MB** | 0 MB |
| 600-660 s (entering the world) | **285 MB** | **109 MB** | **9.8 MB** | 4.9 MB |

and the packages that appear *only* in the third window name the event: `T001_M_PL.umap` — the
persistent level, as opposed to the sub-levels touched at 58 s — together with
`World_P_Streaming.umap` and `SkitSystem/LevelSequence/CS_CP1_001_010/CS_CP1_001_010_GEvt.umap`,
the opening chapter's cutscene sequence.

So on a front-loading Unreal title the oracle answers **"which package the engine opened, and
when"**, and the gameplay claim rests on the *when* plus the volume, never on the set. On a title
that streams on demand the membership form is enough — that is what made it decisive on
`PPSA19244`. Check which kind of title you have before quoting it.

## 2026-08-20: the route reaches the opening chapter in Estard, and the game writes a save

**Two independent runs on a branch off `82baa409`**, Linux, AMD Radeon 8060S (RADV STRIX_HALO),
native 3840x2160 through `tools/screenshot`, UE4 recipe, native cadence, isolated `PROSPER_SAVE0`
and `PROSPER_SAVEDATA_DIR`, `PROSPER_FILELOG=1`. Run A 800/800 samples in 802 s
(`stop=request-satisfied guest=running status=ok`); run B 1400 samples with the route steered live
through `PROSPER_PAD_SCRIPT_RELOAD=1`. Both reach the same content state.

| what | run A | run B |
| --- | --- | --- |
| title screen | 34 s | 76 s |
| slot list, `1: Unused` | 63 s | 124 s |
| player-name keyboard | 84 s | 144 s |
| name accepted, System Settings 1/4 | 369 s | 299 s |
| "Adventure log successfully created." | 622 s | 428 s |
| `GameSaveData000.dat` written | 127,224 B | 127,224 B |
| `T001_M_PL.umap` + `CS_CP1_001_010_GEvt.umap` loaded | 650.7 s | 443.1 s |
| first 3D world frames | 662 s | 451 s |

**What is established.** The guest creates a real adventure log — `GameSaveData000.dat`, byte-count
identical across the two runs, alongside the `SystemSaveData999`/`LanguageSaveData998` that the
title screen alone produces — then loads Estard's persistent level and the chapter-1 level sequence
and renders the world: the coastal cliffs, the shrine interior with its standing stones, the harbour
with its moored boat and palms, a night sky with a correctly rendered moon and stars. The opening
chapter's script runs: named-character dialogue boxes ("Maribel"), and story lines including
*"Right, that's enough for one day. Time for me to head back to the castle..."* and *"There's more
to the world than this island... And we're going to prove it!"*

**What is NOT established: free player control.** Both runs stay inside the scripted opening —
cinematic letterbox bars are present on most world frames, and a long-window control probe
(30 s of full left-stick deflection against 30 s neutral, repeated, with confirms confined to the
neutral gaps) did not produce a clean separation, because by that point the composite had degraded
to mostly uniform frames and the measurement had nothing to measure. **So this is progress past
rung 2 without a demonstrated rung 3**; whoever picks this up next should aim the probe at the
first frames after the chapter script hands over, not at an arbitrary later window.

**Run B ended in a guest fault (#2778).** Its primary thread SIGSEGVs on a null address
(`rip=0x5c00048e0 addr=0x0`, `rbp == rsp`) at roughly **965 s**, about 530 s after the level load
and well into the chapter; presentation freezes on one frame from sample 945. Run A's clean
`guest=running` finish is **not** a negative arm — its 800-sample budget ran out at 802 s, before
the moment where run B died. A reproduction needs a run of at least ~1100 s.

**The composite is severely degraded throughout the world phase, and it gets worse as the chapter
runs.** Over run A's world window, 98 of 140 samples carry structured content, 15 are near-white and
0 near-black; over run B's later window (610-860) it is 99 structured, 19 near-white and **78
near-black**. Surfaces render as flat black silhouettes or as rainbow-checkerboard noise, water and
sand read as saturated orange, and whole frames flash uniform white or blue. That is #1486 and
#1588 in the world rather than in the menus; none of it blocks the progression above, and all of it
needs its own investigation.

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
[`assets/screenshots/dragon-quest-vii-name-entry.webp`](../../assets/screenshots/dragon-quest-vii-name-entry.webp),
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

1. `g_rtt` (`frontends/shared/live/live_renderer.cpp`) is keyed by **base address alone**, so the later
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
> `tests/gpu/test_pipeline_render.cpp`, not by replaying an artifact.

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

`tests/gpu/state/test_render_state.cpp` used to assert `occurrence_count(..., "MODE=2 ") == 1` under the message
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
  without resting on the mapping it is used to support. Within the residue, ch4/ch6 correlate
  +0.96 and ch5/ch7 +0.95 while ch4/ch5 is -0.04, so
  the surround tier pairs even=left / odd=right; ch2 correlates near-equally with both groups
  (centre-like). Reproduced identically on a second independent 150 s run.
  **The 8-channel port (ctx1/port2) is exactly zero — and that is the GUEST's silence, not
  prosper's blindness (#1721, settled).** 0 non-zero of 55,175,168 samples, and prosper forwards it
  anyway. The reading is now measured rather than inferred, because an exact zero from a read cannot
  by itself distinguish "the guest mixes silence here" from "prosper is reading a buffer the guest
  never fills" — a wrong address is exactly as zero as a silent mix. `PROSPER_AUDIO_STAMP=2:8:13000`
  (see `AUDIO.md`) writes a per-channel-distinct pattern into that port's own grain after consuming
  it and reports what comes back on the next push: **`CLEARED` on every arm** — the stamp is gone
  and the grain is exactly zero, so the guest *actively writes* the buffer at `0x250101e000` and
  writes silence into it. A live bus carrying nothing. The control that makes this worth quoting ran
  on the same port in the same run: the probe's own read-back reported **256 non-zero samples on
  each of the 8 channels**, so the reader demonstrably can report a non-zero sample here. Calibrated
  against port1 in an identical run at the same offset, which returns `OVERWRITTEN` — the guest
  filling it with content. Two further hypotheses are dead: this title imports exactly **14**
  AudioOut2 entrypoints and `sceAudioOut2ContextBedWrite` is not among them (nor in any of the 44
  local dumps), so there is no second PCM path prosper could be missing; and it never imports
  `sceAudioOut2GetSpeakerInfo`, so nothing in its own setup names the bed order.
  **Rung 4 for audio: the project owner confirmed by ear that the music plays and sounds right.**
  That establishes real audio reaching the device at sane levels through the guest's own path. It
  establishes **nothing about the channel order** — with ten of the twelve channels empty, every
  mapping that routes ch0/ch1 to the two sides produces a bed that differs only by the ~1e-9 residue
  on ch2 and ch4..ch7 — about 150 dB below the content, and inaudible by any measure. Not
  "bit-identical": those channels are a residue, not exactly zero, and this document insists
  elsewhere that those are different findings. A left/right swap
  is inaudible without a reference in any case. Do not cite the listening test as layout evidence.
- **Historical pre-fix measurement**, with `PROSPER_AUDIO_FLOW=1` (see `AUDIO.md`) on a
  `reach-title-screen.pad` run of the direct SDL3 frontend. The title creates **two** AudioOut2
  contexts, each with one MAIN
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
  stereo with ten of its twelve channels carrying nothing audible. Measure the channels before
  designing a fold for them; see
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

## The field state is reached — 2026-08-28, and the blocker was the route's patience

**Rung 3.** `scripts/dragon-quest-vii/reach-field-control.pad` reaches the field state in Pilchard
Bay: the field HUD (circular minimap bottom-left, party block bottom-right reading
`Lv.1 / HP 22 / MP 7`), the area-entry banner, the player character standing in the world, and quest
markers over doors. **144 frames spanning t=652-1240 s (588 s).** Linux, RADV STRIX_HALO, native
3840x2160 through `tools/screenshot`, direct frontend, unmodified captures, UE4 recipe, isolated
save roots, master `68f89186`.

Every number in this section comes from one committed script, so it can be re-derived rather than
taken on trust — two earlier drafts published figures from scratch analysis and both were wrong:

```bash
python3 scripts/dragon-quest-vii/classify_field.py field <SHOTS>       # 0 / 0 / 144 / 190
python3 scripts/dragon-quest-vii/classify_field.py world <SHOTS>       # 36/144 run3, 107/190 run4
python3 scripts/dragon-quest-vii/classify_field.py locomotion <SHOTS> --window ...
python3 scripts/dragon-quest-vii/classify_field.py selftest
```

### What was actually in the way

Not a control, not a screen, not the renderer: **how many confirms the opening chapter is given.**

| route | confirms delivered | frames with the field HUD |
| --- | --- | --- |
| `reach-gameplay.pad` (confirm every 15 s, ends t=700) | 41 | **0** |
| a probe route (stick windows, see below) | 37 | **0** |
| `reach-field-control.pad` (every 2 s, ends t=1180) | **447** | **144** |

The two comparison routes also stop earlier, so spacing is not their only difference from the third
— but it is the one that matters, because every screen in the chapter waits indefinitely for
confirm, so a press landing early is absorbed rather than lost.

**Worth generalising:** on a story-opening JRPG, "the route never leaves the cutscene" is a claim
about the ROUTE until the confirm count has been pushed by an order of magnitude.

### Locomotion is MEASURED, and the measurement is on the HUD

`probe-locomotion.pad` mashes Cross to t=700, then alternates eight stick windows against eight
matched neutral windows inside the field state. All eight stick deliveries appear in the pad log
(`axes=left-stick-left/right/up/down`).

| | windows changed >= 15 | median | range |
| --- | --- | --- | --- |
| stick held | **8 of 8** | 24.88 | 22.41 - 37.29 |
| neutral | **0 of 8** | 3.01 | 0.00 - 12.67 |

The **counts** are identical at `--guard` 0, 2 and 4, so the guard band is **not** load-bearing
for the 8/8 vs 0/8 result (the magnitudes do move: stick median 26.67 at guard 0) — an earlier draft
claimed the neutral result depended on it, which running the tool refutes.

**The measure is masked minimap CHANGE, not phase correlation.** Two earlier drafts used
correlation and it fails at both ends here. A long walk changes *which part* of the map is drawn, so
the two crops share almost no structure and the peak lands at zero — reporting a long walk as "did
not move"; two of eight stick windows read exactly 0.0 px that way while their minimaps were plainly
of different places. And the crop must be **disc-masked**: the minimap is a circle in a square box
and the corners show the world behind it, which collapses to black in one frame and white in the
next. Unmasked, that background flip alone scores 56 on a window whose map is pixel-identical.

**On a title whose composite is broken, the HUD is the reliable place to look for locomotion.**

Visually: across the first stick window the quest-marker house that sat centre-left ends up
upper-right, a cliff face enters from the left, and the minimap scrolls to match.
`assets/screenshots/dragon-quest-vii-walked-to-cliff.webp` is that frame.

### The composite: geometry is correct, the lit-material shading is not

Reviewed by eye on the checked-in captures, and the distinction matters for where to look next.
**The geometry is right** — the harbour house, its door and windows, the rowing boat, the cliffs,
the foliage and the character are all in the correct places at the correct shapes. **The 2D/UI path
is correct too**: HUD, minimap, party block and area banner all render cleanly. What is wrong is the
**shading of lit surfaces** — the house, the cliffs and the boat blow to white while the water
crushes far too dark. That reads as one path rather than a general composite failure — though a same-frame
observation cannot separate a broken lighting path from a partially-failing composite, so treat it
as the working hypothesis it is.

Quantitatively, over the field frames: **36 of 144 (25%) render a recognisable scene in run 3, and
107 of 190 (56%) in run 4**; the rest lose the world to a uniform white, a crushed black, or a flat
blue speckle. Quote the run — they differ by more than a factor of two, and the 25% alone
understates the title. Longest run of strictly consecutive rendered samples: 4 (run 3), 10 (run 4).
#1486, #1588.

### Instruments — and why the obvious one cannot work HERE

**A stick-versus-neutral motion probe could not answer in the window it was run.** Matched 32 s
windows alternating neutral against full stick deflection, stick delivery proven in the pad log
(`axes=left-stick-left/right/up`), produced neutral-window medians of 84.2, 2.2, 15.9 and 66.5 — a
spread *larger* than any stick-versus-neutral difference. The reason is now known and is not a
property of the title: **that run reached the field HUD zero times**, so every window sampled a
cutscene, and a cutscene whose own activity varies 40x cannot be a baseline. The probe is the right
experiment aimed at the wrong phase — see the Ruled out entry, which scopes it rather than
forbidding it.

**Classification is ONE test: the party block's HP bar**
(`scripts/dragon-quest-vii/classify_field.py`). Generic alternatives were tried and removed:

- *HUD-corner colour* fails in both directions on real captures here — a torn-composite cutscene
  whose bottom corners hold saturated structured water passes it (that produced two false
  "gameplay" frames in runs 1 and 2), and a flat blue collapse scores 1.00 while containing nothing.
- *A cinematic-bar veto* is a no-op only **above a bar threshold of ~0.052** (0 rejected at 0.06 and
  0.10 across all four runs), because a real cutscene's bars cover the party block anyway. At the
  **0.04 it actually shipped with** it rejected nine genuine "Pilchard Bay: Church" frames whose
  world had collapsed to black, reading unrendered darkness as letterboxing: 190 field frames
  became 181.
- *A collapse or brightness floor* is inverted here — field frames are dark precisely BECAUSE they
  are HUD over an unrendered world. And a structure floor is not safe on this title either: a
  genuine collapse (run 3 frame 188, three distinct colours, nothing rendered) measures sigma
  **13.14** and sits *over* a 12.0 floor, while its neighbour 195 measures 11.56 and is caught. The
  distance from a threshold to the nearest real frame is not margin — it is bounded by whichever
  collapse sits just under the line.

The HP bar separates **0.0201 from 0.008860 across 1,370 frames** of four runs.

**The selftest's coverage is itself re-runnable, and its GAPS are recorded.**
`scripts/dragon-quest-vii/mutants.txt` lists every mutation and `classify_field.py mutants` applies
each one. 30 entries: 25 that must redden, 2 controls that must not (a no-op edit, and a *safe*
`HP_BAR_MIN` retune inside the measured band), and **3 marked `UNPINNED` — mutations that DO move a
published number and that this selftest does not catch**. Those three are a coarser sampling
resize, a narrower luma band, and a small `WORLD_BOX` shift; the reason is that no case has been
written for them yet — **not** that none can exist. An earlier draft claimed the gap was structural;
a reviewer disproved that by building a catching case for each in one sitting. They are listed with
their measured effect so the boundary of the guarantee is visible rather than implied, and so the
next person knows these are open work rather than a wall.

The runner rejects a mutation that does not COMPILE rather than scoring it as a reddening: an
earlier version had eight arms whose replacement indentation was one space short, which raised
`IndentationError`, exited non-zero, and read as "the selftest caught it" while testing nothing.
Replacement indentation is now taken from the matched line, so the class of bug cannot recur.

Thresholds are asserted numerically against measured class boundaries rather than
against constructed frames, because a frame lands *beside* the value it names — that rounding left a
5% band in which `HP_BAR_MIN` passed while destroying the headline. Box controls are drawn at fixed
pixel positions with the expected value **hardcoded**; deriving it from the box under test is what
made three earlier versions of this selftest incapable of failing.

## Ruled out — eliminated, do not re-run these
- **"The opening chapter script is a wall."** **Falsified 2026-08-28.** It is long, not closed:
  raising the confirm rate from one per 15 s to one per 2 s takes the run from 0 field-HUD frames to
  144. See *The field state is reached* above for the three-run table. #1874.

- **"A stick-versus-neutral motion probe can be run in the chapter-script phase."**
  **Falsified 2026-08-28**, and the result is void rather than negative: across four matched windows
  the NEUTRAL medians alone ranged 2.2 to 84.2, so no contrast the experiment could produce was
  interpretable. The cause is the PHASE, not the title — that run reached the field HUD zero times,
  so every window sampled a cutscene. **This does not rule out the probe** — run inside the field
  window it answers, and `probe-locomotion.pad` did exactly that on 2026-08-28. Aim it at the phase,
  and prefer the minimap to the world: see *Locomotion is MEASURED* above. #1874.

- **"No cinematic bars means the run reached gameplay."** **Falsified 2026-08-28**, twice over.
  A present collapsed to uniform white has no dark rows, so a bar detector reports it as un-barred:
  the first version of this measurement returned 61 "un-barred" world frames on a run that never
  left its cutscene, every one RGB(255,255,255). A *colour-count* guard then still missed this
  title's dominant collapse — flat blue with a magenta speckle, 18 distinct colours across a full
  8.3 MP frame. What separates all of them is **structure** rather than brightness or colour count — but the
  gap between a threshold and the nearest real frame is **not** a margin, and this title shows why:
  a genuine collapse (run 3 frame 188 — three distinct colours, nothing rendered) measures sigma
  **13.14** and sits *over* a 12.0 floor, while its neighbour 195 measures 11.56 and is caught. The
  closest sub-floor non-field frame across runs 3 and 4 is 11.95, i.e. 0.05 units away. (An earlier
  draft quoted "3 units of room" from a frame at t=480 s, which is not even in the field phase.) A
  brightness floor is worse still and actively inverted: real HUD-over-dark frames
  measure 8.26-8.9 against a default floor of 8.0, with 73 of 144 within 1.0 of it, while the blue
  garbage measures 29.7-31.9. That narrowness is why the per-title classifier
  (`scripts/dragon-quest-vii/classify_field.py`) keys on the **HP bar** instead, which separates by
  ~1.5x on both sides and does not consult a collapse test at all. And absence of bars is not
  presence of gameplay in any case: a MENU has none either.
  #1874.


- **"Reading a `Map/Product/World/**.umap` package means the run reached the world."** **Falsified
  2026-08-20** over two runs: two World maps are read at **0.0 s**, before the title map, and the
  whole Estard sub-level cluster at **57-58 s**, while the run is still on the save-slot screen and
  roughly ten minutes before the world is entered. This title front-loads, so package *residency* is
  not *activation* and the membership form of the oracle reports gameplay on a menu. The form that
  discriminates — bytes per minute by content directory, plus the persistent level and the
  level-sequence package appearing only in the load window — and the derivation are in
  *Residency is not activation* above. #1874, #2779.

- **"Every compute program this title dispatches executes."** **Falsified 2026-08-19 on `2703a6c3`.**
  A `reach-title-screen.pad` arm through `tools/screenshot` (60/60 samples at 3840x2160, 301 s,
  `stop=request-satisfied guest=running status=ok`) with `PROSPER_COMPUTE_PROGRAM_CENSUS=1` reports
  **65,536 dispatch decisions over 19 programs**, of which **`0x3017400000` executes 0 times against
  2,312 skips**. It is **UE volumetric fog**: two `class=2` sampled inputs (bindings 60, 61) and one
  `class=4` storage output (binding 65) all of exactly **2,088,960** B = `120 x 68 x 32 x 8` — a
  3840x2160 view at `GridPixelSize=32`, `GridSizeZ=32` — with two further bindings on the same
  `120x68x32` grid at 4 and 32 B/cell, under a 3-D dispatch whose Z group count is 32. It rejects on
  `pc=646 words=856a0f0e` = **`s_cselect_b32 vcc_lo, s14, s15`** (llvm-mc, gfx1030), the
  `VCC_LO`-as-scalar-scratch shape `is_gtav_wave64_vcc_lo_scalar_cselect`
  (then `rdna2_to_spirv.cpp:4888`; that file was split by #2752) declined because it required
  inline-constant sources. The same **`s_cselect_b32`** class stops the same pass in *Plucky Squire*
  (`0x3015fd0000`) and *The Pathless* (`0x200ea80000`, `0x200ead0000`). *Little Nightmares III* loses
  its froxel program `0x30114c0000` to the **M0** reject `s_mov_b32 s14, m0` instead, which is a
  different gap. #2747, #2741.
  **That predicate now admits scalar-data sources** (renamed `is_wave64_vcc_lo_scalar_cselect`,
  `rdna2_alu_support.hpp`), so **this** program — `0x3017400000`, whose reject is the
  `s_cselect_b32` — is expected to recompile; the M0-blocked programs are **not** affected.
  **Not re-measured on this title**, and #2747's prediction is explicitly that restoring the pass
  moves no title off its rung.
- **"That missing fog is why this title has not reached gameplay."** **Not supported, and stated so
  it is not assumed.** Volumetric fog is a lighting pass, not a progression gate; the title screen and
  the first-run setup render with it absent, and the run above is clean end to end. The only other
  rejected program, `0x3013e00000`, **executed 26,146 times against 1 skip** — the self-recovering
  descriptor transient of #1581, not a wall. #2747.

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
