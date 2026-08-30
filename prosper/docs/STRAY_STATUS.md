# Stray (`PPSA02101`) — status

Tracker: [#2883](https://github.com/mattias800/prosper/issues/2883). Engine: Unreal Engine 4.
Route: [`prosper/scripts/stray-PPSA02101/`](../scripts/stray-PPSA02101/README.md).

**Rung 2.** A Cross-only route accepts the brightness-calibration screen and reaches the first map
load (`hk_project_mainstart`, t ≈ 37 s, absent from every default run). Two separate things are wrong
after that, and they are worth keeping apart because they have different evidence:

| what | issue |
| --- | --- |
| the world composites as a flat **letterboxed** clear — no scene renders | [#2932](https://github.com/mattias800/prosper/issues/2932) |
| the **title screen** renders its menu but not its background | [#3126](https://github.com/mattias800/prosper/issues/3126) |

## READ THIS BEFORE QUOTING A NUMBER FROM THIS DOC

**`max_nonblack = 0.1140` is the brightness-calibration screen, NOT the title screen.** The title
screen is **0.0069**. Both are held steady states and both look like "the run settled", so the two are
easy to confuse — and every live A/B in the section below was originally run against the wrong one.

| screen | `max_nonblack` | what it looks like |
| --- | --- | --- |
| brightness calibration | **0.1140** | **correct and complete** — three cat-head silhouettes at increasing brightness, the instruction text, the 16-step slider, `△ Defaults` / `✕ Accept` |
| title screen | **0.0069** | menu only — `START GAME` / `SETTINGS` / `CREDITS`, the version string and `✕ Select`, everything else black at 8× brightness |

The calibration screen rendering correctly is real, previously unrecorded progress: it is a full
screen of the title's own art and UI, drawn right.

**Reaching the title screen at all needs `PROSPER_NULL_PAGE=1`.** Without it a run either faults
(`rc=90`, a `[nullpage]` report) or survives and renders pure black after Accept. With it the route
still only lands roughly one run in three, so **repeat any A/B here** rather than trusting a single
arm.

**`reach-title-flip.pad` does not reach the title screen.** Measured over a 320 s run: 16 of 16
captured frames are the calibration screen at 0.1140. The route README's claim that it stops at the
title is wrong and needs correcting.

## Three routes, and which to use

| route | reaches | notes |
| --- | --- | --- |
| `reach-title-hold.pad` | the **title screen** (0.0069) | needs `PROSPER_NULL_PAGE=1`; lands about one run in three |
| `reach-title-flip.pad` | the **calibration** screen (0.1140), held | flip-anchored and deterministic — a stable oracle for that screen, and *not* a title route despite its name |
| `reach-first-map.pad` | past calibration to the first map load | |

`reach-title-flip.pad` was landed as a title route and is not one; the banner above records the
measurement. It is kept because holding calibration deterministically is genuinely useful — that
screen renders correctly — and because renaming a committed route would break every citation of it.

## Performance: this title is not GPU-bound

Measured 2026-08-29 with an F8 `.prperf` capture over a 5.02 s window at the title screen, read with
`tools/perf/performance_capture_report.py`:

| bucket | ms |
| --- | --- |
| `gpu-device` | 221.7 (≈ 4% of the window) |
| compute | 2023.8 |
| texture materialisation | 1137.4 |
| buffer copy | 602.0 |

One compute program, `0x3011300000`, accounts for ~605 ms of that at roughly 32 ms per dispatch at
3840×2160. So the cost is on the host side of the boundary, and pointing a GPU profiler at this title
answers a question it does not have. Detail on [#3126](https://github.com/mattias800/prosper/issues/3126).

## The unresolved image ops — established on CALIBRATION

> **The five-op census below was read on the calibration screen**, like every other live census above
> § *The title screen's REAL numbers*. The resource-table finding it rests on is a statement about
> shader stages and holds regardless of screen; the count of five does not.

`[mimg-unresolved]` reports five image ops on one routed boot whose descriptor resolves to nothing;
every draw using those shaders is discarded. What is **established**: for the shader stages that were
dumped, the resource table is complete and correct — each declared read-only T# is present, the
SGPR-resident one under DIRECT (`sgpr_base`) provenance and the EUD-resident ones under INDIRECT
(`srt_offset = (offset_dw - num_user_sgprs) * 4`) provenance, which is exactly what the table shows.

What was **not** established at the time of that census is which stage the five failures belong to.
Every line reported `program=0x0`, because `recompile_fragment_impl` hardcoded a zero program address
([#3130](https://github.com/mattias800/prosper/issues/3130)) — so no fragment-shader recompile failure
on any title could be attributed to a shader. **That is fixed** (#3132): fragment diagnostics now carry
the real guest program address, which turns this from an inference into a lookup. The five failures
above predate the fix and were never re-attributed, so they remain unassigned to a stage — re-running
the census is what would assign them.

Separately, four stages declare a *writable* 8-dword T# that reaches no resource table at all
([#3128](https://github.com/mattias800/prosper/issues/3128)); one of them has four image ops against a
completely empty table. Whether that is what its image ops want is untested — see Ruled out.

## The dropped-draw census — measured on CALIBRATION

> **VOID as title-screen evidence, and this warning covers BOTH censuses below it.** Every number in
> this section was read on the brightness-calibration screen (`max_nonblack` 0.1140), not on the
> title screen (0.0069). The section is kept because these numbers are cited elsewhere and need
> somewhere to point — **not** because anything in it survives as title-screen evidence. An earlier
> version of this banner claimed two *mechanism* findings did survive: that no `CB_TARGET_MASK`
> register is being lost (so this is not the *Oregon Trail* defect #1946), and that the
> colour-masked-off draws come from one shader. Both are marked VOID in `## Ruled out`, and the
> banner cannot exempt them: a trace that ran only on calibration says nothing about which registers
> reach the GPU on the title screen, mechanism or quantity. Every number here describes the wrong
> screen. Do not
> quote the 1024, the 7, the 32,649 or the 92% as facts about the title screen; § *The title
> screen's REAL numbers* has that screen's own census, and it disagrees.

`PROSPER_DROPPED_DRAW_CENSUS=1` on the **calibration** route, at 1024 discarded draws:

| reason | count | targets |
| --- | --- | --- |
| `no-effect(early)` | 1015 | `0x9fc0000000` (511), `0x9fc2000000` (504), two others |
| `shader-recompile` | 7 | `0x9fc2000000` (4), `0x9fc0000000` (3) |

So **on calibration** the unresolved image ops cost 7 draws, and nearly everything discarded is
dropped because every colour target's write mask is zero with no depth/stencil side effect. That
conclusion does **not** transfer: the same census on the title screen reads ~3800 `shader-recompile`
of 8192 discarded.

**But the dropped draws are not where the missing picture is either.** `PROSPER_COLORSTATETRACE=1`
over a shorter window traced **32,649** draws, and every single one carries a *decoded*
`CB_TARGET_MASK` — the presence flag is 1 on all of them, so no register is being lost, which is what
would have made this the *Oregon Trail* defect (#1946):

| `CB_TARGET_MASK` | draws |
| --- | --- |
| `0x0f` | 26,774 |
| `0xff` | 1,655 |
| `0x07` / `0x0c` | 884 / 838 |
| **`0x00`** | **2,498** |

Roughly **92% of draws write colour and execute**. The colour-masked-off draws all come from **one**
fragment shader, `0x3010660000`, which exports a full `0xf` colour mask and runs with
`DB_DEPTH_CONTROL` `0x70`/`0x62` — Z_WRITE off, no stencil — so under the RDNA rule (the masks are
upstream hardware gates that an export can narrow but not enable) they genuinely write nothing, and
`PROSPER_FORCE_COLORWRITE=1` admitting them makes the composite worse.

**So the title screen is missing content while the overwhelming majority of its draws execute** — but
see the frame dissection above before concluding, as I did at first, that the discarded draws
therefore do not matter. They are few and each one is full-screen. The one thread still worth pulling on the dropped side is whether `CB_TARGET_MASK=0` is
*stale* for that shader rather than current — #1706 established that prosper's decoded
`CB_COLOR_CONTROL.MODE` is not per-draw-trustworthy because a utility sequence's bits stay latched
onto later ordinary draws, and the same shape would explain one shader consistently reading zero.

## The title-screen frame, dissected offline

A **captured F9 bundle of the title screen already exists** and reproduces the defect deterministically
without booting the game — the fastest loop this title has. Its submit-14072 capsule inspects as:

```
submit=14072 3840x2160 draws=92 computes=53 operations=156 shaders=113 failed=11
```

**Eleven of 156 operations never realize** — ten draws and one dispatch — and every one of the
`shader-recompile` drops is a **full-screen 3840×2160 pass** with `CB_TARGET_MASK=0xf`, scissor
`0,0..3840,2160`, alternating between the two swap targets `0x9fc0000000` / `0x9fc2000000`. That is
why the count looked negligible and the picture is not: seven-to-eleven draws is nothing by count and
the whole composite by area. **Do not dismiss these by draw count** — that mistake is what sent this
investigation down the `no-effect` path.

`gpu_replay --retry-failed-stage <failure>:<stage>` with `PROSPER_DBG=1` names each cause exactly:

| failure | stage | reject |
| --- | --- | --- |
| 1, 2 | vertex `0x300f190000` | `v_mbcnt_lo/hi_u32_b32` with **SGPR masks** (`s4`/`s5`, `s6`/`s7`) at pc 277-286 |
| 3 | fragment `0x30be800000` | MIMG `op=0x1`, `recompile-reject-mimg-address extra=1` at pc=134 |
| 6-9 | fragment `0x300c010000` | `s_mov_b32 s0, m0` — `scalar-data-reject pc=37 special=s124 tracked=0` |
| 10 | compute `0x300e390000` | the same `s_mov_b32 sX, m0` at pc=157, behind a `nested-backedge-in-body` loop reject at pc=688 |

Two more fragment programs fail in other draws of the same frame: `0x300e500000` (`unsupported=29`,
first reject pc=20) and `0x3011560000` (`unsupported=1`, first reject pc=383).

The vertex one is the deepest. `0x300f190000` contains **both** MBCNT forms: the canonical all-ones
lane-id pair at pc 4/7, which prosper can lower, *and* four general **SGPR-mask** MBCNTs at pc 277-286,
which need real cross-lane mask state in a vertex stage. `rdna2_to_spirv.cpp` fails those closed on
purpose — its comment says the wave approximations are "an exception for the one captured Astro
wrapper, not a property of the GS_ALLOC_REQ opcode" — and the presence of a non-all-ones form is
exactly what sets `logical_mbcnt_invalid` and disqualifies the `ngg_logical_lane` model. So this is a
wave-semantics-in-vertex frontier, not a missing opcode.

**Reproducing any of it takes seconds, not a boot:**

```bash
gpu_replay <capsule>.prgcap --inspect-only                     # the failure/operation list
PROSPER_DBG=1 gpu_replay <capsule>.prgcap --retry-failed-stage 6:1
gpu_replay <capsule>.prgcap --dump-failed-shader 1:1 vs.bin    # the raw guest program
```

## The scene targets are black AT SOURCE

The obvious next hypothesis after "most draws execute" is that the world renders into an offscreen
HDR target and the tonemap/composite that reads it is lost. **It is not.** Every render target the
title-screen frame samples can be dumped straight out of the capture — no boot, no guessing which
draw wrote what:

```bash
gpu_replay <capsule>.prgcap --dump-rtt-seed 0x3096fd0000 out.bin   # writes a BMP
```

| seeded target | format | brightest channel |
| --- | --- | --- |
| `0x3096fd0000` 3840×2160 | rgba16f | 5 |
| `0x30a4a20000` 3840×2160 | rgba16f | 0 |
| `0x3071dd0000` 1920×1080 | r11g11b10f | 6 |
| `0x300a790000` / `0x3025380000` 480×270 | r11g11b10f | 4 / 0 |
| `0x3092b10000` 3840×2160 | rgba8 | 0 |

Every HDR scene target is black at source, so the world's colour never exists to be lost downstream.
**This is not a composite defect.** The draws run and write nothing — the same family as *Grand Theft
Auto V* and *Sonic Frontiers*, not a missing pass.

**Two of the targets are traps rather than evidence.** `0x3094b60000` and `0x30a29e0000` score
`nonblack=0.2543, mean=63.8` — by far the brightest things in the frame — and both are a **white
quadrant covering the top-left 1920×1080**, i.e. prosper's own seed-miss fill. `0x30563d0000` carries
the seed-miss *gradient*. A brightness metric ranks all three above any real content in this frame,
which is the recorded hazard: open the image before believing the number.

## Measured on CALIBRATION, and only one row survives as title-screen evidence

(The one is the composite/tonemap row, which was established on the title-screen bundle. An earlier
version of this section counted two, by crediting the colour-write-mask row with a surviving
"mechanism" its own VOID marking denies.)

Every number in this table was read on the 0.1140 calibration screen. The table is kept because it is
cited, and annotated because three of its four rows were read as ruling out causes on the title
screen, which they cannot do. **The next section has the title screen's own numbers, and they
disagree.**

| candidate | measurement (calibration) | title-screen status |
| --- | --- | --- |
| dropped draws | 7 `shader-recompile`, ~30,000 draws executed | **FALSIFIED for the title screen** — ~3800 there, see below |
| skipped compute | `0x300ba70000` executed **7455**, skipped **2** (`PROSPER_COMPUTE_PROGRAM_CENSUS=1`) | not re-measured on the title screen |
| lost colour write masks | present and decoded on 32,649 of 32,649 traced draws | **VOID** — a trace run only on calibration says nothing about which registers reach the GPU on the title screen |
| composite / tonemap | the HDR sources are black before it runs | holds — established on the title-screen bundle |

## The title screen's REAL numbers (measured on a 0.0069 frame)

Everything above this section that quotes a live census was measured on calibration. Each such
section carries that warning **at its own head, above the first number it covers** — a retraction a
hundred lines below the number it retracts is one most readers never reach, and a banner placed
mid-section silently exempts whatever sits above it. These are the
title screen, `PROSPER_NULL_PAGE=1`, census read at the same cumulative total in every arm:

| | calibration (0.1140) | **title screen (0.0069)** |
| --- | --- | --- |
| draws discarded | 1024 | **8192** |
| `shader-recompile` | 7 | **~3800** |
| `no-effect` | 1015 | ~4400 |

Two targets take almost all of it — `x1260` and `x835` in one run. **That**, not the seven drops this
investigation spent hours on, is why the background is black.

`PROSPER_DBG=1` on the same route gives the remaining work list, in order of instances:

| reject | count | note |
| --- | --- | --- |
| `[vertex-recompile-reject] body or export lowering failed` | 15 | |
| MUBUF `unresolved-operand`, `fmt=12` | 14 | see below |
| MIMG `fmt=14` | 3 | #3134's family |
| `scalar-data-reject special=s124` | 4 | fixed by #3133 |

### The MUBUF class, decoded

```
sh=/77  pc=57 words=e00c2000,6a010000 op=0x3 src=0(k2),4(k1),106(k6)  stage=vertex
sh=/221 pc=66 words=e0042000,6b020900 op=0x1 src=0(k2),8(k1),107(k6)  stage=vertex
```

Both are `buffer_load_format_*` with `idxen` — **vertex attribute fetch** — in small vertex shaders,
with `VCC_LO`/`VCC_HI` as the soffset. The opcodes themselves are implemented (`0x0`-`0x3` are all
handled), so the reject is an operand, not a missing instruction. The next thing to establish is
whether the failing operand is the soffset or the SRSRC: `allow_smem` is `(rt != nullptr)` for
graphics, so **a vertex stage that arrives with no resource table fails every buffer op it has**, and
vertex fetch is a buffer op — that would take the draw with it. A probe on `rt == nullptr` at the
vertex recompile answers it in one run.

### What #3133 is worth here

Measured on this screen, census at the same total, two arms per build:

| build | `shader-recompile` @ 4096 discarded |
| --- | --- |
| baseline | 1026, 989 |
| with #3133 | **866, 886** |

A ~13% reduction, groups non-overlapping. The visible frame does not change — the remaining ~870
drops still discard the background.

## Ruled out

One line per falsified hypothesis, the evidence, and the link. Do not restart these without
contradictory new evidence.

- **"The descriptor is at the right SGPR base and mis-classified as a buffer."** Falsified. A
  whole-table dump (SRSRC set + finished table + the guest's four raw sharp arrays, per stage) shows
  the tables for the dumped stages are complete: five declared T#s, five present. #3126.
- **"The table being consulted and the code being scanned belong to different stages."** Falsified by
  the same dump — same stage, and the join is exact once the sharp arrays are printed alongside. #3126.
- **"The lost `sreg_srt` tag explains the `srsrc=s8/s16` failures."** Void, not merely wrong: the
  diagnostic's own `written=0` says the shader never wrote those SGPRs, so nothing was `s_load`ed and
  no tag could be lost. The resolver correctly falls through to `by_sgpr_base`, which is what returns
  null. #3126.
- **"Admitting the dropped writable T#s (#3128) fixes the title screen."** One arm says no: admitting
  them as `StorageImage` left both `mimg-unresolved` (5) and `max_nonblack` (0.1097) exactly
  unchanged. That arm may have used a class/provenance the MIMG resolver does not match on, so the
  question is open — but do not assume the fix moves this title. #3128.
- **RETRACTED — "A wall-clock pad route reaches the title screen." (was: *Falsified*.)** The evidence
  was 0.1095 once, then 0.0063 / 0.0000 / 0.0063 on three unmodified reruns, read at the time as "the
  timed route works once by luck". Under the corrected number map that reading **inverts**: ~0.11 is
  the *calibration* screen and ~0.006 is the *title* screen, so **two** of the three "failures" (the
  0.0063 pair) are the runs that reached the title, the "lucky" 0.1095 sample is the one that did not,
  and 0.0000 is neither screen — a black frame is still a failure. Two of four, which is consistent
  with the "roughly one run in three" this doc records elsewhere; an earlier version of this row said
  three of four by counting the black frame as a success. This doc now commits a wall-clock
  route (`reach-title-hold.pad`, Cross at 150 s/160 s) as the title route for exactly that reason. The
  row is kept rather than deleted because the mistake it records — adopting a single sample as an A/B
  baseline — was real even though its conclusion was upside down. #3127.
- **"The unresolved image ops are what is missing from the title screen."** ~~Falsified by the
  dropped-draw census~~ — **this row was wrong and is retained as a warning.** The census reading (7
  of 1024 discarded draws are `shader-recompile`) is correct, and the inference from it is not: every
  one of those draws is a **full-screen 3840×2160 pass**. Draw *count* is not area, and the failing
  shaders are the composite. #3126.

**VOID, not falsified — every row below whose bullet begins `VOID`**, and only those. The scoping is
per-row and stated in the row itself, never positional: an earlier version of this paragraph said
"the next three rows", which silently changed meaning the moment a row was inserted, and did. Every one was measured
on a route that settles on the
**brightness-calibration** screen (`max_nonblack` 0.1140), not the title screen (0.0069). They are
correct statements about calibration and say nothing about the title screen, so they are neither
evidence nor falsifications for it. Re-run each with `PROSPER_NULL_PAGE=1` and a window that reaches
0.0069 before quoting any of them. #3126.

- **VOID — "prosper is losing `CB_TARGET_MASK` for the no-effect draws" (the #1946 shape).**
  `PROSPER_COLORSTATETRACE` reported the presence flag as 1 on 32,649 of 32,649 traced draws — on
  calibration. #3126.
- **VOID — "The dropped draws are where the missing picture is."** ~92% of draws carried a non-zero
  write mask and executed, and the colour-masked-off ones were a single fragment shader
  (`0x3010660000`) — on calibration. #3126.
- **VOID — "`no-effect(early)` is discarding content that should render."** One arm said no:
  `PROSPER_NO_EARLY_NO_EFFECT=1` moves the same draws to the late `no-effect` verdict with
  `max_nonblack` unchanged at 0.1140, and `PROSPER_FORCE_COLORWRITE=1` makes the composite **worse**
  (0.1140 → 0.0296). Neither lever admits content. That does not prove the masks were decoded
  correctly, only that admitting these draws wholesale is not the fix. #3126.
- **"The `srsrc=s0` failures are a size-0 slot wrongly claimed by the V# path."** Falsified by a
  built arm: making a guest-declared 8-dword slot decline the V# claim when its eight dwords decode
  as a valid T# changed nothing — on the draw that recompiles, those eight dwords are **residue**
  (`degenerate T#`), so the guard correctly falls through and the slot is still claimed. The real
  shape is #1590's: the T# is absent on the draw that happens to compile the shader. #3126.
- **"The world renders into an HDR target that the composite then loses."** Falsified: every seeded
  scene target in the title-screen frame is black at source (brightest channel 0-6 across five HDR
  targets). #3126.
- **VOID — "A skipped compute dispatch collapses the composite" (the charter's LUT/exposure shape).**
  `PROSPER_COMPUTE_PROGRAM_CENSUS` reports the only program with any skips at **executed=7455,
  skipped=2** — but that census was read on **calibration** and has not been re-run on the title
  screen, so it is not a falsification for it. This row said `Falsified` and was corrected: the
  measurement is real, the scope was overclaimed. #3126.
- **MIMG `SRSRC` is not a user-SGPR index.** It names any SGPR, so a scan that treats every SRSRC as
  a user-data slot reports mostly false positives — scratch registers the shader loaded a descriptor
  into. An earlier list of "bases missing from the table" derived that way is discarded. #3126.
