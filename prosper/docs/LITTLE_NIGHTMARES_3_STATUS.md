# Little Nightmares III (`PPSA05143`) — status and evidence

Unreal Engine 4. **Rung 2 — title screen** (2026-08-05, master `e98a8fdb`;
`assets/screenshots/little-nightmares-3-title-screen.webp`). Tracked on
[#1893](https://github.com/mattias800/prosper/issues/1893).

## Where it stands (2026-08-05)

**Rung 2 reached: the title screen renders.** On a default launch the title now plays its whole
splash sequence — Bandai Namco → Supermassive Games → Unreal Engine → Wwise — and then presents its
own title screen at native 3840x2160: the *Little Nightmares III* logo, the `Player` slot bottom
left, the copyright line, and the `⊗ Start` prompt bottom right, all legible.

The render-thread stall that used to end every run before the title is **gone**. It was #1962, and
its cause was #1982's ordered-DMA `Jump` decline, fixed by
[#1987](https://github.com/mattias800/prosper/pull/1987). A 360 s bounded arm on `e98a8fdb` ran to
its own end with 36/36 samples source-distinct, 26 pixel-distinct, no `LowLevelFatalError`, no
watchdog abort, and **zero** `[agc] ordered DMA submit rejected` lines — against a hard wall at
t≈60 s before.

```text
[shot] done: 36 screenshot(s); source-distinct=36 pixel-distinct=26
       max-source-stale=0.0s max-pixel-stale=30.0s status=ok
[progress] t=356.9s submits=9817 draws_cum=376986 dispatches=80095 flips=1067 presents=6606
```

### The open defect: the UI's background clears to yellow instead of black

Menu frames are intermittently degraded: the background behind the UI is `RGB(255,255,0)` where it
should be black. **The content itself is correct** — every layer composites over that background
with its own alpha, exactly as it does over black.

> **Superseded framing — do not reuse.** This was first written up as "red and green forced to
> maximum", `out = (255, 255, in.b)`. That fitted the title screen, whose content is all
> white-on-black at one alpha, and it is **wrong**. The EULA screen falsifies it: its dialog panel
> is `(0,0,0)` clean and `(102,102,0)` degraded, where `out = (255,255,in.b)` demands `(255,255,0)`.
> #2014.

The background-colour model fits every measured pixel on **two different screens with three
different alphas**. With content `C` at alpha `a` over background `B`, `out = a·C + (1-a)·B`, and
only `B` differs — `(0,0,0)` clean against `(255,255,0)` degraded:

| element | clean | degraded | implied `(1-a)` |
| --- | --- | --- | --- |
| title-screen background | `(0,0,0)` ×8,090,540 | `(255,255,0)` ×8,090,521 | 1.0 |
| title logo, fully opaque | `(255,255,255)` ×151,941 | `(255,255,255)` ×151,941 | 0.0 |
| title-screen footer text | `(102,102,102)` ×14,539 | `(255,255,102)` ×14,537 | 0.6 |
| EULA dialog panel | `(0,0,0)` | `(102,102,0)` | 0.4 |
| EULA body text | `(153,153,153)` ×31,976 | `(194,194,153)` ×31,949 | 0.16 |

Two of those rows are load-bearing. **Fully opaque white is byte-identical in both frames** — 126,501
pixels on the EULA screen — so nothing is being clamped or channel-swizzled; opaque content is
already correct. And the blue channel is *never* touched in any row, in either frame, which is what
identifies the wrong value as a background colour rather than a per-channel transform.

This is a much narrower place to look than a channel map: the question is why the surface the UI
composites onto is cleared to `(1,1,0)` instead of `(0,0,0)` on roughly two thirds of frames.

What is and is not bounded:

- **It tints frames that contain no content at all, so it is NOT switched on by the post-process
  chain.** It is tempting to read "the logos are clean and the title screen is not" as an onset, and
  an earlier draft of this doc did. Two independent observations kill it, and the second needs no
  cross-build argument:
  - The earliest tinted frame on record is the *empty* composite at `frame_seq=4`, t≈4.0 s (#1962,
    build `ea299e97`); a frame with nothing drawn over a yellow background is exactly the pure
    `RGB(255,255,0)` seen there. Clean logo frames at t=20/40/60/80 s are interleaved with it.
  - On **`e98a8fdb` itself**, ten consecutive samples (t=120–210 s) are pure `(255,255,0)` across
    all 8,294,400 pixels — one colour, no content — while that arm did not composite the title
    screen until t≈230 s. Ten of its twenty-four tinted samples therefore carry nothing that any
    post-process chain could have acted on.

  So the tint is present and intermittent from the first seconds and applies to empty frames, and
  **the post-process/tonemap stage is not localised by this evidence** — do not narrow to it on the
  strength of the splash frames.
- **It is not frontend-specific.** The native SDL3 `prosper-app` window shows it too, and there it
  is *persistent* — three grabs 40 s apart at the title screen are byte-identical yellow. Headless
  `screenshot` sees it on roughly two thirds of samples, with correct black frames interleaved.
- **The rate is 24/36 in both 360 s arms, but that is not yet a calibrated discriminator.** The two
  arms are the default and the `PROSPER_NO_PACKED_R11_STORAGE=1` arm — i.e. the same pair an A/B
  would compare, so their agreement cannot also serve as the baseline's run-to-run variance. With
  n=1 per configuration the null spread is unknown. Worse, the two 24s are not even counting the
  same thing: the treated arm reached the title screen ~120 s later, so most of its tinted samples
  are empty composites where the default arm's are title-screen frames. Establish a null with two
  **same-configuration** arms, and compare within a fixed content window, before reading any future
  "unchanged at 24/36" as a negative.

Open as [#2014](https://github.com/mattias800/prosper/issues/2014).

### 2026-09-05 re-measurement on `c067aeef`: the tint is gone, a flat scanout and a GPU hard recovery are not

Everything in the section above was measured in August. Re-run on `c067aeef` (`tools/screenshot`,
default route, no input, isolated `PROSPER_SAVE0`/`PROSPER_SAVEDATA_DIR`, 40 samples / 5 s / 200 s),
the picture is different in three ways.

**The splash sequence renders correctly.** Twenty-seven consecutive samples from t=10 s to t=140 s
are black-background frames — Bandai Namco, Supermassive Games, Unreal Engine, Wwise — decoded and
viewed, not inferred from `distinct_rgb_colors`. No tint of any kind.

**The title screen is not reached.** At t=140 s, the last content sample, the frame is still the
Wwise splash. See #3340; whether that is a regression or a long-standing condition the August
rung-2 run happened to land in front of is **not established** — the doc's t≈110 s title screen and
the t≈115 s collapse are five seconds apart, so a marginally slower boot misses it with no change to
the fault. Treat "regression" as a hypothesis until an arm at `4d7a2ded` says otherwise.

**The run ends in a GPU hard recovery.** `radv: GPUVM fault detected at address 0x80025e677000`,
`CLIENT_ID: (TCP)`, `PERMISSION_FAULTS: 3` — a shader vector-memory read touching a page it may not.
Afterwards every graphics submission fails and `fresh` freezes (5,260 at `retained=1`, still 5,260 at
`retained=2048`). The twelve byte-identical yellow samples that follow are the retained frame being
re-served; the wall simply froze on a flat one. #3340.

**The flat frames are painted by a draw, and one lever moves them.** `PROSPER_UNIFORMLOG` (extended
in `f326a5f7`) names the producer as a pass rendering into the guest's own registered VideoOut
buffers, `0x9fc0000000` / `0x9fc2000000`, 3840x2160 at `VK_FORMAT_B8G8R8A8_UNORM`. `PROSPER_PASS_LOG`
shows every such pass carries exactly **one** colour-writing draw — a fullscreen composite — so the
flat frames are ones where that draw's output is uniform. Three 120 s arms, identical route and
instrumentation, code point `91812f96`:

| arm | full-coverage VO passes | of scanout passes that **produced pixels** | rate |
| --- | --- | --- | --- |
| control | 2,218 | 2,287 | **96.98%** |
| `PROSPER_LEGACY_CB_DISABLE_MASK=1` | 25 | 2,111 | **1.18%** |
| `PROSPER_CB_EFC_NO_COLOR=1` | 2,255 | 2,326 | 96.95% |

> **Superseded denominator, kept visible rather than swapped out.** These rates were first published
> over the arms' **total** VideoOut pass counts (12,266 / 4,645 / 12,639), giving
> **18.08% / 0.54% / 17.84%**. That denominator includes deferred passes that render nothing, so it
> measured how many passes ran as much as what they wrote. No conclusion moves — the direction and
> the ~80x separation are identical on either normalisation — but the pair a reader quotes does, so
> the old numbers are named here and are not used anywhere else in this document.

Read the **rate**, not the count. Note what the corrected denominator settles: passes that produced
pixels are **2,287 / 2,111 / 2,326** across the three arms, within 8% — so the lever changed **what
those passes wrote**, not how many of them ran. That retires the trap-255 caveat the original
denominator required.

So draws whose `CB_COLOR_CONTROL.MODE` decodes as DISABLE, executed as ordinary colour draws, flood
the scanout. Since #1724 the renderer derives the colour write mask from
`CB_TARGET_MASK & CB_SHADER_MASK` and **ignores `MODE`** by design, so this is not a mishandled mask —
it is the population that decision created. **The lever is not a candidate fix**: #2932 measured the
same lever taking `PPSA02058`'s content samples from 3 of 24 to 0 of 24. Which reading is true here
is open — either these are genuine colour-disabled passes prosper wrongly lets write, or they are
ordinary composites carrying a stale MODE (#1706), and suppressing the second kind removes real
content. `tools/colorstate/colorstate_report.py --by-program` is the instrument that separates them
by naming the shader.

### Which draws flood the scanout (2026-09-05, `PROSPER_COLORSTATETRACE`)

160 s default route, `PROSPER_COLORSTATETRACE=3840x2160`, reduced with
`colorstate_report.py --scanout-prefix 0x9fc --by-program`. 379,160 records, 47,395 draws, 10,233
writing the registered scanout. **This run was healthy** — 19 of 20 samples carry content, no GPUVM
fault, no publish wall — which matters for how the numbers below may be read.

**Only two pixel-shader programs write the scanout at all:**

| program | scanout draws | writing colour | modes | effective mask |
| --- | --- | --- | --- | --- |
| `0x3017370000` | 5,173 | 5,056 | DISABLE 3,852 / NORMAL 1,320 / EFC 1 | `00`x117, **`07`x5,054**, `0f`x2 |
| `0x300f0a0000` | 5,046 | 2,616 | DISABLE 2,513 / NORMAL 2,533 | `00`x2,430, `0f`x2,616 |

Four other programs account for 14 draws between them.

`effective=07` is **RGB written, alpha not written**. Be precise about what that explains: it
accounts for the **zero alpha** in the observed `(255,255,0,0)` — alpha stays at the `(0,0,0,0)`
clear because nothing writes it. It does **not** explain the RGB. Blue is *inside* the write mask and
is being written **as zero by the shader**, so `(255,255,0)` is the program's own export and remains
unexplained. #2014 is about red and green at maximum, and a mask that only reaches the alpha channel
cannot close it.

**Exposure to #1724's change**: `mode=0` with a non-zero effective mask is **3,818 of 6,366** scanout
`mode=0` draws, **60%**. *The Plucky Squire*'s equivalent is **4.79%**
(`tools/colorstate/README.md`). An order of magnitude more exposed, which is why the
`PROSPER_LEGACY_CB_DISABLE_MASK` lever moves this title 33x and is a trade on others.

**The mode does not track shader identity.** Both scanout programs appear under *both* modes, and
`0x300f0a0000` is nearly 50/50. For `0x3017370000` the write mask holds at `07` across the flip —
the field moves while the state it is supposed to describe does not. Where `MODE=DISABLE` is real and
intended, an engine normally zeroes the mask alongside it (Plucky's depth/shadow prepass does exactly
that, which is why so few of its draws were affected). This is evidence for #1706's stale/latched
reading, and it is the **opposite** of what an earlier reading of the lever asymmetry suggested;
recorded on #1706.

**Exact, replacing a lower bound:** exactly **1** ELIMINATE_FAST_CLEAR draw among 10,233 scanout
draws. The EFC null recorded below is therefore a null over a one-draw population.

**The defect-level metric, which is closer to the frame than the pass census above.** Counting the
*retained uniform frames* by colour across the three 120 s lever arms:

| arm | uniform retains | yellow `(255,255,0,0)` | black `(0,0,0,0)` |
| --- | --- | --- | --- |
| control | 64 | **63** | 1 |
| `PROSPER_LEGACY_CB_DISABLE_MASK=1` | 128 | **4** | 60 (+1 opaque black) |
| `PROSPER_CB_EFC_NO_COLOR=1` | 64 | **63** | 1 |

Under the lever the uniform frames do not disappear — they turn **black**, i.e. the pass's own
`(0,0,0,0)` clear showing through where the flooding draw was suppressed. That is a stronger
statement than the pass-coverage rate: it is measured on the frames prosper actually retained, and it
identifies the yellow as a *draw's export* rather than a clear or a fallback.

The pass-coverage rates in the section above are normalised over passes that **produced pixels**, and
that denominator is stable across the arms — see the table there for the figures and for the
superseded denominator they replaced.

**What this run cannot say.** The tool's per-guest-minute series is a single `??:??` bucket — the log
carries no Unreal timestamps to attribute against — so there is no good-phase/bad-phase contrast
here, only a population. `tools/colorstate/README.md` is emphatic that a population alone can
mislead: on Plucky the `mode=0` fraction is *higher* while the world renders correctly than while it
is black. The next arm is `PROSPER_SKIP_DRAW_PROGRAM=0x3017370000`, a one-program A/B rather than a
process-wide lever.

### The flooding draws, named exactly (2026-09-05)

`PROSPER_SKIP_DRAW_PROGRAM=0x3017370000`, 120 s, against the matched control:

| | control | skip `0x3017370000` |
| --- | --- | --- |
| uniform retains, **yellow** | **63** of 64 | **0** of 65 |
| uniform retains, black | 1 | 65 (28 `(0,0,0,0)`, 37 `(0,0,0,255)`) |
| content samples of 30 | 28 | **29** |
| full-coverage / non-empty scanout passes | 96.98% | **0.52%** |

**Declining that one pixel shader removes the flat frames entirely and the title still renders.**
18 `[draw-decline] … reason=skipped-by-selector` lines confirm the selector fired, and its arming
line is in the log.

Grouping the colour-state records by the **(vertex, pixel) pair and the raw register word** — not by
the pixel shader alone — identifies the population exactly:

```text
 3734  es=0x300e900000  ps=0x3017370000  cb-control=1:00cc0000  mode=0  target-mask=1:00000007
 1320  es=0x3016e60000  ps=0x3017370000  cb-control=1:00cc0010  mode=1  target-mask=1:00000007
```

The guest programs `CB_COLOR_CONTROL = 0x00cc0000` (MODE=DISABLE) **together with a non-zero
`CB_TARGET_MASK` of `0x7`**, deliberately, 3,734 times, always with that one vertex shader — and the
two words differ **exactly in bits [6:4]** with no mixing across 5,054 draws. Those two groups
*exhaust* that population: 3,734 + 1,320 = 5,054, which is exactly the `effective=07` count from a
different aggregation, so no third group can exist within it.

**One qualification the whole-population check found, which the per-program view hid.**
`colorstate_report.py --by-pipeline` reports that **1 of 12 `(vertex, pixel)` pairs carries more than
one raw word** — this same pair also appears once under `0x00cc0020` (MODE=ELIMINATE_FAST_CLEAR).
That single draw sits at `effective=0f`, outside the 5,054, so it does not touch the closure above;
but "the register value is pipeline-determined" is true of 11 of 12 pairs on this run, not of all
twelve, and it should be stated that way. On hardware those draws
write no colour. Since #1724 prosper derives the write mask from `CB_TARGET_MASK & CB_SHADER_MASK`
and ignores `MODE`, so it writes them, and what they write floods the scanout.

**The shader does not manufacture a constant.** 782 instructions, 4 `OpImageSampleImplicitLod`, 34
`OpSelect`, **no branches**, one store of a computed `OpCompositeConstruct`, exported through a
`PackHalf2x16`/`UnpackHalf2x16` round trip with `±65504` clamps — a genuine tonemap-shaped chain. So
`(1,1,0)` is a *computed* value rather than a constant this shader holds.

**"R and G arrive `>=1.0`" is a RE-DESCRIPTION of the observed `UNORM8` pixel, not a second
measurement.** `(255,255,0)` in an 8-bit unorm target is exactly what any value `>=1.0` becomes, so
the statement adds an interpretation and no evidence; nothing here has read the shader's output
before the colour block. It must not be cited as independent confirmation. What *is* measured is the
module's structure — computed, not constant — and that is the part that excludes a manufactured
value. That is this issue's **original** "red and green forced to maximum" framing, correct at
the shader level even though the background model that replaced it was falsified at the frame level.

**No change is proposed from this.** #1724 landed on measured cross-title evidence, and #2932
measured the same suppression taking `PPSA02058` from 3 of 24 content samples to 0 of 24. The narrow
reading is that `MODE=DISABLE` with a non-zero mask is a real guest idiom that at least one title uses
at scale, and that treating the mask as the sole authority is wrong for it. Deciding that is #1706's
question, not this document's.

### Past the title screen: the EULA, via a checked-in input route

`scripts/little-nightmares-3/reach-gameplay.pad` presses `cross` at the title screen and the title
advances to its **End-User License Agreement** screen — Bandai Namco's EULA text, a `Decline` /
`Accept` selector with `Accept` highlighted, and the `Navigate / Skip / Select / Back` prompt row.
It renders completely and, in the clean frames, correctly —
`assets/screenshots/little-nightmares-3-eula.webp`.

This is still **rung 2**, not rung 3: a EULA is a menu, not gameplay with real GPU draws. It is
recorded because it establishes two things the next lane would otherwise have to re-derive — that
guest input is reaching this title at all, and where the route stands.

The route is written to be self-invalidating. Run it with `PROSPER_PAD_SCRIPT_LOG=1` and check the
`[pad-script]` transitions before believing any null result:

```text
[pad-script] elapsed=120.249 frame=730 read=733 buttons=cross
[pad-script] elapsed=124.219 frame=737 read=740 buttons=neutral
…8 press/release pairs, each observed at a real pad poll…
```

Without that log, "the guest ignored the input" and "the input was never applied" are the same
observation. Note also that the route's wall-clock windows are calibrated against a **default-route,
lightly-loaded** boot: under GPU contention the title screen arrives later and the presses can land
early. Re-check the log rather than the frames when the route appears to do nothing.

**What remains for rung 3:** accepting the EULA and reaching gameplay. Whether the remaining presses
in the route already accept it is not established — the run that discovered the screen was still on
it at its bound.

### #2003 is not inert here — but it is not the reason the wall went

This title calls both APIs #2003 changed. Under `PROSPER_SVCLOG=1` (9,270 `[svc]` lines in the run,
which is the positive control that the logger was live):

```text
[svc] sceAppContentAppParamGetInt(0x1, …)        once, before the t=20 s sample
[svc] sceNpEntitlementAccessGetSkuFlag(…)        x2, between the t=100 s and t=120 s samples
```

Both `GetSkuFlag` calls land in the window where the title screen composites. The dump answers them
from its own bytes: `sce_sys/param.json` declares `applicationDrmType: "upgradable"`, which
`derive_sku_flag` (`hle_addcontent.cpp:372`) maps to `Full`, and `userDefinedParam1: 0`. Before
#2003 the NID was unregistered and fell to the dispatcher's stub, which reports success while
leaving the out pointer untouched — `hle_service.cpp:3941` names `PPSA05143` as one of the titles
that then acts on uninitialized stack residue.

**Not established:** whether #2003 is *necessary* for rung 2. The wall was at t≈60 s, before either
call, and its removal is attributable to the ordered-DMA rejection going 1 → 0 (#1987, measured).
Settling it needs a revert arm, which has not been run. Do not restate "#2003 is inert for this
title" — that claim was made on this tracker from an unarmed `svc_log` and withdrawn.

### Other things visible in the same run

- Two of 36 samples are a blue/magenta noise band rather than content (t=150 s, t=190 s). *The
  Oregon Trail* shows a "corrupted blue/magenta frame" in its own startup sequence
  (`OREGON_TRAIL_STATUS.md`), so this may be one shape across UE4 titles rather than two defects —
  filed together on that hypothesis as
  [#2028](https://github.com/mattias800/prosper/issues/2028). It is **not** #2014: #2014 changes only
  the background the UI composites onto and leaves the content correct, this replaces the content
  entirely, and both occur in different samples of the same run.
- 17 distinct compute programs are dropped with `[compute] skip unsupported program 0x…`. Each is
  logged once, so the line count is **not** a dispatch count. Per the charter this is a fatal gap
  regardless of whether it turns out to relate to the tint —
  [#2022](https://github.com/mattias800/prosper/issues/2022).
- `sceAgcDcbDrawIndirect` is still unimplemented ([#1977](https://github.com/mattias800/prosper/issues/1977)).

## Reproduction

```bash
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 PROSPER_NULL_PAGE=1 \
  ./build-linux/screenshot <DUMP_ROOT>/PPSA05143-app0 \
      --seconds 10 --count 36 --timeout 380 --out <OUT>
```

The title screen first appears at roughly t=110 s on the **default** route, so any arm shorter than
about two minutes will miss it entirely and see only the splash sequence. `PROSPER_NULL_PAGE=1`
matches the earlier arms on this title and does not change the outcome either way. There is no input
route: `scripts/little-nightmares-3/reach-gameplay.pad` reaches the EULA screen (see above), and
nothing past that has been driven.

**Length is not sufficient on its own, only on the default route.** Three of three default-route
arms reached the title screen. Both `PROSPER_NO_PACKED_R11_STORAGE=1` arms ran markedly further
behind: the 360 s one did not composite the title screen until roughly t=230 s, and a 200 s one of
the same configuration **never reached it at all** and is void for any question about the title
screen. That arm is recorded here rather than dropped, because a bound chosen from the default
route's t≈110 s would have looked generous and still produced nothing. The GPU was shared with peer
lanes throughout, so this is not offered as a timing measurement of the switch — only as the reason
that arm answers nothing.

## Ruled out

Read this before forming a hypothesis.

| Hypothesis | Verdict |
| --- | --- |
| The `Your options save has corrupted and has been deleted` modal at t=170 s is a defect in the title, or in this dump | **Falsified — it was prosper's, and prosper did the deleting.** Save data was not namespaced by title: `/savedata0`'s per-save host directory was `<shared root>/<guest dirName>`, and UE4 titles pick generic names, so this title read an `OptionSettings` slot **another Unreal title** had written, correctly judged it not its own format, and deleted it. A/B in #2734: identical binary and route against a private empty save root reaches the title screen at t≈300 s. Fixed by namespacing both save roots by title id (#2734, `docs/SAVE_DATA_LAYOUT.md`); verified on a live 200 s `boot_trace` of this dump, which writes its own `PPSA05143/OptionSettings/ue4savegame.dpx.sav` and leaves a foreign payload seeded under another title id byte-identical. |
| #2022's **17 distinct compute programs** are 17 blocked programs, and the dedup only hides *how often* | **Partly falsified — four of them are not blocked at all, and the true population is larger.** A 501 s arm on `9ea76a52` (50/50 samples, exit 0) with `PROSPER_COMPUTE_PROGRAM_CENSUS=1` reports **131,072 dispatch decisions over 85 programs, 13 of which never execute**, 16 distinct rejected programs plus 1 on the descriptor-contract path. `0x4186c1f00` executed 2,806 / skipped 1, `0x4187ccc00` 510 / 9, `0x300eb00000` **26,307** / 4 and `0x3016540000` 2,798 / 12 — all self-recovering transients (#1581) that a line count could not tell from a wall. #2747, #2022. |
| The 17 are one gap | **Falsified.** Every blocking instruction disassembled from its exact words with `llvm-mc -mcpu=gfx1030`: three are the #2741 family (`s_mov_b32 s14, m0`; `s_cselect_b32 vcc_lo, s33, s32`; `s_cselect_b32 vcc_lo, 7, vcc_lo`), one needs **`image_bvh_intersect_ray`** — RDNA2 hardware ray tracing, 0 of 2,802 — two are `s_cbranch_execz` control flow, `0x3012020000` is a plain `v_med3_i32 v17, 7, v0, 0`, and the rest are buffer/global address resolution. #2747. |
| The skipped compute programs are this title's own bug | **Falsified.** `0x30114c0000` is UE **volumetric fog** and it is the *same program* as *The Plucky Squire*'s: byte-identical blocking dword `be8e037c` (`s_mov_b32 s14, m0`), byte-identical dispatch `groups=30x17x64 local=8x8 threads=240x136`, and three `class=2` sampled inputs plus one `class=4` storage output all of **16,588,800** B = `240x135x64x8` for the 3840x2160 view. `PROSPER_COMPUTE_DISPATCH_LOG=0x30114c0000` logged **512 dispatches, every one `outcome=recompile-empty`**. #2747, #2741. |
| A 316 s arm is long enough to census this title | **Falsified, and the effect is large.** Same binary, same instrument, same route: **31 programs at 316 s, 85 at 501 s** — 185% more programs for 58% more run time, because the population grows with route depth and because the census only prints at power-of-two totals (#2746). The fog program above is absent from every census row of the 316 s arm while appearing in 73 of its `[compute-table]` lines. #2746, #2747. |
| The stall before the title is the undelivered-GPU-completion family (#232 / #208 / #210 / #984) as its own defect | **Superseded.** The completion was never generated, because the submit owing it was declined: `PROSPER_EOPLOG=1` censused 2,706 `FIRE`, 1 `SKIP(rejected)`, 0 `OWE`. Same root cause as #1982, fixed by #1987. #1962. |
| The flat yellow frame is a real guest screen whose text layer is missing (e.g. a health warning) | **Falsified.** It is pure `RGB(255,255,0)` over the whole 4K frame, first composited at `frame_seq=4` with the identical crc — before any title content exists — and the frame the freeze landed on varied between runs. #1962. |
| The yellow tint is a per-scanout-buffer defect (one flip buffer composited wrong) | **Falsified.** The manifest's `front_index` does not correlate with the tint: 14 tinted / 6 untinted on `front_index=0` and 10 / 6 on `front_index=1` over 36 samples. (Two of the twelve untinted are the blue/magenta noise frames below, not clean content; moving them to either bucket leaves the two columns uncorrelated.) #2014. |
| This title composites content over a yellow BACKGROUND (`out = a*C + (1-a)*B`, `B = (255,255,0)`) | **Does not reproduce on `c067aeef`.** 27 consecutive samples spanning t=10-140 s are correct black-background splash frames, decoded and viewed rather than inferred from a colour count. The only uniform-yellow samples in a 200 s run are one at t=5 s, before any content exists, and twelve after a GPU hard recovery, all byte-identical. The alpha model was correct when measured in August and is retained above as the record; it is not the current state, and the uniform frame is now known to be a flat *scanout* paint rather than a background under content. #2014, #3340. |
| The flat scanout paint is an ELIMINATE_FAST_CLEAR pass writing its bound pixel shader over the target (#1588's mechanism) | **Falsified on this title, with a live control.** `PROSPER_CB_EFC_NO_COLOR=1` over a matched 120 s arm gives **96.95%** full-coverage scanout passes against a **96.98%** control — normalised over passes that produced pixels — and an identical `[uniformlog]` ordinal (#64). (An earlier revision of this row quoted `17.84%` against `18.08%`, the same counts over the arms' total pass counts; the null is identical either way.) Code point `91812f96`. **Scope this null carefully**: it was measured on a title whose MODE=2 population is *small* — `>=4` draws against `>=32,768` MODE=0 draws in a 200 s run — so it says the EFC lever is not this title's mechanism, and says nothing about a title with a large MODE=2 population. The lever that does move it is `PROSPER_LEGACY_CB_DISABLE_MASK`, a different lever acting on a different population. #2014, #1588, #2932. |
| The uniform colour comes from prosper's substitute texture for an invalid descriptor binding (the 2x2 magenta/cyan checkerboard at `live_renderer.cpp:7278`), turned yellow by the `B8G8R8A8` R/B swizzle | **Falsified by construction — the substitute is not armed.** That `poison_tex` is reachable only under `PROSPER_DESCRIPTOR_VALIDATE=poison` (`live_renderer.cpp:7274`, `if (!mode \|\| strcmp(mode, "poison")) return;`), which no arm on this title has ever set. The swizzle arithmetic also does not work: texel (0,0) of that pattern is magenta `(255,0,255)`, and an R/B swap leaves magenta unchanged. Recorded because it is an attractive story — the run does emit `[mimg-unresolved]` lines — and because the arithmetic half is the part that is easy to get wrong in one's head. |
| `PROSPER_CLEARLOG`'s all-zero clear-word census retires the fast-clear hypothesis | **VOID, not negative — the instrument printed a constant.** `extract_render_state` assigned `rs.color0_clear_word0`/`_word1` two lines BELOW the diagnostic that printed them, so it read `RenderState`'s member initializers: every `[clearlog]` line ever printed says `word0=0x00000000 word1=0x00000000`, in every title, for every target. The dedup key used the same unassigned field, so the "33 distinct `(base, format, clear-word)` combinations" were 33 distinct `(base, format)` pairs. The extractor's fields were always correct, which is why every existing assertion on them passed. Fixed and pinned by a mutation-checked arm in `tests/gpu/state/test_render_state.cpp` (#2014). The `PROSPER_CLEAR_DEBUG` blue control from the same session is a different instrument and still stands. |
| The uniform frame is the DCC fast-clear materialiser's output | **Falsified by construction, not by census.** `gfx10_dcc_fast_clear_rgba8` (`src/gpu/texture/tile.cpp:92`) accepts only 3- or 4-component surfaces and only the embedded `0000/0001/1110/1111` codes, and `materialize_uniform_rtt` maps its bytes to RGBA8 unchanged. Its **entire** reachable set is `(0,0,0,255)`, `(0,0,0,0)`, `(255,255,255,255)`, `(255,255,255,0)`, `(255,0,0,0)` and `(0,255,255,255)`. `(255,255,0,x)` is not in it and cannot be, whatever the run contains. **Qualified the same day, by the author, before anyone relied on it:** that is a statement about the MATERIALISED SURFACE, not about what a consumer sees. `backend_sampled_component_swizzle` (`tests/fixtures/render_runner.h:365`) swaps the R and B selectors when the sampled target's guest format is `B8G8R8A8`, so a uniform **cyan** `(0,255,255,255)` -- which IS in the set, from code `0x80` with `alpha_is_on_msb=false` -- reaches a shader as `(255,255,0,255)`. A composite that samples such a surface and writes RGB only, over a `(0,0,0,0)` clear, produces exactly the observed `(255,255,0,0)`. So the row rules out the materialiser as a DIRECT producer and does **not** rule out the chain through a swizzled consumer; `PROSPER_DCCLOG=1` is the one-run test. Note the converse too: flat **white** IS directly reachable, which makes this path a live lead for #2932's signature (a). |
| The yellow tint comes from the packed-R11G11B10 compute storage path | **NOT falsified — the arm is inconclusive, and is recorded here so nobody counts it as a negative.** `PROSPER_NO_PACKED_R11_STORAGE=1` over a 360 s arm gives **24 / 36** tinted samples against the default arm's 24 / 36. But the switch only reaches the *compute* path (`gpu_executor.cpp:4719`) and the packed emission is further gated on a 3-component `Float10_11_11` storage image (`rdna2_to_spirv.cpp:9516`); nothing logs whether that path was ever taken, so "the switch moved nothing" cannot be told apart from "the switch was never in circuit". Needs a counter of dispatches compiled with `packed_r11=true` before it means anything. #2014. |
| `[agc] WaitRegMem … dependency violated` is the lead for the stall | **Falsified — instrument noise.** 31 events on *The Pathless* (`PPSA01826`, UE4), which renders its title screen for a full 140 s arm without stalling, against 40 on this title, and on this title they all stop *before* the stall began. #1962. |
| `crc=666f7b3f` fingerprints this title's wall | **No — it is just "black 3840x2160".** The same crc was the frozen frame of Crisis Core (#1982) and Sonic Frontiers (#1968), which have different causes. Do not group titles by it. |
| The `0x30016000` UE pooled-allocator fault (#1945 / #1226) bounds this title | **Not seen** in any arm of six on `ea299e97`, nor in any arm on `e98a8fdb`. |
| #2003 is inert for this title (no `sceNpEntitlementAccessGetSkuFlag` line in the run logs) | **Void, not negative — the instrument was never armed.** `svc_log()` is gated on `PROSPER_SVCLOG`, which those arms did not set. With it armed the title calls `GetSkuFlag` twice, in the same window the title screen composites. Claimed and withdrawn on #1893. |
| The boot dies in the guest at `addr=0x80` | **Falsified — the faulting instruction was prosper's own.** `k_ef_create` (`hle_kernel.cpp:1792`) storing through a guest out-pointer of `0x80`; the `rip=eboot+0x…` label was instrument-trap 22. The `0x80` itself came from `sceAjmInitialize` rejecting this title's config revision, fixed by #1966. Filed as #1963. |

## Ladder

- [x] Rung 1 — real graphics from the live renderer (splash sequence, native 3840x2160)
- [x] Rung 2 — title screen reached and rendered (degraded by #2014)
- [ ] Rung 3 — gameplay with real GPU draws — route reaches the EULA screen, not gameplay
- [ ] Rung 4 — manual visual verification
- [ ] Rung 5 — PS5 hardware-oracle comparison
- [ ] Rung 6 — reviewed automatic gameplay snapshot guard
