# Sonic Frontiers (`PPSA03831`) — status

Tracker: [#1891](https://github.com/mattias800/prosper/issues/1891). Engine: **Hedgehog Engine 2
(Needle renderer)**, SurfRide UI, statically linked CRIWARE middleware — determined from the eboot's
own embedded shader source paths (`Library\hedgehog\…`, `Library\needle\…`) and the dump's
`NeedleShader.pac` / `raw/hedgehog/` asset trees, not assumed from the publisher. *Sonic Origins*
(#1871) and *Sonic Racing: CrossWorlds* (#1895) share parts of the same stack.

## Current rung — gameplay reached, world not rendered

A committed input route (`scripts/sonic-frontiers-PPSA03831/reach-gameplay.pad`) takes the title
from boot to **`GameModeStage` running a Cyber Space stage (`w6d01`)** with real GPU work: the
guest streams all one hundred `w6d01_trr_s00..s99` terrain sectors plus `w6d01_gedit`, loads
`ui_gamemodestage*.pac`, streams `sound/cyber_sound/bgm_cyber.awb`, and its submitted draw rate
rises from **48 draws/flip on the title screen to a sustained 449 draws/flip** in the stage.

The gameplay HUD composites correctly at 3840x2160 — ring counter, the five Red Star Ring slots,
the boost gauge, and **a stage timer that advances monotonically with the guest's flips**
(00:52.39 -> 00:56.76 across one 55-sample capture; 01:02.36 -> 01:05.83 across a second run).
A running stage clock is the discriminator this title offers and a menu cannot fake it: no
aggregate frame metric was used to make the call. Checked-in capture:
`assets/screenshots/sonic-frontiers-cyberspace-hud.png` (direct unmodified `tools/screenshot` frame,
3840x2160, route arm, stage clock at 00:55.89).

**What is not there is the world.** The 3840x2160 frame is black behind the HUD, because
**16 of the stage's 32 compute programs never execute** (#2790). See the section below. So the rung is deliberately not
ticked as a rendered-gameplay milestone: the route reaches gameplay, and a rendering defect stands
between that and a gameplay screenshot.

## What stood between the title screen and gameplay: a twelve-page boot notice queue

A no-input arm never leaves the title screen because a **modal notice queue** opens over it and
stays. Its pages are the game's own post-update notices — "Extras", "Update notification",
"Game update", "Update Details", "Update", "Additional options", "Action Marks", "New Game+" — each
a blue header band over a full-width body panel. Measured behaviour, all from live captures:

| Question | Answer, and how it was measured |
| --- | --- |
| How many pages? | **Twelve.** Confirms 40 flips apart: presses 1-12 each advance one page and press 13 activates a main-menu entry. |
| What advances it? | **Face buttons only.** A single-button sweep (triangle, square, circle, options, touchpad, l1, r1, right, left, down, up, cross, 60 flips apart) advanced the panel on every face button and on **none** of the four d-pad directions — `right`, `left`, `down` and `up` left the header on "Update Details" for 240 flips, then `cross` advanced it. |
| Where is the cursor afterwards? | On **"Extras"**, the last of the six main-menu entries, so five `up` reach "New Game" whether or not the list wraps. |
| Is the queue route-stable? | Only against an isolated save area. `PROSPER_SAVE0` selects it, as it selects the rest of this title's route. |

This is why 405 dense confirms (one every 20 flips) got no further than six did: the pages are
consumed one per press with an animation between them, so spacing, not volume, is what clears the
queue. It is also why the earlier reading of this panel as "renders almost no text" (#2206) is
incomplete — see below.

## Route

`scripts/sonic-frontiers-PPSA03831/reach-gameplay.pad`, flip-anchored, with a header explaining
every window. The shape:

| Flips | Input | Reaches |
| --- | --- | --- |
| f1100-f1540 | 12 x `cross`, 40 apart | clears the boot notice queue |
| f1700-f1940 | 5 x `up`, 60 apart | main-menu cursor "Extras" -> "New Game" |
| f2100 | `cross` | confirmation dialog |
| f2300 | `cross` | answers it |
| f2600 | `cross` | `GameModeOpening` + `raw/event/scene/ev0020*` |
| f2900 | `cross` | `GameModeStage` + `w6d01` terrain + `gedit` + stage HUD pack |
| f3300+ | held `cross` | skips the in-engine opening into the stage |
| f5200+ | `left-stick-up` blocks | forward motion under player control |

```bash
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
PROSPER_PAD_SCRIPT=@prosper/scripts/sonic-frontiers-PPSA03831/reach-gameplay.pad \
PROSPER_SAVE0=~/frontiers-work/save/run1 \
  ./build/screenshot <DUMP_ROOT>/PPSA03831-app0 \
  --warmup-seconds 420 --seconds 3 --count 55 --out ~/frontiers-work/shots/run1
```

**Reproduction:** the file-oracle result reproduced on four CPU-only `boot_trace` arms and the live
HUD-with-running-clock result on two `tools/screenshot` arms. The `--warmup-seconds` figure is a
host-speed convenience, not part of the route — the route itself is flip-anchored and the same
windows drive a CPU-only arm at ~30 flips/s and a live 3840x2160 arm at ~3 flips/s unchanged.

## The world is black in the stage — 16 of 32 compute programs never execute (#2790)

The gameplay HUD is correct and complete; behind it the frame is black. The cause is upstream of the
composite: with `PROSPER_COMPUTE_PROGRAM_CENSUS=1` in the stage the final census block reads

```
[compute-census] 131072 dispatch decisions over 32 program(s)
```

and **sixteen programs are listed with `executed=0`** — the census prints per-program detail only for
programs that skipped at least once (#2745), so sixteen of the thirty-two never ran at all. Three of
them dispatch **2880 threads wide**, the exact width of this title's scene target, so they are
screen-space passes over the frame the player is supposed to see.

The reject classes, and their share of the sixteen:

| Reject | Programs | Encoding |
| --- | --- | --- |
| `unresolved-operand`, SOP2 reading **SCC** (`ssrc0=253`) | 5 | `886a6bfd` `fmt=0 op=0x10` |
| `unresolved-operand`, SOPP `s_cbranch_execz` | 2 | `bf880027` `fmt=4 op=0x8` |
| `unresolved-operand`, SOP2 | 2 | `856a802b` / `856a802f` `fmt=0 op=0xa` |
| `unresolved-operand`, MIMG | 2 | `f0040308,00000101` `op=0x1 dim=1`; `f0380328,00091103` `op=0xe dim=5` |
| `unresolved-operand`, SOP1 writing `dst=126` (EXEC_LO) | 1 | `befe3bff,00000000` `fmt=1 op=0x3b` |
| `unresolved-operand`, VOP1 with SDWA | 1 | `7e2c0ef9,00061216` `fmt=7 op=0x7` |
| `compute-cfg-reject reason=exact-wave-dispatcher-unsafe guest-barrier=1` | 2 | — |
| `unrecorded` | 1 | — |

The SCC group is not an unknown encoding: `rdna2_emit_alu.cpp:764-792` deliberately **poisons** the
tracked SCC when a wave-mask op writes `SCC = (mask != 0)`, because that is a cross-lane reduction
the per-invocation model cannot form, "so a later consumer … rejects instead of misreading". The
poison is the right default; the cost here is five whole programs, and the `s_cbranch_execz` pair
looks like the same family reached through EXECZ. So the largest single lever is plausibly
**wave-level SCC/EXECZ semantics for compute** rather than sixteen unrelated gaps.
`CONFIDENCE: MED` on that grouping — the SOP2 operand decode and the family argument are inference;
the census numbers and encodings are measured.

### What the presented frame looks like, and what it rules out

- Almost every published frame is `guest_scanout` — prosper composited nothing for that flip and
  republished the guest's own display buffer, which holds only the HUD. In the stage window:
  3 composited of 55 samples, 2 of 50, and 3 of 100 across three arms.
- The guest-composited HUD spans `x[64..3776] y[59..2090]` of 3840x2160 — full frame, correctly
  placed, about 1% of pixels non-black.
- The frames prosper *does* composite are confined to the top-left **2880x1620**, exactly 75% of each
  axis and unscaled, and contain a flat blue-grey gradient — sky and fog with no geometry, which is
  what a scene target looks like when its shading passes never ran. 2880x1620 is Hedgehog Engine 2
  dynamic resolution at 75%; `CONFIDENCE: MED` that the missing step is the guest's own
  upscale/resolve.
- **The present path is not broken in general.** In the same arms prosper composites the in-engine
  *cutscene* correctly at full width (`x[0..3839] y[272..1887]`, letterboxed). Only the stage fails.
- **It is not a `--warmup-seconds` artifact.** The control arm used `--warmup-seconds 90`, so the
  renderer was live continuously from flip 1517 — before `GameModeStage` loaded at flip ~2900 — and
  produced the same HUD-over-black frames from flip 3429 to 4325 and the same 2880-wide rect.

This is the frontier for this title, and it is plausibly a **Hedgehog Engine 2** finding rather than
a Frontiers one: *Sonic Origins* (#1871) and *Sonic Racing: CrossWorlds* (#1895) share the Needle
stack, and CrossWorlds' "the composite then goes uniform" (#2013) deserves a census taken the same
way before it is treated as unrelated.

## Rung 2 — title screen and main menu (still current, still checked in)

A default launch reaches the whole 4K opening sequence, the auto-save notice screen, the title
screen and the main menu. Checked-in captures, all direct unmodified `tools/screenshot` frames from
a no-input arm at 3840×2160:

| Screen | Asset |
| --- | --- |
| SEGA logo | `assets/screenshots/sonic-frontiers-sega-logo.png` |
| Cyber Space opening | `assets/screenshots/sonic-frontiers-opening-sequence.png` |
| Sonic Team logo | `assets/screenshots/sonic-frontiers-sonic-team-logo.png` |
| Middleware credits | `assets/screenshots/sonic-frontiers-middleware-credits.png` |
| Auto-save notice | `assets/screenshots/sonic-frontiers-autosave-notice.png` |
| **Title screen** | `assets/screenshots/sonic-frontiers-title-screen.png` |
| **Main menu** | `assets/screenshots/sonic-frontiers-main-menu.png` |

## What reaching rung 2 took: one unregistered NID

The title stalled after `raw/ui/ui_gamemodeinitialize.pac` for four investigation sessions, with the
symptom migrating from "the guest stopped submitting" to "prosper stopped publishing" to "the guest
composites nothing over an empty scene". The proximate cause was upstream of all three:

**`sceSaveDataTransferringMountPs4` (`RjMlsR8EXrw`, `libSceSaveData`) was not registered**, so it
reached `prosper_on_unimpl`'s `return 0` — which for this contract *is* `SCE_OK`. That is the FALSE
SUCCESS class (#2081). Frontiers zeroes a 32-byte mount-point result, calls this to look for a PS4
save to import (the main menu's own "Carry over from PlayStation®4" entry), is told the mount
succeeded, then formats `"<mountPoint>/gamedata"` out of the still-empty result and opens
**`/gamedata`** at filesystem root. That open fails `ENOENT`; the title retries once per frame,
forever, and `GameModeInitialize` never hands off to `GameModeTitle`.

Fixed by answering from local inventory: prosper has no PS4 save-data area and no local dump carries
one, so the honest answer is `SCE_SAVE_DATA_ERROR_NOT_FOUND` with the result untouched — the same
answer its already-registered sibling `sceSaveDataTransferringMount` (`WAzWTZm1H+I`) gives, whose
comment records the *identical* downstream signature on Dragon Quest VII (a `/GameSaveData245.dat`
open at filesystem root). Two titles, two sibling NIDs, one defect.

### The measured A/B

Two 60 s CPU-only `boot_trace` arms, same host, same session, binaries differing by exactly this
commit (`PROSPER_NO_COMPUTE=1 PROSPER_FILELOG=1 PROSPER_PROGRESS=5 PROSPER_PROGRESS_UNIMPL=1`):

| | before | after |
| --- | --- | --- |
| `/gamedata` open attempts | ~1,450 (every frame to the end of the arm) | **0** |
| dispatcher hits on `RjMlsR8EXrw` | 1,319 | **0** (registered) |
| distinct unimplemented NIDs | 4 | 3 (the `libSceJson2` trio, #1967) |
| last content opened | `/gamedata` (ENOENT, forever) | `ui_gamemodetitle_en.pac`, `bgm.awb` |
| `draws_cum` at t=60 s | 53,459 | **94,842** |

## Reproduction — title screen only (no input)

```bash
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
timeout --foreground -s TERM -k 5s 340s \
  ./build/screenshot <DUMP_ROOT>/PPSA03831-app0 \
  --seconds 20 --count 15 --timeout 320 --out ~/frontiers-work/shots \
  --manifest ~/frontiers-work/manifest.json
```

**Sample past 50 s.** The renderer publishes nothing for the first ~946 present callbacks and a
*successful* frame prints no log line, so a short arm reads as rung 0 (instrument trap 87). The
title screen lands around **t = 200 s** and the main menu around **t = 220 s** on this host.

For anything that does not need pixels, a CPU-only `boot_trace` arm with `PROSPER_NO_COMPUTE=1
PROSPER_FILELOG=1` reaches `ui_gamemodetitle_en.pac` in about 25 s and is the fast iteration loop —
the guest's file sequence is the progression oracle for this title.

**Save state changes the route.** The mounted `/savedata0` area is `PROSPER_SAVE0` — **not**
`PROSPER_SAVEDATA_DIR`, which controls only the save-data-*memory* store. Frontiers writes
`option/`, `arcade/` and `challenge/` there on first boot and reads them on every later boot, so a
run inherits whatever an earlier run left. Since #2734 that root is namespaced by title id, so it is
no longer shared with *other titles* — but it is still shared with every earlier run of Frontiers
itself, including a peer lane's. Pass `PROSPER_SAVE0=<private dir>` for a first-boot arm; both routes
reach the title screen (measured), but they are not the same route.

### The errno is `CONFIDENCE: MED`, and two other titles disagree with the guess — a lead, not a finding

Frontiers gates on the **sign** alone (`test eax,eax; js` at all five call sites), so `NOT_FOUND`
(`0x809F0008`) is sufficient for everything this title does and the rung-2 result does not rest on
it. But the corpus says something about the *precise* code that nobody should have to re-derive:

**Two independent titles have a dedicated arm for `0x809F000F` and none for `0x809F0008`.** Of the
five local titles that call `sceSaveDataTransferringMountPs4`, three const-compare the result —
PPSA03839 against `0x809F0003` (a retry loop), and **PPSA07809 and PPSA08804 against `0x809F000F`**.
PPSA08804's compare is inside its error arm *past* the branch, at `+0x4e41a32`. When this was
written that made it invisible to `nid_gate_scan`, which stopped at the first branch and bucketed the
site as a plain non-zero test; the scan now follows both arms and reports it as `const` (PR #2637),
so it no longer has to be taken on the hand-read. `--no-follow-arms` reproduces the older reading.

`0x809F000F` appears nowhere in prosper, and the PS5 3.20 library dump carries names and NIDs only —
no constants — so what it means is unresolved. Start from this rather than from
`0x809F0008` if a title ever turns out to need the exact code. (Review of PR #2208.)

## Known defects on the title screen

- The title-screen heading renders the string **"Try Again"** where the SONIC FRONTIERS logo
  belongs, in large blue type. The surrounding menu strings are correct and legible ("New Game",
  "Language", "Carry over from PlayStation®4", "Copyright", "User manual", "Extras"), and the
  version string `1.41` draws, so this is a wrong string/asset selection rather than a text-render
  failure.
- Shortly after the main menu a full-width panel with a blue header band slides over the title
  screen and stays. Its body renders almost no text — a handful of glyph marks and a diamond
  cursor — over a correct SurfRide background. No input was driven in either arm.

Both are filed as [#2206](https://github.com/mattias800/prosper/issues/2206).

## The Cyber Space stage's compute frontier (2026-08-21)

The route on [PR #2791](https://github.com/mattias800/prosper/pull/2791) reaches `GameModeStage` on
Cyber Space `w6d01`; the HUD composites and the world behind it is black. `#2790` measured the cause
as compute programs that never execute, and the census is the instrument:

```bash
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_COMPUTE_TRANSLATE_ONLY=1 \
PROSPER_COMPUTE_PROGRAM_CENSUS=1 \
PROSPER_DBG_PROGRAM=0x2005714000,0x2005717e00,0x200571bd00 \
PROSPER_PAD_SCRIPT=@<route>/reach-gameplay.pad PROSPER_SAVE0=<private dir> \
  ./build/boot_trace <DUMP_ROOT>/PPSA03831-app0
```

**Guest program addresses are run-local — re-derive them from the `[compute-census]` per-program
lines of the run you are in, not from this document.** And **read the program count on the census
line first**: this route desynchronises when the host is busy, and an arm that stayed in the menus
reports `over 5 program(s)` at the same denominator as one that reached the stage reports
`over 30 program(s)`. Nothing else in the block distinguishes them. Check `uptime` first — on a
shared machine the load that breaks the route is not yours and is not visible from the arm. `PROSPER_DBG_PROGRAM` then narrows the
verbose recompiler stream to those addresses; `PROSPER_DBG=1` is ~1.5 GB here and desyncs the pad
script before it reaches the stage.

At `262144 dispatch decisions over 32 program(s)`, **14 programs are listed and every one is
`executed=0`** (the census prints per-program detail only for programs that skipped at least once,
#2745). Three of them dispatch at the display width — `240x135x1` groups of `16x1`, `16x2` and
`16x3`, i.e. 3840x135, 3840x270 and 3840x405 threads — and are the screen-space passes the world
depends on.

**All three reach the CFG dispatcher's body.** #2790's handoff named two levers,
`s_cbranch_execz` and `image_load_mip`; both were reject PCs printed by the *straight-line* emitter,
which these programs only reach after two earlier routes decline. The first decline is ordinary
route selection (`backward else`, `role=route-decline`); the second was a real defect in the Wave64
MUST dataflow, fixed by adding V_LSHL_ADD_U32 to `scalar_alu_source_words`' B32 list. With that in
place the `wave64-ambiguous-mask-read` decline does not occur anywhere in the run, all three programs
reach the CFG dispatcher's body, and all three stop at:

```text
[mimg-mip] program=0x2005714000 image_load_mip declined pc=33 shape=0 proven_zero_mip=0
           img_dim=5/1 samples=1 mips=12 mip_tail=0 compressed=0 array_in_gfx=0
           addr=0x2026900000 2048x2048x1 fmt=1 tile=27 dmask=0x3 unorm=0 glc=0 layer_stride=0
```

`IMAGE_LOAD_MIP` where the *resource* declares `2D_ARRAY` with a **12-level mip chain** while the
*instruction* addresses it `dim:2D`, and the mip operand is not one of the recognised provably-zero
shapes. `rdna2_emit_alu.cpp` only ever specialises this op away after proving the mip is zero and
the resource is single-level; a 12-mip resource has no such proof, so it declines. Tracked as
[#2818](https://github.com/mattias800/prosper/issues/2818).

### `image_load_mip` is the FIRST blocker of at least two, not the only one (2026-08-21)

**This section used to say "All three now block on exactly one instruction", and #2818's summary
still says `IMAGE_LOAD_MIP` is "the single remaining blocker for all three". That is falsified.**
Behind the MIMG decline, all three programs stop again on **`s_getpc_b64`**.

The A/B is one variable in one binary: a **measurement-only** build (never merged, and it must never
be) that accepts the declining `IMAGE_LOAD_MIP` as an ordinary LOD-0 `OpImageFetch`, discarding the
guest's mip operand. That result is knowingly WRONG whenever the guest asks for a level other than
zero — it exists only to answer "is anything behind this reject", and it is the cheapest instrument
that can answer it, because the recompiler aborts at its first fatal site and therefore reports
exactly one.

| program | reject on master | reject with `IMAGE_LOAD_MIP` accepted |
| --- | --- | --- |
| `0x2005714000` | `pc=33 words=f0040308 fmt=14 op=0x1` (`image_load_mip`) | `pc=517 words=be801f00 fmt=1 op=0x1f` (`s_getpc_b64`) |
| `0x2005717e00` | `pc=81 words=f0040308 fmt=14 op=0x1` | `pc=527 words=be801f00 fmt=1 op=0x1f` |
| `0x200571bd00` | `pc=81 words=f0040308 fmt=14 op=0x1` | `pc=527 words=be801f00 fmt=1 op=0x1f` |

**The census does not move**: 13 programs listed and every one `executed=0`, at
`65536 dispatch decisions over 30 program(s)`, in both arms, with the three scene-width programs'
skip counts unchanged (39 / 40 / 335 against 39 / 40 / 317). So this is the fourth measured instance
of "a decline was cleared and nothing about the frame or the census changed", and the first where
the reason is visibly that another decline was waiting one route further in.

`s_getpc_b64` is rejected by `rdna2_emit_alu.cpp:1062` unless the PC-relative embedded-table
pre-pass (`detect_pcrel_tables`, `rdna2_cfg_support.hpp`) folded a table load from this shader — the
same family as R-Type Delta's #2783. **That half is now fixed** ([#2859](https://github.com/mattias800/prosper/issues/2859)):
the pre-pass recognised an untyped (`MUBUF`) and a scalar (`SMEM`) consumer but not a **typed**
one, and Frontiers' consumer is `tbuffer_load_format_x v10, v10, s[0:3], 0 offen` at BUF_FMT 22
(`32_FLOAT`) — a one-component 32-bit typed format, which performs no conversion at all, so the
existing constant-lookup fold was already exactly right for it and only the `!is_format` guard stood
in the way.

### The blocker list for these three programs is exactly two, and one of them is now closed

`tools/shader_inspect` on the `PROSPER_SHADER_DUMP` bytes answers "how many blockers" directly, and
nobody had pointed it at these programs. Its generic-coverage enumeration lists every instruction
the per-instruction emitter refuses:

```text
$ shader_inspect exec_cs_2005714000.bin
generic-coverage total=437 alu=420 exp=0 table=3 unsupported=14 first=MIMG/0x1
generic-unsupported pc=0033 fmt=MIMG op=0x1        <- image_load_mip (#2818)
generic-unsupported pc=0344 fmt=SOPP op=0x8        <- s_cbranch_execz, lowered by the CFG dispatcher
   ... eleven more SOPP branches ...
generic-unsupported pc=0517 fmt=SOP1 op=0x1f       <- s_getpc_b64 (#2859)
generic-unsupported pc=0536 fmt=SOP1 op=0x1f
```

All three programs give the same two non-branch families and nothing else. **Read that enumeration
before costing out any single reject** — it is a static per-instruction pass, so it over-reports
(the `SOPP` branches are fine, and it does not run `detect_pcrel_tables`, so it still lists a
`s_getpc_b64` whose table the real pipeline folds), but it bounds the problem from above, which one
terminal reject line can never do.

And the pair is now measured to be the *whole* list. With #2859's fold in place **and** the same
measurement-only LOD-0 build, in a routed `boot_trace` arm at
`65536 dispatch decisions over 30 program(s)`, all three programs **disappear from the census skip
list entirely** — 13 listed before, 10 after, and the three that left are exactly
`0x2005714000` / `0x2005717e00` / `0x200571bd00`. The census lists only programs that skipped at
least once, so leaving it means they recompiled and executed.

So `IMAGE_LOAD_MIP` really is the last blocker for these three now — which is what #2818 claimed
before it was true, and for a different reason.

**How to rebuild the measurement arm** (it is deliberately not in the tree — it renders wrong
content whenever the guest asks for a level other than zero, and a screenshot from it would be
believed): in `rdna2_emit_alu.cpp`, immediately before the `is_zero_mip_load && (...)` decline,
accept the instruction when an env switch is set and `res->sample_count == 1 &&
!res->compression_enabled`. Execution then falls through to the ordinary `image_fetch_2d` at LOD 0.
Run it only under `PROSPER_COMPUTE_TRANSLATE_ONLY=1`, where nothing is submitted.

**The generalisable rule, and it cost this lane its first two hours: a terminal reject line is a
lower bound of one.** It names the site the recompiler stopped at, which is the first fatal one on
whatever route it took — never how many more are behind it. Before costing out a fix for a named
blocker, spend one arm proving there is nothing behind it.

### Where the guest's mip levels actually are

The `[mimg-mip]` line now carries the resource's identity, and with `PROSPER_TDUMP=1` the picture is
unambiguous. The declining resource is **2048x2048, `IMG_FMT 64` (32_32_FLOAT -> RG32F),
`type=13` (2D_ARRAY) with `depth=1`, `tile_mode=27` (SW_64KB_R_X), `BASE_LEVEL=0`, `LAST_LEVEL=11`,
`MAX_MIP=11`** — a complete 12-level pyramid of a 2048x2048 two-channel float surface, which is
exactly 12 levels for a 2048-wide chain.

And the guest binds **thirteen** descriptors to one such allocation:

```text
[tdump] t=2025e500 c4000000 01ffc1ff d1b0022c 00000000 007000b0 ... mips=0..0  max_mip=11 depth=1
[tdump] t=2025e500 c4000000 01ffc1ff d1b1122c 00000000 007000b0 ... mips=1..1  max_mip=11 depth=1
        ... one per level ...
[tdump] t=2025e500 c4000000 01ffc1ff d1bbb22c 00000000 007000b0 ... mips=11..11 max_mip=11 depth=1
[tdump] t=2025e500 c4000000 01ffc1ff d1bb022c 00000000 007000b0 ... mips=0..11 max_mip=11 depth=1
```

Twelve **single-level** views, one per level, plus one **whole-chain** view. That is a pyramid the
guest builds itself: each level is written through its own single-level descriptor and the finished
pyramid is read back through the chain descriptor at a runtime LOD.

**So the guest's mip levels are ordinary guest memory**, at the byte offsets
`tiled_mip_level_layout` already computes — and prosper's existing single-level path places each of
them correctly *today*, one descriptor at a time (`image_base_level_view` applies `BASE_LEVEL`'s
`mip_offset`, and SW_64KB_R_X is one of the tile modes it supports, tail included). What prosper
cannot do is view them together. That matters for whoever takes #2818, because it says the faithful
implementation **uploads real guest bytes** rather than synthesizing levels, and the machinery to
locate each one is already written and already exercised by the guest's own twelve descriptors.

**What the renderer can and cannot do with mips today**, because getting this wrong sends the next
investigation to the wrong file:

* The **graphics** path does build chains, gated to plain-2D (`img_dim == 1`), depth-1, non-storage,
  non-RTT, RGBA8 sampled textures declaring `declared_mip_levels > 1`
  (`tests/fixtures/render_runner.h:6044-6055` — that file is the live offscreen Vulkan backend,
  included by `frontends/shared/live/live_renderer.cpp:39`). `tests/gpu/test_texture_mip_render.cpp`
  is a registered ctest asserting a declared 3-level chain samples level 2.
* The **compute** path does not: its single `VkImageCreateInfo` for guest images sets
  `ici.mipLevels = 1` unconditionally (`frontends/shared/live/live_compute.cpp:7089`).
* **And the graphics chain is GENERATED, not uploaded.** Staging carries level 0 only; levels
  1..N-1 come from a linear-filtered `vkCmdBlitImage` cascade at upload time
  (`render_runner.h:6257-6259`, `:7245`).

Frontiers' resource misses that gate on two counts at once — it is `img_dim=5` (2D_ARRAY) and it is
a compute binding. But the third bullet is the one that matters most for whoever takes #2818:
widening the gate would produce levels **synthesized by downsampling level 0**, not the guest's own
mip data. That would render, and it would be wrong, in the exact way the "no plausible constants"
rule exists to prevent.

**The census did not move when the MUST defect was fixed** — 14 programs, all `executed=0`, before
and after at the same denominator — and neither did the composite. Both were measured, not assumed.

## Ruled out

One line per falsified hypothesis with the evidence that killed it. Read this before forming a new
one. Twelve rows were established on #1968 / #2023 and are copied here so they survive those issues
being closed; the rest are this document's own. **Do not restate the row count in this paragraph** —
a stated total is stale as soon as the next lane appends, and every lane that adds a row would have
to remember to update it. The last one did not (review of #2820).

| Hypothesis | Verdict and evidence |
| --- | --- |
| The stall after `ui_gamemodeinitialize.pac` is a rendering, present or publish defect | **Falsified.** It was one unregistered Sony import. With `RjMlsR8EXrw` registered and nothing else changed, the same binary reaches the title screen and the main menu — while the renderer, present path and publish gate are byte-identical. (This PR.) |
| The guest stops submitting GPU work after the opening logo | **Falsified.** A same-process A/B across the freeze recorded 11 `agc_driver_submit_dcb`, 4 draws, 7 dispatches and 1 flip per frame *after* it, against 15/1/3/1 during the intro. (#1968.) |
| The SONIC TEAM logo movie never signals completion and UI init waits on it | **Falsified.** `PROSPER_DENY_SUBSTR=.usm` makes the open fail and the guest reaches the *identical* terminal state — 177 file opens against the control's 178, the difference being exactly the denied movie. (#1968.) |
| The stall is prosper's Videodec2 HLE, or the AvPlayer consumer-driven-EOF defect (#1973) | **Falsified — neither path is reachable here.** The eboot's 39 `DT_NEEDED` entries name no `libSceVideodec2`, `libSceAvPlayer` or `libSceAjm`, and neither do the three `sce_module/` PRXs; decode is CRI Sofdec2, statically linked. (#1968.) |
| Some guest thread is blocked on an unposted wait | **Falsified.** `guest_bt --all` over all 60 guest threads found every one parked in an ordinary idle wait, with the main thread in the engine's own frame limiter. (#1968.) |
| The guest polls a Sony service that never changes / waits on a Sony answer | **Falsified at the service layer.** A 154-handler sweep over a 15-flip window recorded exactly four calls per frame: `sceUserServiceGetEvent`, `sceSystemServiceGetStatus`, `sceSaveDataUmount2`, `agc_dcb_set_flip`. The wait was not a *poll* — it was an ENOENT retry the service layer cannot see. (#2023.) |
| The guest is blocked on an **unimplemented** NID | **Falsified as stated, and it was the right question asked with a blind instrument.** The `PROSPER_PROGRESS_UNIMPL` arm that returned "3 distinct, 5 calls, none polled" ran a route on which the guest had **no save data**, and Frontiers only reaches the PS4-transfer probe after reading `/savedata0/optiondata`. Re-run against a populated save area, the same instrument reports a fourth NID called **1,319 times**. A per-title save area is part of the route (see *Reproduction*); a census taken on one route does not bound the other. (#2023, this PR.) |
| #657's skipped `64x64x6` layered-image dispatch contributes | **Falsified.** It fires exactly twice, ~20,000 submits before the wall, and never again. Still a real gap; not this. |
| #1967 (`libSceJson2`) is the proximate cause | **Falsified.** Three one-shot calls early in boot; no json2 handler is entered during the stalled window. Still a real latent defect. |
| The declined ordered-DMA submit of #1982 | **Falsified.** `grep -cE 'ordered DMA submit (rejected\|not executed)'` = 0 across every arm of this title. |
| Same defect as Little Nightmares III (#1962) | **Falsified — the opposite shape.** Here `present_count` climbed while `frame_seq` froze and submits kept arriving; there both froze together. |
| `pixel_crc32=666f7b3f` links this to #1962 / #1982 | **No — it is just "black 3840×2160"** and recurs on unrelated titles. Never group by a black-frame hash. |
| The frame going black shortly *before* the publish wall shares the wall's cause | **Falsified.** With the publish wall removed (#1990) the black survived unchanged; the last publishable frame was already black. Two defects. |
| A no-input arm sits on the title screen because prosper stalls, or because the menu is unreachable | **Falsified.** A twelve-page modal notice queue is open over the menu. Twelve confirms clear it and the thirteenth activates a menu entry; the same binary then reaches `GameModeStage`. Nothing in the emulator was changed. (This document.) |
| The panel over the menu "renders almost none of its text" (#2206) | **Incomplete rather than wrong.** With any face button pressed, the same panel renders its header *and* body correctly — "Extras: The acquired additional content will be accessible from the Extras menu.", "Update Details: The following content has been added in the update: -Action Chain Challenge -New Koco -Birthday Decorations -Status map -New Game+". The blank panel is the *no-input* state of a queue nobody had advanced. |
| The title heading permanently draws "Try Again" instead of the logo (#2206) | **State-dependent, not permanent.** With the notice queue cleared, the SONIC FRONTIERS logo renders correctly at 3840x2160. In the same frames the six main-menu entries do *not* render their text, while the original rung-2 capture rendered the entries and got the heading wrong — the two are anti-correlated, which points at string/element resolution rather than at the text renderer. |
| The d-pad can drive the boot notice queue | **Falsified.** A twelve-button sweep advanced the panel on every face button and on none of `up`, `down`, `left`, `right`. A route that used directional windows there would silently deliver nothing. |
| Pressing confirm often enough clears the notice queue | **Falsified.** 405 confirms at 20-flip spacing reached exactly the same state as 6 did; 12 confirms at 40-flip spacing cleared it. Presses landing inside a page's transition animation are discarded, so spacing decides the outcome and volume does not. |
| The black world in the stage is an artifact of `--warmup-seconds` skipping the renderer past the stage's setup | **Falsified by a control arm.** With `--warmup-seconds 90` the renderer is live continuously from flip 1517, before `GameModeStage` loads at flip ~2900, and the same HUD-over-black frames appear from flip 3429 to 4325 with the same 2880-wide composited rect. (#2790.) |
| The black world is a compositing/present defect | **Falsified as the primary cause.** Sixteen of the stage's thirty-two compute programs have `executed=0`, three of them 2880-thread-wide screen-space passes; and the same build composites the in-engine cutscene correctly at full width. The composite is downstream of a scene target that was never shaded. (#2790.) |
| The three scene-target-width stage programs are blocked by `s_cbranch_execz` and `image_load_mip`, the encodings their reject lines name | **Half falsified.** Both were reported by the straight-line emitter, two routes downstream of the decline that mattered. Live, with `PROSPER_DBG_PROGRAM` on each address, the CFG dispatcher declined `wave64-ambiguous-mask-read` at pc481 / pc481 / pc471 — one missing entry in `scalar_alu_source_words`, which charged `v_lshl_add_u32 v7, v6, 2, vcc_lo` (a 32-bit read of VCC_LO used as scalar scratch) the whole VCC pair. With it fixed that decline occurs **0** times, all three reach the dispatcher body, and all three converge on `image_load_mip`. **`s_cbranch_execz` is dead as a lever here** — the dispatcher lowers it fine. See `RECOMPILER_REMAINING.md` § Ruled out. (This row said "`image_load_mip` **alone**"; the row below falsifies the *alone*, not the rest of it — `s_getpc_b64` is waiting behind the MIMG site.) |
| Fixing a recompiler decline that unblocks these programs will change the frame | **Not established, and twice now it has not.** #2758 took `executed=0` to `executed=6` with no image change; #2801 cleared five SCC-site declines with no image change; this change cleared three dispatcher declines with **no census change at all** (14 listed, all `executed=0`, both arms at `262144 dispatch decisions over 32 program(s)`) and no composite change. A fourth instance is the row below: accepting `image_load_mip` moves the reject on and changes neither the census nor the frame, because another decline is waiting behind it. Measure the composite separately — non-black percentage and bounding box — before claiming anything about the world. |
| `IMAGE_LOAD_MIP` is the single remaining blocker for the three scene-width stage programs (#2818, and this document's own earlier wording) | **Falsified by a one-variable A/B.** A measurement-only build that accepts the declining `IMAGE_LOAD_MIP` at LOD 0 moves all three programs' terminal reject from the MIMG site to **`s_getpc_b64`** (`be801f00`, SOP1 op 0x1f) at pc 517 / 527 / 527, and leaves the census byte-for-byte where it was — 13 listed, all `executed=0`, at `65536 dispatch decisions over 30 program(s)`. `image_load_mip` is the first of two blockers, and the second — the PC-relative embedded-table fold refusing a TYPED consumer — is fixed by this PR (#2859). With both cleared the three programs leave the census skip list entirely, so the pair is the complete list. **A terminal reject line is a lower bound of one**: the recompiler aborts at its first fatal site, so it can never say how many are behind it; `tools/shader_inspect`'s generic-coverage enumeration bounds it from above in one command. (#2859, this document.) |
| A live `tools/screenshot` arm may use `--warmup-seconds` to reach the stage faster on this route | **Falsified in two arms, and the failure is silent.** `--warmup-seconds` suppresses Vulkan rendering, which raises the guest's flip rate to ~17-18/s (6,994 flips in 420 s on one arm, 1,669 in 90 s on another) against ~3.2-3.5/s once the renderer is live — but the boot's own progression is paced by asset loading and movie playback in WALL CLOCK, not by flips, so a flip-anchored route fires its windows against a guest that is nowhere near the state they were authored for. At `--warmup-seconds 90` the five `up` presses at f1700-f1940 were delivered ~50 s BEFORE the main menu existed, and the arm then activated "Extras" with the f2100 confirm and sat in the notices list for the rest of the run (open the frames: sample 7 is the title screen with the cursor still on "Extras", sample 24 is the "Update Details" notice page). At `--warmup-seconds 420` the same route produced **40 identical all-black 3840x2160 frames, one distinct frame in 3,752 publications, and 5 compute programs seen** — which reads exactly like "this title renders nothing", and is apparatus. Run the live arm with **no warmup**; the CPU-only `boot_trace` arm is the fast loop and it does reach the stage. (This document.) |
| A routed `boot_trace` arm on this title reaches `GameModeStage` reliably, so its census can be read without checking | **Falsified — it desynchronises under HOST LOAD, and the failure is silent.** Measured across nine routed arms on one host, same route: the ones taken while the machine was quiet reached the stage (`65536 dispatch decisions over 30 program(s)`); the ones taken while other agents were building and linking on the same box did not (`over 5 program(s)` at the same denominator, having sat in the menus for the whole run). `uptime` read a load average of **28** during the failing streak. This is the same mechanism as the `--warmup-seconds` row above — the route's windows are anchored on display flips while the boot's own progression is paced by asset loading and movie playback in wall clock, so anything that moves the flip rate relative to wall clock moves the inputs off their targets. On a shared machine that is **not observable from inside the arm**, so: check `uptime` before starting, and **read the program count on the census line before believing anything in the block.** 5 means the arm never left the menus and every number under it is about a different part of the game; 30 means it is in the stage. A census quoted without that number is worthless. (This document.) |
