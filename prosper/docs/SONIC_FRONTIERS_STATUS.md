# Sonic Frontiers (`PPSA03831`) — status

Tracker: [#1891](https://github.com/mattias800/prosper/issues/1891). Engine: **Hedgehog Engine 2
(Needle renderer)**, SurfRide UI, statically linked CRIWARE middleware — determined from the eboot's
own embedded shader source paths (`Library\hedgehog\…`, `Library\needle\…`) and the dump's
`NeedleShader.pac` / `raw/hedgehog/` asset trees, not assumed from the publisher. *Sonic Origins*
(#1871) and *Sonic Racing: CrossWorlds* (#1895) share parts of the same stack.

## Current rung — 2 (title screen reached and rendered)

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

## Reproduction

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

**Save state changes the route.** The mounted `/savedata0` area is `PROSPER_SAVE0` (default
`/tmp/prosper-savedata0`, *shared by every lane and every title on the box*) — **not**
`PROSPER_SAVEDATA_DIR`, which controls only the save-data-*memory* store. Frontiers writes
`option/`, `arcade/` and `challenge/` there on first boot and reads them on every later boot, so a
run inherits whatever a peer's run left behind. Pass `PROSPER_SAVE0=<private dir>` for a
first-boot arm; both routes reach the title screen (measured), but they are not the same route.

## Known defects at rung 2

- The title-screen heading renders the string **"Try Again"** where the SONIC FRONTIERS logo
  belongs, in large blue type. The surrounding menu strings are correct and legible ("New Game",
  "Language", "Carry over from PlayStation®4", "Copyright", "User manual", "Extras"), and the
  version string `1.41` draws, so this is a wrong string/asset selection rather than a text-render
  failure.
- Shortly after the main menu a full-width panel with a blue header band slides over the title
  screen and stays. Its body renders almost no text — a handful of glyph marks and a diamond
  cursor — over a correct SurfRide background. No input was driven in either arm.

Both are filed as [#2206](https://github.com/mattias800/prosper/issues/2206).

## Ruled out

One line per falsified hypothesis, with the evidence that killed it. Read this before forming a new
one. The first row is this document's own; the rest were established on #1968 / #2023 and are copied
here so they survive those issues being closed.

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
