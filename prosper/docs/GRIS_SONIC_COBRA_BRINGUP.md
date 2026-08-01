# GRIS, Sonic Origins, and Space Adventure Cobra bring-up

Validated on Linux on 2026-07-25, with GRIS opening gameplay and Cobra tutorial combat revalidated
on 2026-07-31. This note records the reproducible evidence for issue #1356. Raw PCM, verbose logs,
and GPU captures are local diagnostics and are intentionally not committed.

## Result matrix

| Title | Revision | Visual milestone | Audio evidence |
| --- | --- | --- | --- |
| GRIS (`PPSA09804`) | 01.001.000 | Native 1920×1080 opening gameplay with scripted movement | CLEAN on current master over the first 35 seconds: `rms=0.0082`, `peak=0.1173`, duplicated grains 0.0% |
| Sonic Origins (`PPSA05325`) | 02.002.000 update targeting 02.001.000 | Blocked: update-only dump is missing two base title assets | AudioOut2 port 17 runs, but guest PCM is zero while startup is blocked |
| Space Adventure Cobra — The Awakening (`PPSA17337`) | 01.004.000 | Native 1920×1080 tutorial combat with scripted progression | CLEAN, `rms=0.0436`, `peak=0.1880`, duplicated grains 0.0% |

## Visual evidence

### GRIS

![GRIS — New Game title](screenshots/issue-1356-gris-title.png)

Route: `scripts/gris/reach-title-screen.pad`. The title appears without input; the comments-only
route keeps a neutral scripted controller connected and prevents an evidence run from selecting
**NEW GAME**.

![GRIS — opening gameplay](../../assets/screenshots/gris.png)

Route: `scripts/gris/reach-first-gameplay.pad`. Poll-safe Cross edges select the default New Game
entry and cover the opening sequence's timing variation. Right is held from 78 through 150 seconds.
On exact master `2562269711f89b59f7f3038eab1bb4dcf8468b52`, the direct native `screenshot` route
reached the controllable ink-ground scene around 130 seconds: the character moved and animated under
the scripted input, then settled after the route returned to neutral. The retained 170-second run
contained 85 source-distinct and 85 pixel-distinct unmodified 1920×1080 frontend frames.

### Space Adventure Cobra — The Awakening

![Space Adventure Cobra — The Awakening title](screenshots/issue-1356-space-adventure-cobra-title.png)

![Space Adventure Cobra — tutorial combat](../../assets/screenshots/space-adventure-cobra.png)

Route: `scripts/cobra/reach-title-or-gameplay.pad`.

The route combines early flip-anchored Cross pulses with wall-clock hold-Square actions for the
opening movies and later Cross pulses for dialogue. A fresh-save, default-configuration run observed
Square at 60, 70, 80, 90, and 100 seconds, reached the readable `Press R2 to shoot with the
Psychogun.` tutorial at 85 seconds, and continued through full-color combat for the complete
180-second bound. All 36 captures came from distinct composited source frames; 33 were pixel-distinct.

Write watches remained fully enabled in that run: 63 registrations covered 65,504 pages and handled
8,447 write faults without a worker crash. The fault path temporarily restores host `%fs` around the
host mutex and `mprotect`, then restores guest `%fs` before resuming the interrupted store. A
production-handler regression test locks that TLS boundary. The normal screenshot frontend produced
the committed images at 1920×1080 with no debug shader, resource override, render scaling, sparse
rendering, warm-up shortcut, or write-watch escape hatch.

## Reproduction

Run from the repository root and point each command at a legally obtained app directory:

```bash
PROSPER_PAD_SCRIPT=@prosper/scripts/gris/reach-title-screen.pad \
  prosper/build-linux/screenshot /path/PPSA09804-app0 \
  --seconds 1 --count 35 --timeout 90 --out "$HOME/prosper-artifacts/gris-shots"

PROSPER_PAD_SCRIPT=@prosper/scripts/gris/reach-first-gameplay.pad \
  prosper/build-linux/screenshot /path/PPSA09804-app0 \
  --seconds 2 --count 85 --timeout 210 --out "$HOME/prosper-artifacts/gris-gameplay-shots"

PROSPER_SAVEDATA_DIR="$HOME/prosper-artifacts/cobra-savedata" \
PROSPER_SAVE0="$HOME/prosper-artifacts/cobra-save0" \
PROSPER_PAD_SCRIPT=@prosper/scripts/cobra/reach-title-or-gameplay.pad \
  prosper/build-linux/screenshot /path/PPSA17337-app0 \
  --seconds 5 --count 36 --timeout 195 --out "$HOME/prosper-artifacts/cobra-shots"
```

Capture final mixed PCM with `PROSPER_AUDIO_DUMP="$HOME/prosper-artifacts/<title>"`. GRIS's title is silent, so use
`scripts/gris/reach-first-gameplay.pad` for its audio exercise; the title-screen route remains the
visual evidence route. Analyze the active port using its logged format; GRIS uses stereo float32 on
port 1 (its auxiliary port 3 remained silent):

```bash
python3 prosper/tools/audio_analyze.py "$HOME/prosper-artifacts/gris.port1.raw" \
  --fmt f32 --channels 2 --rate 48000 --tail-seconds 30
```

The 2026-07-25 validated GRIS verdict was:

```text
CLEAN: corr(block 1024f)=+0.095 neighbor-max=+0.047 spike=+0.048 (threshold 0.35) dup-grains=0.0% rms=0.1800 peak=1.2689
```

The 2026-07-31 exact-master gameplay rerun opened the same stereo float32 port 1 and auxiliary
stereo s16 port 3. Port 1's first 35 seconds were non-zero and passed the repetition check:

```text
CLEAN: corr(block 1024f)=+0.008 neighbor-max=+0.012 spike=-0.004 (threshold 0.35) dup-grains=0.0% rms=0.0082 peak=0.1173
```

The final 30 seconds of the 170-second route were silent and therefore produced no repetition
verdict. That quiet tail is not evidence that the complete capture or output path was silent.

The validated Cobra verdict was:

```text
CLEAN: corr(block 1024f)=-0.049 neighbor-max=+0.032 spike=-0.081 (threshold 0.35) dup-grains=0.0% rms=0.0436 peak=0.1880
```

## Ruled out

One line per dead hypothesis, the evidence that killed it, and where that evidence lives.

| Hypothesis | Verdict and evidence | Source |
|---|---|---|
| Sonic Origins' black startup loop is a prosper defect | **Falsified — it is an incomplete dump.** The supplied 02.002.000 directory is an *update* image targeting 02.001.000, not a merged base+update app. Live file tracing shows the only unresolved startup requests are `raw/ui/ui_startup.pac` and `raw/ui/rpl_texture/ui_title_nocopy.dds`, both absent. Everything else initializes — renderer, connected pad, CRI sound banks, AudioOut2 pump, 300+ flips in 100 s with no guest fault. Correct behaviour without those assets is a black loop and silence. **No compatibility milestone is claimed until a complete dump exists.** | this doc, `COMPATIBILITY.md` |
| A PS5 `launchActivity` Game Intent routes around the missing UI assets | **Falsified.** The update does declare `launchActivity` support and ship the classic RSDK files, and the guest genuinely receives and consumes an exact `TITLE_SONIC_1_CLASSIC` intent — its `activityId` property is read and recognized. It still requests both missing UI files before it will open `raw/retro/Sonic1u.rsdk`. The activity experiment narrows the blocker; it does not bypass it. Truthful default no-intent behaviour is preserved. | this doc |

## Sonic Origins dump audit

Sonic reaches a stable frontend frame loop with all decoded GPU operations realized, a connected
scripted controller, CRI Atom banks loaded, and AudioOut2 advancing. A 100-second `boot_trace` run
observed more than 300 flips and exercised the full route without a guest fault. It cannot reach an
authentic title screen with the supplied files.

The dump identifies itself as:

```text
contentVersion:       02.002.000
targetContentVersion: 02.001.000
originContentVersion: 01.000.000
```

With `PROSPER_FILELOG=1`, the complete unresolved-path set is:

```text
3 /app0/raw/ui/ui_startup.pac
3 /app0/raw/ui/rpl_texture/ui_title_nocopy.dds
```

Neither file exists anywhere under `PPSA05325-app0`. After those failures the game publishes black
scanouts and its correctly initialized AudioOut2 buffers remain zero. This is a content-integrity
blocker, not evidence of title compatibility. Resume the visual and audio validation with a merged
base+02.002.000 dump containing both files; do not alias another PAC/DDS or use a black frame as a
success screenshot.

### Game Intent activity audit

The update does contain the four classic RSDK data files, so an authentic PS5 activity launch was
also tested rather than assuming the normal menu was the only route. `sce_sys/param.json` permits the
standard `launchActivity` intent. Guest disassembly independently shows that Sonic reads its
`activityId` property and recognizes `TITLE_SONIC_1_CLASSIC` as its Sonic 1 Classic boot selection.

Prosper now models that shell action with `PROSPER_GAME_INTENT_ACTIVITY_ID`: it advertises one pending
System Service event, delivers event `0x10000017` once, and implements the Game Intent receive,
property, and terminate contracts. With the exact activity selected, the guest trace proves that
Sonic consumes all three relevant calls:

```text
sceSystemServiceReceiveEvent
sceNpGameIntentReceiveIntent
sceNpGameIntentGetPropertyValueString(..., "activityId", ..., 0x21)
```

That route still requests `ui_startup.pac` and `ui_title_nocopy.dds` before it can transfer control to
the classic runtime. A 30-second filtered trace never opens `raw/retro/Sonic1u.rsdk`; a 44-second
native capture remains black after frame 1, and the active stereo float32 48 kHz port 17 capture is
silent (`rms=0`). The activity experiment therefore narrows the blocker but does not change the
compatibility result. Reproduce it with:

```bash
PROSPER_GAME_INTENT_ACTIVITY_ID=TITLE_SONIC_1_CLASSIC \
PROSPER_PAD_SCRIPT=@prosper/scripts/sonic/reach-title-or-gameplay.pad \
  prosper/build-linux/screenshot /path/PPSA05325-app0 \
  --seconds 1 --count 60 --timeout 120 --out "$HOME/prosper-artifacts/sonic-activity-shots"
```

The control is off by default; an ordinary launch continues to report no pending Game Intent.
