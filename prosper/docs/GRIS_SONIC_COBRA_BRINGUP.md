# GRIS, Sonic Origins, and Space Adventure Cobra bring-up

Validated on Linux on 2026-07-25. This note records the reproducible evidence for issue #1356.
Raw PCM, verbose logs, and GPU captures are local diagnostics and are intentionally not committed.

## Result matrix

| Title | Revision | Visual milestone | Audio evidence |
| --- | --- | --- | --- |
| GRIS (`PPSA09804`) | 01.001.000 | Native 1920×1080 **NEW GAME** title; route continues into the opening | CLEAN, `rms=0.1589`, `peak=1.2692`, duplicated grains 0.0% |
| Sonic Origins (`PPSA05325`) | 02.002.000 update targeting 02.001.000 | Blocked: update-only dump is missing two base title assets | AudioOut2 runs, but guest PCM is zero while startup is blocked |
| Space Adventure Cobra — The Awakening (`PPSA17337`) | 01.004.000 | Native 1920×1080 title | CLEAN, `rms=0.0433`, `peak=0.1880`, duplicated grains 0.0% |

## Visual evidence

### GRIS

![GRIS — New Game title](screenshots/issue-1356-gris-title.png)

Route: `scripts/gris/reach-first-gameplay.pad`.

### Space Adventure Cobra — The Awakening

![Space Adventure Cobra — The Awakening title](screenshots/issue-1356-space-adventure-cobra-title.png)

Route: `scripts/cobra/reach-title-or-gameplay.pad`.

The normal screenshot frontend produced both images through the composited present path at
1920×1080. No debug shader, resource override, render scaling, sparse rendering, or warm-up shortcut
was used for the committed visual evidence.

## Reproduction

Run from the repository root and point each command at a legally obtained app directory:

```bash
PROSPER_PAD_SCRIPT=@prosper/scripts/gris/reach-first-gameplay.pad \
  prosper/build-linux/screenshot /path/PPSA09804-app0 \
  --seconds 1 --count 90 --timeout 180 --out /tmp/gris-shots

PROSPER_PAD_SCRIPT=@prosper/scripts/cobra/reach-title-or-gameplay.pad \
  prosper/build-linux/screenshot /path/PPSA17337-app0 \
  --seconds 1 --count 90 --timeout 180 --out /tmp/cobra-shots
```

Capture final mixed PCM with `PROSPER_AUDIO_DUMP=/tmp/<title>` on the same route. Analyze the active
port using its logged format; GRIS uses stereo float32 on port 1:

```bash
python3 prosper/tools/audio_analyze.py /tmp/gris.port1.raw \
  --fmt f32 --channels 2 --rate 48000 --tail-seconds 30
```

The validated GRIS verdict was:

```text
CLEAN: corr(block 1024f)=+0.108 neighbor-max=+0.043 spike=+0.066 (threshold 0.35) dup-grains=0.0% rms=0.1589 peak=1.2692
```

The validated Cobra verdict was:

```text
CLEAN: corr(block 1024f)=-0.043 neighbor-max=+0.047 spike=-0.089 (threshold 0.35) dup-grains=0.0% rms=0.0433 peak=0.1880
```

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
