# Human-authored render snapshots ("snaps")

You play the game. You decide what is correct. The tool remembers.

This replaces the scripted-route guards in `snapshot.py`, whose anchor is a *time window* — and a
time window stops lining up the moment a title's pacing changes, which makes the guard fail while
the picture is perfect. `blue-prince-hall` fails that way today: across all 96 sampled records
`nonblack_ratio` is 1.00 and the richest frame carries 139,312 colours; the route simply never
reaches the entrance hall inside 300–780 s any more.

## Authoring: play the game

```bash
python3 prosper/tools/snapshot/snaps.py author --name blue-prince --dump PPSA25009-app0
```

That launches the title with a real window, audio and pad, records the input route, and puts the
snaps in `~/snaps/blue-prince`. One command rather than four environment variables, because getting
`PROSPER_PAD_RECORD` wrong is not a *visible* mistake: you play the whole session, press F6/F7
happily, and only discover at import time that there is no route and none of it can be replayed.

It refuses to reuse a session directory, for the same class of reason — snap indices restart at 0
each run, so a second session into one directory would overwrite the first session's images while
appending to its manifest. Use a fresh `--out`, or `--append` if you mean it.

The equivalent by hand, if you want to add your own variables:

```bash
PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
PROSPER_SNAP_DIR=~/snaps/blue-prince \
PROSPER_PAD_RECORD=~/snaps/blue-prince/route.pad \
    ./prosper/build-linux/prosper-app <DUMP_ROOT>/PPSA25009-app0
```

Then, whenever the screen shows something worth remembering:

| key | meaning | what it becomes |
| --- | --- | --- |
| **F6** | "this frame looks **correct**" | a regression guard — if it stops matching, that is a FAILURE |
| **F7** | "this frame looks **wrong**" | tracked known-bad state — if it stops matching, that is INFO |

**Take negative snaps.** They are the half that does not exist in the old system, and the reason
titles improve silently: a broken title has no guard, so nothing notices when it starts rendering.
A recorded known-bad frame is the only thing that will ever tell you it changed.

Adopt the session:

```bash
python3 prosper/tools/snapshot/snaps.py import ~/snaps/blue-prince --name blue-prince
```

## Checking: one run per game

```bash
python3 prosper/tools/snapshot/snaps.py check              # everything
python3 prosper/tools/snapshot/snaps.py check blue-prince  # one title
```

The route is replayed once and the presented frame is captured at every authored anchor — **one run
per game**, however many snaps it holds. Exit 1 if any positive snap failed.

```
  snap   0 correct   flip   4000  OK    ssim 1.000
  snap   1 correct   flip   5500  FAIL  ssim 0.412 < 0.85
    expected .../failures/blue-prince-0001-expected.bmp
    actual   .../failures/blue-prince-0001-actual.bmp
    colors 16110 -> 3, nonblack 0.212 -> 0.001
    accept with: snaps.py accept blue-prince 1
  snap   2 incorrect flip   2500  CHANGED  ssim 0.203 -- known-bad frame no longer matches; look at it
```

## Accepting: the web-snapshot workflow

On a failure both images are retained, so the question is answerable by eye: *this is what was
approved, this is what the run produced — is the new one acceptable?* If it is:

```bash
python3 prosper/tools/snapshot/snaps.py accept blue-prince 1
```

A **changed negative** snap can be promoted in the same motion, which is the point of recording it:

```bash
python3 prosper/tools/snapshot/snaps.py accept blue-prince 2 --verdict correct
```

## The anchor, and why it is a flip count

Snaps anchor on the **pad flip ordinal** — display flips since the guest's first pad poll. That is
the axis `PROSPER_PAD_RECORD` writes routes against and `PROSPER_PAD_SCRIPT` replays, so the route
and the snaps index each other. Unlike wall-clock it is boot-speed-invariant (#302): a faster build
does not slide the anchor onto a different scene.

Measured on Blue Prince: two headless runs produced identical anchor landings — including a one-flip
overshoot at 900 → 901 that reproduced exactly — and byte-identical pixels.

### The window, and why it exists

The check does **not** compare only the frame at the anchor. It captures several samples either side
(default ±900 flips, 7 samples, always including the anchor) and takes the best match.

This is required, not defensive, and it was found by running the thing rather than by reasoning. A
check that had passed twice at `ssim 1.000` failed on a third run with `FAIL ssim 0.010,
colors 14446 -> 1`. Nothing about the renderer had changed — builds and CI were running
concurrently, Blue Prince's asset-loading boot got slower, and the anchor landed on a still-black
loading frame instead of the title screen. **Comparing only the exact flip makes the suite a measure
of system load.**

A flip ordinal is boot-speed-invariant with respect to *rendering* changes, which is why it is the
right anchor. It is not invariant to how fast the guest itself gets there.

The output names which offset won, so drift is visible before it bites:

```
snap 1 correct flip 5500  OK    ssim 1.000 (matched -400, landed -400)
snap 0 correct flip 4000  FAIL  ssim 0.014 < 0.85 (matched +600, landed +601)
  NOTE: the best match is at the EDGE of the +/-600 flip window, so the anchor may
  simply have drifted out of range rather than the picture having changed.
```

**An edge match means widen the window, not "the renderer broke."** Re-import with `--flip-window`
before treating it as a regression. `--window-samples` controls the sample count, and a window of 0
degrades to exact-anchor matching for a title that genuinely needs one.

Drift is concentrated in the **pre-input boot phase**, and there is a reason for that: the pad script
replays input *at flip N*, so once the route starts pressing buttons the guest re-synchronises to
the same anchors regardless of speed. It is the loading before the first input that floats. Snaps
taken very early in a boot are therefore the fragile ones — prefer anchoring after the route has
started driving.

Two further consequences worth knowing before you debug something surprising:

- **The capture can overshoot.** The app grabs the first presented frame at or after the anchor, so
  a slow frame lands late. Both numbers are recorded (`actual_f900_at901.bmp`) and the check prints
  `(landed +1)`. If a diff looks inexplicable, read the drift first.
- **A snap taken before the guest's first pad poll cannot be anchored.** The app warns at the moment
  you press the key, names the file `..._unanchored.bmp`, and `import` rejects it out loud. This is
  why the origin must never be established by a host hotkey — see the comment on
  `prosper_pad_flip_ordinal()` in `src/hle/input/hle_pad.cpp`.

## FMVs, and why the anchor survives them

**Movies play in the check exactly as they do when you author.** Nothing here disables video decode,
and neither do the old guards.

This is the case where a flip anchor earns its keep. An FMV of N frames contributes N flips however
slowly it renders — so Blue Prince's intro, measured at **4.8 fps** against ~180 fps at its menu
(#2215), shifts the run's wall-clock enormously and its flip anchors **not at all**. A wall-clock
anchor would be useless here; a flip anchor is unaffected.

What the movie *does* eat is real seconds. So the check does **not** stop on a timer: it runs until
the last anchor has been captured, and the timeout is a safety net for a hung run. Sizing a timeout
for a quick boot would cut the route off mid-run and report every later snap as `NOT REACHED` — a
failure with nothing to do with rendering, which is the whole class of bug this system exists to
remove.

## Save state: fresh by default, and why that is not optional

Both halves run with **both** save roots redirected to empty per-run directories:

    PROSPER_SAVEDATA_DIR -> SaveDataMemory slots (the whole save path for Unity titles)
    PROSPER_SAVE0        -> the /savedata0 file mount

Redirecting only the first leaves file-mount titles reading your real saves.

This is not hygiene, it is correctness. A title with a save offers **"Continue" above "New Game"** —
so the same D-pad inputs select a *different item*, and the route does not merely mismatch, it
diverges into a different part of the game. Authoring against real saves also silently writes into
them.

`--savedata preserve` exists and is deliberately loud about the consequence: a session authored that
way will not reproduce on a machine whose save state differs, including CI and anyone else's clone.

**Wanting a save is legitimate** — it is how you would skip a long intro and author deep-game content
without replaying an hour of route. The right shape for that is a save FIXTURE stored with the snap
set and copied into the fresh directory before the run, so the starting state is part of the
committed definition rather than a property of one machine. Not built yet; `preserve` is not a
substitute for it.

## Where things live, and what is never committed

| path | committed? | what |
| --- | --- | --- |
| `prosper/tools/snapshot/snaps/<name>.json` | **yes** | signatures, verdicts, anchors |
| `prosper/scripts/<name>/route.pad` | **yes** | the recorded input route (plain text) |
| `~/.local/share/prosper/snap-refs/<name>/` | **no** | the reference images |
| `prosper/tools/snapshot/failures/` | **no** | retained expected/actual pairs |

**Game imagery is never committed.** That is why the store holds a 144-byte luminance thumbnail
rather than a PNG. The consequence is worth stating plainly rather than discovering: on a fresh
clone the numbers still work, but *"show me what it used to look like"* only works for whoever
authored the run. Re-author to get the images back.

## What the comparison actually measures

A 16×9 luminance thumbnail compared by SSIM, with a 0.85 floor — the same shape and bar the old
gameplay guards use, chosen to survive subtle pixel improvements while catching a collapse. Colour
count and non-black ratio are recorded beside it and printed on a failure, because SSIM alone cannot
separate "the scene changed" from "the scene vanished": two flat frames of different flat colours
score well against each other.
