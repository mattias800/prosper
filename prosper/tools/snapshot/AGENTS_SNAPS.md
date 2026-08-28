# Human-authored render snapshots ("snaps")

You play the game. You decide what is correct. The tool remembers.

This replaces the scripted-route guards in `snapshot.py`, whose anchor is a *time window* — and a
time window stops lining up the moment a title's pacing changes, which makes the guard fail while
the picture is perfect. `blue-prince-hall` fails that way today: across all 96 sampled records
`nonblack_ratio` is 1.00 and the richest frame carries 139,312 colours; the route simply never
reaches the entrance hall inside 300–780 s any more.

## Authoring: play the game

Run the title normally — a real window, audio, a pad — and record the route while you play:

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

Two consequences worth knowing before you debug something surprising:

- **The capture can overshoot.** The app grabs the first presented frame at or after the anchor, so
  a slow frame lands late. Both numbers are recorded (`actual_f900_at901.bmp`) and the check prints
  `(landed +1)`. If a diff looks inexplicable, read the drift first.
- **A snap taken before the guest's first pad poll cannot be anchored.** The app warns at the moment
  you press the key, names the file `..._unanchored.bmp`, and `import` rejects it out loud. This is
  why the origin must never be established by a host hotkey — see the comment on
  `prosper_pad_flip_ordinal()` in `src/hle/input/hle_pad.cpp`.

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
