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

**Pacing makes the run length a floor, not just a ceiling.** Because the check now paces flips to
`det_fps`, a set cannot finish faster than `deepest_candidate_flip / det_fps` seconds however fast
the machine is. For an anchored set that is small — `alexkidd`'s deepest candidate is 3134 + 900 =
4034 flips, about **67 s** at 60/s. A **scan** snap extends its own anchor by `scan_forward` (12,000
flips by default), so one scan snap puts a hard **~200 s minimum** on the whole set. Size `--timeout`
above that floor, not merely above a fast machine's observed run.

### Do not anchor a snap mid-fade — measured, and it is the one real trap

Two identical headless runs were driven through Blue Prince's New Game intro and sampled at the same
anchors. At flip 8000 they agreed (**SSIM 0.995**). At flip 10000 they scored **0.465** — a failure.

Opening both frames explains it, and the explanation is reassuring about FMVs and unforgiving about
transitions: **both runs were on the same page of the same cutscene text, word for word.** The drift
across the movie was small. What differed was that one run was *mid-fade* (non-black 0.695 vs 0.365)
and had a subtitle the other had not yet shown.

So the rule is not "avoid FMVs". It is:

> **Take snaps on visually STABLE moments.** A title screen, a menu, a held gameplay view. Not
> during a fade, a wipe, a subtitle transition, or a rapidly cutting cinematic.

A few flips of drift on a stable scene is invisible. The same drift during a fade is a large
luminance change, and SSIM is sensitive to luminance by design — it is what makes it catch a
collapse to black.

The window sampling is geometric for this reason (dense near the anchor, sparse at the edges), since
drift is small: with `--window-samples 9` over ±900 the offsets are `0, ±43, ±196, ±478, ±900`
rather than uniform ±225 steps. That resolves a short fade far better than even spacing, but it
cannot rescue a snap taken in the middle of one.

## Flips are paced on BOTH sides, which is what lets the clock stay off

Routes are **flip-anchored** (`fN` = display flips since the first pad poll), so flip N is only a
fixed moment in the game if flips happen at a fixed *rate*. Pinning the guest clock used to provide
that, at the cost of lying about every guest time source. Pacing provides it honestly: the guest
keeps a real clock and simply flips at 60/s, the way a vsync-locked console does.

Both authoring and checking set `PROSPER_FLIP_PACE_FPS` to the set's `det_fps`, and they must agree.
Without it the route does not drift, it **diverges**: a press window lands at a different guest time
and can be swallowed. Measured on `alexkidd` with the check unpaced — **all four snaps failed**, with
colour counts collapsing 14,548 → 108, which is a different part of the game rather than a shifted
anchor. No amount of scanning fixes an input that landed in the wrong place.

An old set stored before the `det_clock` key existed reads as **clock ON**, because there was no way
to author one with it off. The author default being `off` does not change how such a set is replayed.

**Pacing can only slow a fast host down — it cannot speed a slow one up.** `flip_pace_wait()` sleeps
when it is ahead of schedule and *re-anchors* when it is behind, so on any title/host combination
that cannot **sustain** `det_fps` the pacer is inert and the drift described above comes straight
back. This is the same shape as vsync, which caps a maximum rather than guaranteeing a rate — and
the numbers in the next section are the warning: Blue Prince measured 180 fps windowed at its menu,
20 fps in gameplay and **4.8 fps during its FMV**. The last two are below any sane `det_fps`.

So pacing is what makes an anchored snap reliable *on frames the host can render at rate*. For
anything sitting behind a load screen or a movie, use a **scan** snap (`Shift+F6`) — that is exactly
the case it exists for, and no amount of pacing substitutes for it.

## The guest clock: OFF by default (and why it used to be on)

**Both halves now run with the guest's REAL clock.** `--det-clock on` remains available per title and
is off by default, because pinning the clock is a genuine correctness compromise: `PROSPER_DET_CLOCK`
replaces *every* time source the guest has — `sceKernelReadTsc`, `GetProcessTime`, and the wall-clock
anchor behind `CLOCK_REALTIME`, `gettimeofday`, `time()` and `sceRtc*` — with
`anchor + flips × (1/DET_FPS)`.

A guard recorded under that clock tests a machine nobody plays on. GRIS shows the divergence is not
theoretical (below). Drift is handled by scanning for the frame instead; see SHIFT+F6 above.

The history is kept because the measurement behind it is still true and still instructive:

Without it the anchors are frame-rate dependent and simply cannot correspond. Measured on the first
real authoring session: authoring windowed ran at **60.4 fps**, the headless check at **77.1 fps**,
and a time-based intro logo therefore burned ~28% more flips in the check. The drift matched the
ratio almost exactly — +600 flips at authored anchor 2516 against a predicted +694, and past the
±900 window entirely by anchor 4336. The picture was identical; only the rate differed, and the
check reported a confident FAIL showing an intro logo where gameplay had been snapped.

"Just run the check windowed too" does not fix this. Vsync caps the maximum, it does not guarantee a
rate: Alex Kidd authored at 60.4 fps windowed, while Blue Prince ran at 180 fps windowed at its menu,
20 fps in gameplay and 4.8 fps during its FMV. Any title that cannot hold its cap drifts again, the
rate becomes tied to the developer's display refresh, and a 17-title matrix would need 17 windows.

**Verified rather than assumed.** The same route was replayed at two deliberately different host
speeds with the clock on — 78.2 fps against 52.4 fps, a 1.49x spread:

| anchor | SSIM | |
| --- | --- | --- |
| 1800 | 0.9618 | pass |
| 2500 | 0.9917 | pass |
| 4300 | 0.9784 | pass |

All three comfortably above the 0.85 bar, where the uncorrected rate difference had pushed an anchor
past its whole window.

## Analog sticks record as DIRECTIONS, like a d-pad

A stick past its dead zone records as `left-stick-left`, `right-stick-down` and so on, `+`-joined
with any buttons held at the same time, in the exact vocabulary `PROSPER_PAD_SCRIPT` replays. A
recorded interval round-trips through the parser unchanged.

This is deliberately coarse, because the script vocabulary is **full-deflection only** — there is
nothing finer to record. The consequence is worth knowing before authoring a 3D title: a gentle tilt
records and replays as a full push, so a route that depends on fine analog control will not
reproduce its exact path. Prefer short, decisive movements when authoring, and take the snap once
the camera has settled.

The previous behaviour was strictly worse: sticks were not recorded at all, so a stick-driven route
replayed as **standing still** — silent, total, and indistinguishable from the game ignoring input.

The dead zone (48 of 128) is generous on purpose: a resting stick drifts, and a recorder that emits
an interval on every wobble produces a route full of one-flip noise entries that replay as real
input.

### SHIFT+F6 / SHIFT+F7 — a snap to SCAN for

Some frames cannot be found at a fixed offset, because what precedes them takes a different length of
time on a different machine: a loading screen, an FMV, a streaming pause. Mark those with **shift**
held, and the check sweeps a wide span forward of the anchor instead of hugging it.

You are the only one who knows which frames those are, and you know it at the moment you press the
key — which is why it is a modifier rather than a setting decided later.

Scanning is **bounded and forward-biased**. What the bound buys is finite search cost and a finite
run — *not* protection from matching the wrong occurrence, because `best_match` takes the argmax over
every sampled offset rather than the first match. That earlier framing was wrong and is corrected
here rather than repeated.

**The real hazard is a scene that recurs INSIDE the span**, which the bound does not address: at 60
flips/s the default forward span is about 200 s of play, easily long enough to contain a menu you
return to. Use an anchor snap for anything that recurs; keep scans for frames that sit after
something of variable length and appear once.

Sampling across a scan is uniform, unlike the tight window's geometric spacing: a scan is used
precisely when the match is *not* near the anchor, so weighting toward the anchor would spend the
samples in the least likely place.

**This is what replaced the deterministic clock.** Drift stopped being something to eliminate by
lying to the guest about time, and became something to search past — which covers slow test runners,
variable loading, FMVs and frame-rate differences with one mechanism, and leaves normal play
untouched at whatever frame rate the machine can manage.

### The clock is not universally safe — GRIS is the counterexample

`--det-clock off` exists because the pinned clock **breaks some titles outright**, and this was found
by a person trying to author one.

GRIS, with a New Game route reaching its opening FMV, measured over 150 s:

| | frames | `Forcing submitDone` stalls |
| --- | --- | --- |
| clock **off** | **42,000** | 0 |
| clock **on** | **1,680** | **47** |

A 25x collapse and a frozen white screen where the movie should play. Booting to its title screen is
unaffected — the FMV is the trigger, and the mechanism is obvious in hindsight: video playback is
A/V-sync sensitive, and a clock advancing per *flip* rather than per *second* makes the decoder wait
for a presentation time that never arrives at the rate it expects. `PROSPER_DET_CLOCK` is marked
`CONFIDENCE: MED` in `hle_kernel_time.cpp`, and this is what that caveat looks like in practice.

**If a title freezes or an FMV stops during authoring, re-author with `--det-clock off`.** The
symptom is unmistakable and appears while you are sitting there. With the clock off the anchors are
frame-rate dependent again, which is what the search window absorbs — such a set may need a wider
`--flip-window`.

The setting is recorded per title, so the check reproduces how the session was authored.

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
