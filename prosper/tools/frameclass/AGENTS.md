# `tools/frameclass/`

Answers **"is anything drawn in this picture, is any of it legible, and is it only a HUD?"** for a
directory of captured frames. The last of those is the one worth having: a HUD or notice over a world
that never rendered is the signature of a title that executes but does not draw, and it is invisible
to every "is the picture changing?" metric.

What is genuinely new here is narrower than it looks. `tools/screenshot`'s JSONL manifest **already**
carries `distinct_rgb_colors` and `nonblack_rgb_pixels`, over every pixel, in the same directory as
the PNGs (`capture_manifest.cpp`) — if you only need those, read the manifest and skip this tool.
What this adds is the peak channel value, the classification, and running over any directory of
frames including ones with no manifest.

Both obvious hand-rolled versions of the classification are wrong in ways that read as a result:

- **Downsampling before counting colours destroys thin UI.** A 4K frame resized to 160x90 loses white
  menu text on black entirely. Measured on *Stray* (`PPSA02101`) on 2026-08-27: a frame reported as
  `colours=61 FLAT-BLACK mean=(0,0,0)` contained START GAME / SETTINGS / CREDITS and a legible build
  stamp, at full brightness. The title was one edit from being written up as not rendering.
- **"Fraction of non-black pixels" scores a flat WHITE clear as a perfect frame** — 100% non-black,
  and every "is it black?" test passes it.

## The classes

| class | meaning |
| --- | --- |
| `FLAT` | nothing rendered, or a uniform fill: peak ≤ noise, or ≤ 8 colours covering ≥ 50% |
| `SPARSE` | nothing rises above near-black (peak < 48), whatever the coverage |
| `UI-ON-BLACK` | legible content covering **under 2%** — a HUD, notice or logo over an absent world |
| `LIT` | something legible is drawn over more than 2% of the frame |

Corpus split over the 129 committed frames in `assets/screenshots/` **as of 2026-08-27**: LIT=124,
UI-ON-BLACK=4, SPARSE=1, FLAT=0. That corpus grows, so re-run rather than quoting the figure; what is
worth remembering is which frames are *not* LIT, since that set is small enough to check by eye.

## `LIT` does not mean "a game scene", and no threshold can make it mean that

This is measured, not caution. On the committed corpus, UI and real scenes **interleave on every
statistic this tool computes**:

| frame | coverage | colours | what it actually is |
| --- | --- | --- | --- |
| `crisis-core-title.png` | 10.13% | 4096+ | text and a glow on black (key art missing, #2057) |
| `stray-brightness-calibration.png` | 10.13% | 228 | a settings menu on black |
| `messenger-title.png` | 12.41% | 36 | real pixel art |
| `oregon-trail-gameloft-splash.png` | 16.03% | 2159 | a flat logo on black |
| `blue-prince-title.png` | 21.83% | 4096+ | real rendered 3D art |

Coverage puts a logo above a scene; colour count puts pixel art below a logo. **An earlier revision
of this tool drew the boundary at 10% coverage and claimed the corpus showed a "clean gap" there.**
It did not: that gap came from taking the minimum over frames a *previous revision of the same tool*
had labelled scenes — the assumption under test. Both frames defining it are UI. Deciding "is the
world rendering?" above 2% needs eyes on the frame; this tool narrows which frames need them.

**The 2% bound on `UI-ON-BLACK` is where the class was independently validated**, on frames chosen by
someone other than the tool's author: `sonic-frontiers-cyberspace-hud` (0.81%),
`sonic-frontiers-autosave-notice` (1.05%) and `metaphor-loading-mascot` (1.49%) are exactly the
titles recorded as "world black behind the HUD" (#2790) and "the background does not draw" (#2952).

Two ordering rules are load-bearing, and each is pinned by a selftest case:

- **`SPARSE` is keyed on the PEAK, not on a count of bright pixels, and coverage does not rescue it.**
  `bendy-title.png` peaks at 127 across 93% of the frame and is a perfectly visible picture, so a
  "pixels brighter than 128" test called it empty; conversely a fifth of a frame at peak 15 is still
  nothing anyone can see.
- **A low colour count means "flat" only when those colours cover ≥ 50%.** At ≥ 90% a 2.39:1
  letterbox (black bars plus a uniform fill, ~75%) classified as a picture; gating on the count alone
  instead swallowed a black frame carrying prosper's two-colour fps overlay and 6,120 bright pixels.

## Caveats — neither is detectable from pixels

- **`--fps-overlay` burns prosper's own counter into the PNG**, and a black frame carrying only that
  overlay classifies `UI-ON-BLACK` — reading as "the guest is drawing" when nothing of the sort
  happened. Do not classify overlay runs; `screenshot`'s manifest records `run.assertions.fps_overlay`
  if you need to check after the fact.
- **A smooth gradient with no picture in it classifies `LIT`.** prosper's seed-miss gradient is
  exactly that shape, so a frame can score well here and still be a diagnostic fill.

## How it samples

A **1/16 nearest-neighbour stride** — every 4th pixel on both axes — at **native** resolution. Never
a resize: that is the distinction the whole tool rests on. A stride still aliases: a black frame
ruled with 1px vertical lines at `x % 4 == 1`, covering a quarter of the screen, reports
`max=0 colours=1 FLAT`, indistinguishable from a black clear.

The colour count **saturates at 4096**, printed with a trailing `+`. Two `4096+` rows are not
comparable with each other, nor against the manifest's uncapped `distinct_rgb_colors`.

Alpha is discarded rather than composited over black, which looks like a divergence from
`capture_manifest.cpp` (it multiplies RGB by alpha). Measured on a real `screenshot` PNG it is not:
`normalize_capture_rgba` forces alpha to 255 for composited and republished frames, and 0 of
8,294,400 pixels differed. `CaptureSource::RawScanout` is the one path that leaves alpha untouched,
and no sample of it has been checked — recorded so nobody re-derives the negative.

## Running it

```bash
python3 prosper/tools/frameclass/frameclass.py ~/work/shots   # dir of .png/.bmp, or single files
python3 prosper/tools/frameclass/frameclass.py --selftest     # 11 hand-built frames
```

It is a reporting tool, not a gate: **it always exits 0 on a completed classification**, and an
unreadable or truncated PNG is reported on its own line and counted rather than aborting the run.
Non-zero is reserved for the tool failing — bad usage, no input, or a `--selftest` regression.

**`--selftest` is deliberately not registered in ctest**, though the repo has that pattern (see
`tools/colorstate/`). It needs Pillow, which is not a build dependency, and making it one for a
reporting tool that gates nothing is the wrong trade. Run it by hand.

**Re-run `--selftest` if you touch a threshold.** Its cases are built pixel by pixel rather than
sampled from `assets/screenshots/`, deliberately: the thresholds were tuned against that corpus, so a
control drawn from it would confirm the tuning rather than the classifier — which is exactly how the
"clean gap" above came to be believed. Each case pins a mistake this tool has already made, and each
rule is checked by mutation: reverting it reddens the case that names it.

## `letterbox.py`

Answers a different question from `frameclass.py`: not "does this frame carry content" but "is the
title showing a **cutscene**". It measures the cinematic bars, which are geometric and therefore
survive the colour degradation that makes chromatic metrics unreliable on several UE4 titles here.
`--require-hud-corners` adds a positive half, `--after` restricts to a phase, `hud_displacement`
phase-correlates a HUD region between two frames, and `--selftest` runs constructed cases.

Three limits, all paid for on `PPSA17942` and all repeated in the file's own header:

- **A collapsed present has no dark rows either**, so absence of bars reads as gameplay. The guard is
  a structure floor, not a brightness floor and not a colour count: a *flat* collapse and a
  *speckled* one (18 distinct colours over 8.3 MP) both have to fail, and a brightness floor is
  actively inverted, since real HUD-over-dark frames are darker than the garbage.
- **Absence of bars is not presence of gameplay** — a menu has none. Pair it with a positive
  per-title marker.
- **The generic corner test is not that marker.** It fires on a colourful cutscene and on a flat
  saturated collapse alike. Where a title draws a distinctive HUD element, key on that instead:
  `scripts/dragon-quest-vii/classify_field.py` uses the party block's HP bar and separates by ~1.5x
  on both sides, where the corner test produced false positives in both directions.

The thresholds here have less headroom than the prose might suggest — on the title they were tuned
against, the structure floor sits about 3 units below the field-frame minimum. Re-measure before
trusting them on a new title, and prefer a per-title marker for anything load-bearing.
