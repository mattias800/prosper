# `tools/frameclass/`

Answers **"is there anything in this picture?"** for a directory of captured frames, and separates
the case a bring-up lane keeps confusing with the two either side of it: **UI drawn over a world that
never rendered**. That is the difference between "this title renders nothing" and "this title is at
rung 2", and between rung 2 and rung 3.

What is genuinely new here is narrower than it looks, so it is worth stating precisely.
`tools/screenshot`'s JSONL manifest **already** carries `distinct_rgb_colors` and
`nonblack_rgb_pixels`, computed over every pixel, in the same directory as the PNGs
(`capture_manifest.cpp:100-114`) — if you only need those two, read the manifest and skip this tool.
What this adds is the brightest channel anywhere, the count of pixels legible enough to read, and the
classification those two make possible. It also runs over *any* directory of PNGs, including frames
that arrived without a manifest.

Both obvious hand-rolled versions of that classification are wrong in ways that read as a result:

- **Downsampling before counting colours destroys thin UI.** A 4K frame resized to 160x90 loses white
  menu text on black entirely. Measured on *Stray* (`PPSA02101`) on 2026-08-27: a frame reported as
  `colours=61 FLAT-BLACK mean=(0,0,0)` contained START GAME / SETTINGS / CREDITS and a legible build
  stamp, at full brightness. The title was one edit away from being written up as not rendering.
- **"Fraction of non-black pixels" scores a flat WHITE clear as a perfect frame** — 100% non-black,
  and every "is it black?" test passes it.

## The classes, and where the boundaries came from

| class | meaning |
| --- | --- |
| `FLAT` | nothing rendered, or a uniform fill — a black clear, a white clear, a two-tone letterbox |
| `UI-ON-BLACK` | legible pixels covering under 10% — a HUD, menu, logo or title over an absent world |
| `SPARSE` | non-black but nothing legible anywhere: a fade, or a frame too dark to read |
| `CONTENT` | a real picture |

The thresholds are tuned against the 128 committed frames in `assets/screenshots/`, which are
human-reviewed milestone evidence and so the closest thing to ground truth this repo has. That corpus
puts a clean gap exactly where the boundary belongs: title screens, logos and EULAs cover
**0.15–8.62%** of the frame, while the least-covered real scene covers **10.13%**.

**An earlier revision of this tool drew CONTENT at 10% and UI-ON-BLACK at 2%**, which dropped all
fourteen lit title screens into the dead band between them and reported them `SPARSE` — including
`gta5-title`, `arcrunner-title-screen`, `metaphor-language-select`,
`little-nightmares-3-title-screen` and `forgotten-city-title`, each cited in `COMPATIBILITY.md` or
`BLOG.md` as its title's milestone. That is the same failure the tool exists to prevent, one
threshold along. Current corpus split: **CONTENT=109, UI-ON-BLACK=18, SPARSE=1** (`max=15`, genuinely
near-black), FLAT=0.

Two ordering rules are load-bearing, and each is pinned by a selftest case:

- **Coverage decides `CONTENT` before legibility does.** A well-covered frame is a picture even when
  nothing in it is bright: `bendy-title.png` is 93.33% covered with 2,899 colours at `max=127`.
- **A low colour count means "flat" only when those colours COVER the frame.** Gating on the count
  alone swallowed a black frame carrying prosper's two-colour fps overlay and 6,120 bright pixels.

## Caveats — both matter, neither is detectable from pixels

- **`--fps-overlay` burns prosper's own counter into the PNG**, and a black frame carrying only that
  overlay classifies `UI-ON-BLACK` — reading as "the guest is drawing its menu" when nothing of the
  sort happened. Do not classify overlay runs; `screenshot`'s manifest records
  `run.assertions.fps_overlay` if you need to check after the fact.
- **A smooth gradient with no picture in it classifies `CONTENT`.** prosper's seed-miss gradient is
  exactly that shape, so a frame can score well here and still be a diagnostic fill.

## How it samples, and the two ways that shows

A **1/16 nearest-neighbour stride** — every 4th pixel on both axes — at **native** resolution. Never
a resize: that is the distinction the whole tool rests on. A stride still aliases, and the shape of
the failure is worth knowing: a black frame ruled with 1px vertical lines at `x % 4 == 1`, covering a
quarter of the screen, reports `max=0 colours=1 FLAT`, indistinguishable from a black clear.

The colour count **saturates at 4096**, and a saturated count is printed with a trailing `+`. Two
`4096+` rows are not comparable with each other; neither is a `4096+` row against the manifest's
`distinct_rgb_colors`, which does not cap.

Alpha is discarded rather than composited over black, which looks like a divergence from
`capture_manifest.cpp:105-108` (it multiplies RGB by alpha). Measured on a real `screenshot` PNG it
is not: `normalize_capture_rgba` (`capture_manifest.cpp:90-97`) forces alpha to 255 for composited
and republished frames, and 0 of 8,294,400 pixels differed. `CaptureSource::RawScanout` is the one
path that leaves alpha untouched, and no sample of it has been checked — recorded so nobody
re-derives the negative.

## Running it

```bash
python3 prosper/tools/frameclass/frameclass.py ~/work/shots   # dir of .png/.bmp, or single files
python3 prosper/tools/frameclass/frameclass.py --selftest     # 9 hand-built frames, one per class
```

It is a reporting tool, not a gate: **it always exits 0 on a completed classification**, and an
unreadable or truncated PNG is reported on its own line and counted rather than aborting the run.
Non-zero is reserved for the tool failing — bad usage, no input, or a `--selftest` regression.

**Re-run `--selftest` if you touch a threshold.** Its cases are built pixel by pixel rather than
sampled from `assets/screenshots/`, deliberately: the thresholds are tuned against that corpus, so a
control drawn from it would confirm the tuning rather than the classifier. Each case pins a mistake
this tool has already made once, and each of the three fixes above has been checked by mutation —
reverting it reddens the case that names it.
