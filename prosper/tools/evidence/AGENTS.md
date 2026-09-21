# tools/evidence

Tools that check whether a capture actually shows what a claim says it shows.

Everything else in `tools/` helps you *produce* evidence — run a route, grab a frame, replay a
bundle. This folder is the other half: given a frame you are about to publish, does it support the
sentence you are about to write next to it? That question has its own failure modes, and they are
not the renderer's.

The founding case is `prerender_check.py` (instrument trap 230). A title that blits a full-screen
loading picture yields a frame that looks *better* than the emulator could plausibly render — which
reads as success rather than as the warning it is. One such capture passed a PR review, a `BLOG.md`
entry and `COMPATIBILITY.md` before a human recognised the artwork; no automated check existed that
could have caught it, and the frame was paired with a genuine "before" shot, so the A/B looked
rigorous while comparing a 3D render against a 2D blit.

**What belongs here:** checks whose subject is the *evidence*, not the emulator — "is this frame the
game's own artwork", "is this frame a duplicate of an earlier one", "does this capture actually
contain rendered geometry". They run against a candidate image plus a dump, need no build, and
should be cheap enough that nobody skips them before publishing.

**What does not:** anything that measures prosper's behaviour. A diagnostic that answers "what did
the renderer do" is a `PROSPER_*` switch or a `tools/gpu_*` tool, even when its output is an image.

**Two ways a frame can be the game's own artwork, and the second is the one that bites.** An EXACT
match is easy to catch on a mean difference. But a real loading screen usually has something drawn
on it — a progress bar, a hint caption — and a small bright overlay moves the mean a long way while
leaving almost every pixel identical to the stored asset. `prerender_check.py` therefore also gates
on the fraction of pixels within 8/255 (`--overlap`, default 90%), which the mean cannot see. This
was found in review: a 2%-of-height "LOADING..." bar drawn on the retracted frame flipped it from
"caught" to "no stored picture explains this frame" — the tool confidently clearing the exact image
it exists to catch.

**A comparison needs STRUCTURE, and brightness is not structure.** This was found in two review
rounds and the second half is the general case. Scoring every pixel makes an overlap measure
darkness: a black frame against a black asset scores 100%, and a repo screenshot scored 88.77%
against an unrelated `pic1.dds` purely on shared blackness. Gating on "has content" fixed the dark
end and left the bright one — many dumps ship an all-white `pic2.dds`, which an all-white frame
matches at mean 0.00 exactly as a black pair does. So the gate is on **luma standard deviation**, of
the candidate and of every asset, plus a coverage floor over informative pixels. A featureless frame
matches nothing at any brightness, and is not progression evidence in the first place.

**A mean ceiling was tried here and removed, which is worth knowing before adding one back.** It
looked reasonable — "a high matching fraction alone is not domination" — but an overlap bar of 75%
already bounds the mean, so any ceiling below that can only discard frames the bar would have
flagged. It cannot prevent a false positive and can only manufacture false negatives. The bound is

    mean ≤ 0.75·f·8 + 0.25·f·255 + (1−f)·15  =  f·69.75 + (1−f)·15

for informative fraction `f`. It increases in `f`, so it maximises at **full** coverage: **69.75**.
Derive it rather than quoting it — it was first published here as 63.75, which dropped the matching
pixels' own contribution (a pixel counted as matching may differ by up to 8 per channel, not 0;
0.75 × 8 = 6.00 is exactly the gap), and a hand-built frame reaches mean 69.67 at overlap 75.00%,
above the figure the same paragraph called unreachable. A companion claim that the worst case was
partial coverage was backwards too: at f = 0.25 the bound is only 28.69. Measured alongside it,
**100 of 233** structured assets (42.9%) carrying a bright
caption panel reach mean 40.5–59.25 while still exceeding the bar, so a ceiling of 40 silently
dropped nearly half. The case that motivated the ceiling was already rejected by the bar anyway.

The count was first published here as "4 of 13", from a probe that filtered to 1920×1080 and so
excluded every 4K asset — where the worst means live. A structured-asset sample runs roughly 6
such 1080p assets to 79 4K ones, so that probe saw about 7% of its own population and could not
express the case it was measuring. The observed worst mean, 59.25, sits under the analytic bound of
69.75 — a consistency check rather than an agreement. Real assets *approach* the bound without being
observed to reach it: saturating it needs every matching pixel to differ by exactly 8 *and* the
non-matching quarter to be pure black. But do not read that as "cannot". An ordinary defect gets
close — a blit with slightly wrong colour conversion shifts every pixel by a small constant, which
keeps them inside the 8/255 match window while adding up to 6.00 to the mean. Measured on a real
asset (`PPSA03001` `sce_sys/pic0.dds`, 24% caption panel): overlap 75.47% at mean 59.36, and with a
uniform +8 shift overlap 75.97% at mean **65.43** — 94% of the bound, from real content.

**And the two statistics need different gates**, which conflating them hid: `mean` is a whole-frame
average that darkness cannot corrupt, so two structured images at mean 0.00 are the same image
whatever fraction is bright. Applying the coverage floor to the mean as well made the tool *pass*
`LEGAL_2.DDS` — white text on black, structured but only 4.09% above the content floor — handed back
as its own candidate. Exact matching is therefore gated on structure only; the overlap fraction, the
statistic a small subset really can mislead, keeps the coverage floor.

Measured on the current thresholds, all positive instances constructed by hand outside the tool:

| check | result |
| --- | --- |
| synthetic flat frames (black/near-black/dark/white, 1080p+4K) x 55 dumps | **0 false matches / 440** |
| true positives (retracted frame, overlays to 20%, both title screens, `LEGAL_1..4`) | all caught |
| worst genuine render clearing the mean ceiling | **2.87%** overlap, against a 75% bar |
| featureless assets, of 1490 large assets across all dumps | 18 (1.2%), all sd <= 1.134; next real asset sd 4.653 |

The last row is why the structure floor is 3.0 rather than a number picked by eye: it sits in an
empty band with clean margin on both sides.

**The rule these tools follow, and the reason they exit the way they do:** *could not check* must
never be reportable as *checked and clean*. `prerender_check.py` exits 0 only when it actually
compared something and nothing matched; when it finds no comparable asset it exits 1, distinct from
both the pass and the match. A tool in this folder that fails silently is worse than no tool, since
its whole purpose is to be trusted at the moment somebody is about to publish a claim.

## `image_likeness.py` — colour-aware comparison against a reference frame

Added 2026-09-21 out of #2790, where the absence of this tool cost most of a session.

**The problem it exists for: every metric this project reaches for by default is
ACHROMATIC.** Non-black percentage, distinct-colour count, near-white fraction,
luminance percentiles — and the snapshot guards' SSIM over compact *luminance*
signatures — all share one blind spot. A frame can score 88% non-black with 3,676
colours and "correct" luminance percentiles while containing none of the reference's
colour at all.

That is not hypothetical. On *Sonic Frontiers*' Cyber Space stage those numbers rose
steadily across a day's work — 454 to 1,271 to 3,676 colours, clipping 42% to 0% —
while the project owner, looking at the same frames, kept reporting that no level was
visible. What the metrics were tracking was an atmosphere shader and some laser
effects. One run of this tool said it in a line: **the reference is 61.6% green and the
candidate 0.0% green**, and the candidate that scored best on every achromatic measure
scored *worse* than baseline on palette intersection (0.094 → 0.080).

**The score is a composite, and that is load-bearing.** `likeness` is
`sqrt(palette intersection * hue recall)`. The palette term alone is very nearly the
achromatic measure this tool exists to replace: a 512-bin RGB histogram is dominated
by whatever bins hold the most mass, which on most frames is the dark and grey bins.
An independent review measured a plain grey gradient at **0.502** against a real dark
frame, ranking third of eleven and above a real menu capture; the tool's own selftest
now reproduces a worse case, a grey gradient matched to the reference's luminance
range scoring **0.740** on palette alone and **0.000** on likeness. That is the
project's recorded gradient trap, reproduced on the tool written to prevent it.

**When the reference is itself achromatic, the second factor becomes STRUCTURE** —
the Pearson correlation of the two downsampled luminance grids, **multiplied by a
gain penalty** `min(slope, 1/slope)` from regressing the candidate on the reference,
and defined as 0 when either image has no variance. The two factors are blended by
how much chromatic mass the reference carries rather than switched at the boundary,
so there is no cliff (measured: a continuous 0.980 → 0.000 ramp across it).

This branch took **three** attempts and each failure is worth knowing, because all
three are the same trap wearing different clothes. Reference: the committed *Little
Nightmares III* title screen, 0.00% chromatic — and 32 of this repository's 190
screenshots are under the floor, so this is not a corner.

All three versions below are measured on **one** candidate set against that
reference, because an earlier version of this table quoted "a real capture" that
was a different image in different rows:

| version | all-black | ref × 0.01 (brightest pixel 2/255) | boot splash | EULA | what ranks first |
| --- | --- | --- | --- | --- | --- |
| v1 — bare palette fallback | **0.968** | 0.968 | 0.981 | 0.942 | splash, then **all-black 2nd — above the EULA** |
| v2 — plain Pearson correlation | 0.000 | **0.869** | 0.771 | 0.000 | **ref × 0.01**, above the real splash |
| v3 — correlation × gain penalty | 0.000 | 0.076 | **0.726** | 0.000 | the real splash |

What v3 fixes is that the matching capture ranks first. `ref × 0.01` still scores
a small non-zero (0.076) and so sits above two real captures that are *different
scenes* and score 0.000 — that ordering is correct, not a residue of the defect.

The palette term is *high* for a black candidate precisely because the reference is
dark, and Pearson is **affine invariant**, so in v2 nothing in the product could see
magnitude: an image that is black to look at correlated perfectly and outranked a
real frame. The gain penalty is what sees it.

**A tie at 0.000 is not a ranking.** Several genuinely unrelated candidates score
exactly 0.000, and among them the order is filename only — `--rank` now prints that
on the first zero row. An earlier version of this paragraph claimed all-black "ranks
last"; it does not, it ties at zero with the other non-matches. What it no longer
does is outrank a capture that actually resembles the reference.

Five modes, in the order they are usually reached for:

```bash
# 1. Is this frame like the reference, and what is it missing?   (no options)
image_likeness.py oracle.png candidate.bmp

# 2. Where is it wrong, spatially? 16x9 keeps cells square on a widescreen frame.
image_likeness.py --grid oracle.png candidate.bmp [--cells 16x9] [--threshold 32]

# 3. Which of these surfaces is closest to the reference?        (no options)
image_likeness.py --rank oracle.png /path/to/dumped/surfaces

# 4. Which REGION of the reference does each surface correspond to?
image_likeness.py --locate oracle.png /path/to/dumped/surfaces [--cells 64x36] [--threshold 64]

# 4b. The same question for ONE surface, printing every region's score.
image_likeness.py --regions oracle.png candidate.bmp [--cells 64x36] [--threshold 32]

# 5. Verify the measure's own claims before trusting a number from it.
image_likeness.py --selftest
```

**Each mode accepts exactly the options it USES, and refuses the rest.** `--rank`
and the default mode score by palette and hue, which have no grid and no tolerance,
so they take no options at all and reject `--threshold`/`--cells` rather than
accepting and ignoring them. "Accepted but unused" is indistinguishable from
"unknown and ignored" at the terminal, and both produce the same false reading:
several runs at several settings returning identical numbers, which looks exactly
like a robust result.

Mode 2 prints a map — lowercase means the candidate is *short* of the reference on
that channel, uppercase means *excess*. On the Sonic frame it renders the defect
directly: the lower two-thirds read `g` on the G-buffer (grass missing) and `B` on the
presented frame (atmosphere flooding the gap).

Mode 4 is the locator: an intermediate surface rarely holds the whole frame, so
scoring halves, thirds and quadrants separately turns "this scores 0.11" into "this
corresponds to the reference's lower half".

**Read it as triage, not as a gate.** It ranks and locates; a human still judges
whether a scene looks right. Two cautions learned while building it:

- **Run `--selftest` before trusting a number, and add an arm when you find a new
  way to fool it.** It is a few seconds, needs no fixtures, and every case in it is
  one that actually broke a previous version: the gradient above, a surface that IS
  the oracle's bottom half being located as "bottom third", a frame compared against
  itself reporting "top-right q" instead of "full", and each mode refusing the
  options it does not use. The arms are written so that reinstating the old
  behaviour reddens a named case — verified by mutation, **fifteen for fifteen**,
  including reinstating the exact pre-review region algorithm rather than a
  convenient substitute for it.

  **That distinction is the lesson, and it cost three rejections — the same
  mistake three rounds running, each time on the arm guarding the previous
  round's finding.** Round 1's region mutation replaced the crop with the whole
  reference; round 2's achromatic mutation set the second factor to 1.0 (giving
  `sqrt(palette)`) instead of reinstating `likeness = palette`. Both are
  *stronger* changes that redden for the wrong reason, and both left the real
  defect undetected — round 3 reinstated the actual round-2 branch and the suite
  stayed 26/26 green. **A mutation you write yourself will drift toward one that
  fails loudly; the only safe one is the diff you are actually replacing.** The first
  round claimed "four for four" and it was false: the region mutation replaced the
  reference crop with the whole reference, which is a *stronger* change that happens
  to redden, while reinstating the algorithm actually under review left the suite
  green. The fixture was two flat colour bands, and stretch-versus-crop is a no-op
  on flat bands — the defect was structurally inexpressible, so the arm tested the
  discriminator and not the domain. The fixture is now deliberately structured (a
  gradient plus four differently-coloured blocks at four distinct positions); do not
  simplify it. Three further arms were found to be unfallible the same way and were
  replaced: the tie-break arm (a self-comparison has a unique winner once cropping
  is correct, so only a flat-on-flat comparison actually ties), the `--cells` arm
  (both values must sit below the old clamp or it maps them apart anyway), and the
  skip-reporting arm (an all-corrupt directory exits non-zero whether or not skips
  are recorded — it takes a *mixed* directory, where the run succeeds and stderr is
  the only channel left). **When you add an arm, mutate the real defect and watch
  that arm go red.**
- **`--threshold` and `--cells` belong to the spatial modes only.** An early version
  dropped `--threshold` outside `--grid`; the repair was incomplete and a review
  found `--rank` still took no options at all while `--locate` accepted `--cells`
  and discarded it. If a mode accepts an option now, it uses it — pinned by four
  arms that compare CLI output at two settings.
  **Compare the result rows, never the whole output.** Every mode echoes its own
  settings in a header — `--locate` its `grid=`, `grid_compare` and `region_match`
  their `tol=` — so a whole-output comparison is satisfied by the echo whether or
  not the options reach the code that scores. Two reviews demonstrated it from
  opposite ends: reinstating the drop-the-options defect in `--locate` left the
  suite 27/27 green with all three `--cells` values byte-identical, and hardcoding
  the tolerance *inside* `grid_compare` left it 29/29 green. The `rows()` helper in
  the selftest drops `===` banners and any line carrying `tol=`/`grid=`, so the
  arms see only what was computed. All six (mode, option) pairs now have an arm.
- **Absolute scores are low even for good renders**, because exposure and tonemap
  differences shift every cell. Compare candidates against each other, and watch the
  per-hue deltas and the spatial map rather than the single number.

Still to do (deliberately not done at once): wire it into `tools/snapshot` so guarded
titles gain a colour-aware check alongside the luminance SSIM, which has the same
blind spot described above.
