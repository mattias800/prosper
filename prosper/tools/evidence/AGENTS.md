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
already bounds the mean at ≤ 63.75/255, so any ceiling below that can only discard frames the bar
would have flagged. It cannot prevent a false positive and can only manufacture false negatives.
That bound is the argument, and it is worth preferring to any count: it is derivable, so it belongs
to nobody's sample. Measured alongside it, **100 of 233** structured assets (42.9%) carrying a bright
caption panel reach mean 40.5–59.25 while still exceeding the bar, so a ceiling of 40 silently
dropped nearly half. The case that motivated the ceiling was already rejected by the bar anyway.

The count was first published here as "4 of 13", from a probe that filtered to 1920×1080 and so
excluded every 4K asset — where the worst means live. A structured-asset sample runs roughly 6
such 1080p assets to 79 4K ones, so that probe saw about 7% of its own population and could not
express the case it was measuring. The observed worst mean, 59.25, sits just under the analytic
bound of 63.75, which is the reassuring part: two independent routes agree at the boundary.

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
