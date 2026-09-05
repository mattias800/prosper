# AGENTS.md - rendering snapshot tests

This is the local real-game rendering regression inventory. Run and review the
full matrix before every release. It is not a day-to-day development or merge
gate: a PR author may skip snapshots entirely during long iterations or run only
useful focused guards, unless the task explicitly requires a run. Snapshot results
do not define whether master or a PR is acceptable; either may contain a detected
regression, including an intentional cross-title
tradeoff from an otherwise correct fix. Release review decides whether to fix the
regression or explicitly accept and document it before publishing artifacts.

```bash
# From prosper/, with build-linux/boot_trace built from the change.
python3 tools/snapshot/snapshot.py check
python3 tools/snapshot/snapshot.py check messenger-scene
python3 tools/snapshot/snapshot.py check dead-cells-splash
python3 tools/snapshot/snapshot.py check blasphemous2-gameplay
```

## Guards that are NOT currently a regression signal

A red guard is only evidence about your change if the guard itself is sound. Two are not, and a lane
that treats either as a regression will spend a session chasing its own tools. **Selection is
positional** — `snapshot.py check <name> [<name>...]`; there is no `--only` flag, and an
unrecognised option is now refused with exit 2 rather than read as a guard name.

- **`gris-gameplay` — do not run it, and do not act on it.** Project-owner decision, 2026-09-02: its
  baseline has not been updated and *GRIS* is to be verified by hand instead. It fails on master
  independently of any branch (#3148), and it has now been seen failing at two *different* points in
  the route — stuck on the publisher splash with `structural matches 0 of 60`, and separately with
  the character visible and SSIM 0.12-0.14 against every baseline. Neither is a rendering
  regression. Exclude it from any subset you run, and if you run the full matrix, do not report its
  failure as a finding.
- **`terminator-boot` clears its own threshold by exactly zero frames** (#3208): 14 consecutive
  content matches against a required 15 in one sweep, and exactly 15 in the next, on builds whose
  difference cannot affect that title's boot timing. Its menu arrives ~42 s into a hard 62 s window,
  so one second of extra boot latency flips the verdict. A failure reading
  `consecutive content matches 14 < 15` is indistinguishable from a real regression that dropped one
  menu frame, and it is almost always the former.

Neither entry means the guarded title is broken, and neither should be "fixed" by lowering a
threshold — that trades a false alarm for a guard that asserts less. Both need a re-profiled window
against the current boot, which is its own reviewed baseline change.

### A wall-clock-anchored guard measures a TIMELINE as much as a picture

**Measured across the whole matrix in #3350: all 17 guards carry
`capture_after_seconds`/`capture_before_seconds`, and not one is event-, flip- or content-anchored.**
So this is not a property of a few entries — the entire release-time regression inventory asks "what
is on screen between second X and second Y". `terminator-boot` below is the previously recorded
instance, but it is not a special case, and reading it as one costs the next lane a session. Every guard here anchors its evidence window on **wall-clock seconds**
from launch, and its route drives input on wall-clock seconds too. So the window does not ask "is the
title drawing the reviewed scene?" — it asks "is the title drawing the reviewed scene **at second
N**?", and those come apart the moment anything changes how long the boot takes.

**The trap is the direction.** The change that moves a boot's duration is very often a *fix*. On
2026-09-05 (#2899) `sceAvPlayerJumpToTime` began publishing the seek-completion notification it had
never sent, so *Space Adventure Cobra*'s opening movies **played** instead of timing out after 15 s
each — strictly more correct, and it moved everything downstream of them. A guard that reddens
because the emulator got better is a guard somebody will eventually silence for the wrong reason.

**What this looks like from the outside**, and why the counters cannot tell you: the frames are a
**real rendered state of the game** that is simply not the guarded one. On the same day
`messenger-scene` failed with `non-black coverage matches 0` and `SSIM ≈ -0.007` against a 0.43
floor — which reads as a collapse, and was a **dialogue screen**: two portraits in bordered boxes
with text beside them, on black, i.e. mostly black *by design*. `320/320 source-distinct,
pixel-distinct=163, guest=running` says the same thing from the other side.
"the route is somewhere else" and "the renderer died" produce indistinguishable counters and demand
opposite responses.

So, before treating any structure- or coverage-only failure as a regression:

1. **Open the retained PNGs.** Ask what state the game is in, not whether the number is low.
2. **Run `when_matched.py`** (below) over a full-timeline capture. It answers whether the run reaches
   the reviewed state at *any* second, which is the question the windowed verdict cannot ask.
3. **Only then** decide. A high score outside the window is a timeline shift and the ROUTE is what
   drifted; nothing above the floor anywhere is not a timeline shift, whatever the frames look like.

And the fix for a drifted guard is to **re-anchor the route on flips**, not to re-capture the
references: flip-anchored routes survive a timing change, wall-clock ones do not, and re-capturing
keeps the guard red-free while quietly deleting the state it was built to protect.

## When a guard fails on STRUCTURE only: `when_matched.py`

`FAIL — structural matches 0 < 21` is the one verdict that does not say what happened. The title
rendered (the other counters crossed) and did not match the reviewed references, and two very
different situations produce that same line:

- the title still reaches the reviewed state but at a **different wall-clock time**, because
  something upstream changed how long the boot takes. The route's anchors then point at the wrong
  moment. Re-capturing the baseline here would be laundering — the references are still right and
  the ROUTE is what drifted.
- the title **never reaches the reviewed state at all**. That is content loss, and re-capturing the
  baseline would delete the evidence of it.

```bash
python3 tools/snapshot/when_matched.py <guard-name> <a run's capture.jsonl>
```

It scores **every** sample of a capture against that guard's `structural_references`, using the
checker's own SSIM over the `luma16x9` signatures the manifest already carries — so it needs no
images, no image library, and it cannot disagree with the checker about what "matching" means. It
prints the best-scoring samples with their timestamps, the best inside the guarded window, the best
anywhere, and how many samples clear the floor at any point in the run.

Read it as: **a sample at or above the floor outside the window** is the timeline shift, demonstrated
rather than argued. **Nothing at or above the floor anywhere** is not a timeline shift, whatever the
frames look like to the eye — and it is the answer that must survive wanting the other one.

`snapshot.py check` deletes its capture directory, so run `tools/screenshot` separately with the
guard's own env, route and scale to get a manifest to sweep. Exit status is a contract: **0** means
the sweep ran and the printed scores are the answer *including when nothing matched*; **2** means it
refused and no number it printed is a result.

**Read that exit status directly, never through a pipe.** `when_matched.py … | tail -1; echo $?`
reports `tail`'s status, so a refusal reads as success — which is exactly the failure the contract
exists to prevent, arriving through the one channel you are not watching while you concentrate on the
output. The tool's own author hit it on the first test (charter trap 40, a pipeline's status is its
last stage's): use `cmd > out 2> err; rc=$?`, or `set -o pipefail`, or read `${PIPESTATUS[0]}`.

First use, #2899: it contradicted the person who wrote it. A `cobra-gameplay` run whose frames were
visibly the right level scored **0.7757 at best anywhere** against a 0.85 floor, with the in-window
best at **0.7582** — and it is that comparison, not the bare "nothing cleared the floor", that
settles it: the in-window best sits barely below the run's global best, so the window is not aimed
at the wrong moment, it is aimed at very nearly the closest thing the route ever produces. The "it
arrives at a different time" reading the author had built from three crossed counters and his own
eyes on the frames was false, and no baseline was touched.

That run also exposed a gap in the guard itself, worth knowing before anyone reads a score from it:
`cobra-gameplay`'s `_note` records exactly two calibration points, an off-scene pre-gameplay movie
frame at **0.3074** and a reviewed in-window worst of **0.8606**. Nothing between them was ever
observed, so a run scoring 0.70-0.78 lands in a band the guard cannot interpret — it distinguishes
"wrong scene" from "the reviewed scene" and says nothing about "right scene, degraded", which is
where that title now sits.

## Contract

- Local only. Game dumps and captured imagery must never be committed.
- Gameplay guards are deliberately coarse. Subtle pixel changes may be valid or
  improvements; use tolerant average/difference hashes to detect major collapse,
  missing layers, lost progression, and wrong dimensions without freezing every pixel.
- A content guard examines all frames in a gameplay-only evidence window and
  requires multiple qualifying frames. Do not treat one richest frame, a logo,
  a menu, or a static screen as proof of gameplay.
- `check` is fully automated. On failure, inspect the representative PNGs and
  log under `tools/snapshot/failures/` before changing code or thresholds.
- Do not lower a threshold merely to pass. Explain intentional contract changes
  and repeat the baseline-review workflow below.

`screenshot` normalizes alpha only for rendered captures because `prosper-app`
presents them through an opaque Vulkan swapchain. Do not re-premultiply those
pixels or calibrate a threshold from a transparency-aware image viewer: neither
matches the user's desktop. Raw scanout is different evidence and keeps guest
alpha unchanged.

### Why the content contract is conjoined, measured on real titles

"Colour count must never be the contract on its own" is not a precaution; it is
a measured result. Two profiled titles settle it by demonstration, and they
matter as a **pair**: in each one, a different metric goes blind, so neither can
be the contract alone.

| title | colour count | non-black coverage |
|---|---|---|
| **Rugrats** (`PPSA23396`) | separates cleanly — menu 56,090 vs gameplay 17,645 | **blind** — title card fully opaque at 1,551 colours |
| **Greak** (`PPSA02849`) | **blind** — 87-colour gap (0.27%) between cinematic peak 32,153 and gameplay floor 32,240 | separates cleanly — letterboxed 0.6667 vs gameplay 1.0000 |

Whichever metric you were about to trust on its own, one of these two titles is
the counter-example. That is the whole argument for conjoining `min_colors`,
`min_nonblack_ratio`, SSIM, `dims` and `min_pixel_changes`.

Profiling the Rugrats route once per second across a whole boot produced both
halves of the argument from one run, and they fail in **opposite** directions:

- **Colour count alone prefers a menu to the game.** The two richest frames of
  the entire run are the GAME MODE selector at 56,071-56,090 distinct colours.
  The best actual gameplay frame reaches 17,645. A colour-only guard would rank
  a menu **3.2x above every frame of the scene it exists to protect**, so it
  would pass a build whose gameplay had collapsed as long as the menu still drew.
- **Coverage alone accepts a nearly colourless screen.** The "BABIES IN
  GAMELAND" level-title card and its fade are **fully opaque — a 1.0 non-black
  ratio — at only 1,551-3,448 colours**. 54 frames of the run reach 0.999
  coverage with as few as 1,551 colours, so a coverage-only guard treats a
  title card as a rendered level.

Greak then supplied the inverse case. Its route passes through a **letterboxed**
level-intro cinematic immediately before gameplay, and that cinematic peaks at
32,153 colours while the gameplay floor is 32,240 — a gap of **87 colours, or
0.27%**. No usable `min_colors` separates them. What does separate them is
coverage: the letterbox bars hold the cinematic at 0.6667 while every gameplay
frame measures exactly 1.0000.

The practical rule: let SSIM decide *scene identity*, and keep `min_colors` as a
gross-collapse floor rather than tuning it to separate two valid scenes. For
Rugrats the tempting floor is ~16,000 — just under the 17,259 gameplay minimum
and just over the 15,030 menu ceiling — but that discrimination is redundant
with SSIM while making the guard fragile against a slightly dimmer healthy
frame. 12,000 was chosen instead: comfortably below the observed gameplay range
and far above every observed failure state (black at 1 colour, logos at most
3,830, title card at most 3,448).

The corollary is that **when the usual discriminator goes blind, the other
threshold has to carry the load, and should be set for that job rather than from
the generic formula**. `greak-gameplay` therefore sets `min_nonblack_ratio` to
0.9 instead of the derived 0.5 (half the lowest reviewed coverage): colour count
provably cannot catch a window drifting into that cinematic, so coverage must,
and the derived 0.5 would have admitted the 0.6667 letterbox. Raising a
threshold this way needs the same evidence as lowering one — here, every
reviewed gameplay frame measured exactly 1.0000 with zero variance across
independent runs, leaving a 10% margin.

### Three later titles each break a different one of the three checks

The Rugrats/Greak pair above shows colour and coverage each going blind on its
own. Three titles guarded on 2026-08-01 sharpen that: in each, **SSIM ends up
carrying the whole discriminating load**, for a different reason.

| title | what goes blind | measured |
|---|---|---|
| **Worms Armageddon** (`PPSA20052`) | colour, catastrophically | gameplay is an **801-colour** scene at scale 4; its own title screen is **75,875** — a factor of **95** (Rugrats' is 3.2x) |
| **GRIS** (`PPSA09804`) | **both**, at once | title is 17x richer in colour than gameplay, *and* coverage is exactly **1.0000** for logos, title, intro and gameplay alike — a bright paper page with no true black anywhere |
| **Space Adventure Cobra** (`PPSA17337`) | coverage, **inverted** | gameplay coverage **0.8152-0.9037** is *lower* than its menus' **0.9908-1.0000**, because the frame's bottom is a dark walkway underside |

Two rules follow that the earlier pair does not give you:

- **A high coverage floor is not the safe direction.** Greak needed 0.9 because
  coverage had to separate a letterboxed cinematic. Cobra is the mirror image: the
  same reflex would have rejected the very state its guard exists to protect. Set
  the floor from *this* title's measured range and direction, never from a habit.
- **When both floors are blind, say so in the `_note` and set them as pure
  gross-collapse floors** rather than tuning them to separate two valid scenes.
  GRIS's `min_colors` is 1200 against a 2,039 gameplay floor and a 38,183 title;
  it is there to catch black output, and nothing else.

### Score one run against the *other* run's references before adopting a window

`verify` builds its references from **both** of its own runs, so a scene the title
**regenerates randomly** still passes: each run matches the half of the reference set
it contributed. Worms Armageddon's Quickstart does exactly this — two fresh-save runs
produced two different maps, each scoring 70/70 against the combined set and looking
like a healthy `CONTENT-STABLE` baseline. Scored **across** runs (run B against run A's
references only) the same window reaches at most **0.6567**, with **0 of 70** clearing
0.85. Such a guard is adopted green and fails on the first `check`.

So `verify` cannot answer "is this window stable"; it is not asking that question. Run
the profile twice and score one against the other. The same test on a route into a
fixed Training map returned **0.9612-1.0000, 47/47** — same test, opposite answer, and
the reason `worms-armageddon-gameplay` guards a Training level rather than the
published Quickstart match. Recorded as trap 36 in `docs/GAME_COMPAT_ORCHESTRATION.md`.

## New Or Changed Baselines

Baseline evidence requires visual review even though routine regression runs do
not. This prevents checksums or thresholds from blessing black output or the
wrong scene.

1. Write the whole candidate entry, **including `_note`**, and set `review` to
   `pending`. Then run `snapshot.py verify NAME`.
2. Inspect every image retained from both independent runs in
   `tools/snapshot/review/NAME/`. Confirm the intended gameplay state, expected
   layers, and progression across multiple timestamps.
   Use the adjacent `runN-evidence.json` when a bad frame needs correlation
   with its flip count, front-buffer index, or renderer publication.
3. For content mode, choose a conservative `min_colors`, require at least two
   qualifying frames, and add `min_pixel_changes` for moving routes.
4. Run `snapshot.py update NAME --reviewed` only after every image is accepted.
   Add `--verified-by=human` **only if a person looked at the images**; see
   *Who verified it, and when* below. Content mode records reviewed luminance
   references for SSIM plus a conservative non-black coverage floor; exact mode
   records the identical pixel hash. Never use exact hashes for threaded
   gameplay merely because one run happened to be stable.
5. Replace the generic `review` string that `update` wrote with the factual note
   describing what was actually inspected.
6. Run `snapshot.py check NAME` after approval.

See `README.md` for every manifest field and the current title matrix.

### Who verified it, and when

Two structured fields record the *provenance* of a baseline, as data rather than as the English
inside `review`:

| field | shape | written by |
| --- | --- | --- |
| `verified_at` | ISO 8601 UTC instant, `2026-09-02T12:34:56Z` | `update NAME --reviewed` |
| `verified_by` | `agent` or `human` | `update NAME --reviewed` |

**`agent` is the default; `human` must be asserted.** The tool cannot observe who ran it, so it has
to be told: `snapshot.py update NAME --reviewed --verified-by=human`. Nothing infers `human` from a
TTY, from `$USER`, from an interactive terminal, or from the absence of an environment variable —
every one of those is a proxy for "somebody was probably there", and a proxy is exactly what would
make the field a lie. The asymmetry is deliberate and is the whole point: a `human` that never
happened is far worse than an `agent` that understates a real review, because the only reason to
record the actor at all is so a human-reviewed baseline can be trusted further than a
machine-approved one. **So if you are an agent, do not pass the flag — not even when a person asked
you to run the command.** They asked for the run; they did not look at the images.

**Only `update --reviewed` stamps.** `check` never does, and must never be made to: checking a
baseline is not verifying it, and a routine regression run that refreshed the date would launder a
baseline nobody re-read into one that looks freshly reviewed. `verify` does not stamp either — it
writes no manifest at all, and it runs *before* the images have been inspected, so a stamp there
would record a review that has not happened yet.

**Absence means "never recorded", and that is the useful signal.** The entries that predate the
fields carry neither, and nothing backfills them — a date recovered from `review` prose or from git
history would read as a recorded fact while being an inference, which is worse than no date at all.
`snapshot.py list` prints the column, so a never-verified or long-stale baseline is visible without
booting anything:

```
  guard                      last verified          configuration
  some-gameplay              NEVER VERIFIED         dump=PPSAxxxxx-app0 min_colors=12000 min_frames=20 min_changes=20 references=16
  other-gameplay             2026-08-31 human    1d dump=PPSAxxxxx-app0 min_colors=12000 min_frames=20 min_changes=20 references=16
  third-gameplay             2026-07-23 agent   40d dump=PPSAxxxxx-app0 min_colors=12000 min_frames=20 min_changes=20 references=16

  3 guards: 1 never verified, 1 agent-verified, 1 human-verified
```

`list` also takes guard names, so `snapshot.py list gris-gameplay` answers "when was this last
looked at" for one entry.

Two mechanical notes. Both fields are excluded from `entry_fingerprint`, like `review`, because
`update` writes them — hashing them would make a second `update NAME --reviewed` refuse its own
candidate with *"snapshot configuration changed after verify"*. And `update` now requires at least
one guard name: an unnamed `update --reviewed` would sweep the whole manifest and stamp every entry
with a verification nobody performed.

### Three ordering traps that silently produce a bad baseline

Each of these leaves a guard that *looks* adopted. None of them fails loudly, so
follow the order above rather than the obvious one.

- **The factual `review` note must be written after `update --reviewed`, not
  before.** `approve_content_candidate` unconditionally overwrites `review` with
  a generic "Approved N composited images ... visually confirmed" string. A note
  written before approval is destroyed by the approval itself, and the guard then
  carries boilerplate that records no evidence while still satisfying `check`'s
  "not pending" test.
- **`_note` must be set before `verify`, because it is part of the candidate
  fingerprint.** `entry_fingerprint` excludes only `hash`, `dims`, `review`,
  `structural_references`, `perceptual_references`, and `min_nonblack_ratio`;
  every other key, `_note` included, is hashed. Adding or editing `_note` after
  `verify` makes `update` refuse with "snapshot configuration changed after
  verify", costing a full two-capture rerun.
- **`min_content_match_ratio` applies to every frame in the window, not to the
  qualifying ones.** The requirement is
  `max(min_structural_matches, ceil(total_window_frames * ratio))`, so at the
  0.75 default, three quarters of *all* window frames must clear both SSIM and
  non-black coverage. A window that merely contains good gameplay but also
  straddles a load, a fade, or a transition will fail on a perfectly healthy run.
  Place the window entirely inside the settled state and confirm the margin with
  `profile_route.py` instead of assuming it.
  The rule is enforced at **two independent sites**, so raising
  `min_structural_matches` alone does not escape it: `content_result` derives
  `required_matches` from `len(summary["records"])` for SSIM, and derives
  `required_nonblack` the same way from `min_nonblack_matches` for coverage. A
  window sized for one is automatically sized for the other.
- A phase-variable checkpoint may opt into `min_consecutive_content_matches`
  instead of the whole-window ratio; configuring both is an error. This is not
  permission to count scattered matches: every adjacent sampler index in the
  required run must jointly clear colour, SSIM, non-black coverage, and
  dimensions, and an omitted index breaks the run. `min_pixel_changes` is scoped
  to that identified run, so unrelated boot animation cannot hide a frozen
  checkpoint. Existing reviewed structural references are required to seed
  plateau identification. `verify` then
  cross-scores each run using only the other run's identified plateau references
  before saving plateau-only evidence and a candidate. Inspect the whole profile
  and score explicit off-scene negatives anyway; a long stable menu, logo, or
  results screen can otherwise become the wrong plateau.

## Environment

- `PROSPER_GAME_ROOT`: directory holding `*-app0` dumps; defaults to the repo.
- `PROSPER_BOOT_TRACE`: `boot_trace` path; defaults to `build-linux/boot_trace`.
- `PROSPER_SCREENSHOT`: presented-capture frontend; defaults to
  `build-linux/screenshot`.

**A guard inherits the shell that launched it.** `snapshot.py` builds the child environment as
`env = dict(os.environ)` and *then* applies the guard's own overrides, so **every `PROSPER_*` you
have exported goes into the run** — the guard only controls the handful of variables it sets itself.
That is useful (it is how a single behavioural line can be A/B-ed through a temporary gate without
maintaining two builds) and it is a foot-gun in exactly the same breath: with several lanes on one
machine, one agent's exported diagnostic can redden another agent's gate with no trace in either
one's notes. `PROSPER_AVPLOG` in a shell that then runs a guard adds per-call logging to the hot path
of a timing-sensitive capture; `PROSPER_NULL_PAGE`, `PROSPER_HWBP` and the render-scale knobs change
the run outright.

**Two commands, before quoting any guard result:**

```bash
env | grep '^PROSPER'                                   # your shell, before you start
tr '\0' '\n' < /proc/<capture-pid>/environ | grep '^PROSPER'   # what the run actually got
```

Record the second beside the result. Recorded as a **near miss** rather than a scar: on 2026-09-05
(#2899) a lane checked this on a live 320 s arm it had already protected with whole-box exclusivity
and found it clean — but it had been setting `PROSPER_AVPLOG` all evening, in per-command `env`
prefixes rather than exports, and nothing but that habit stood between it and a contaminated arm.
