# `present` — scanout, and how fast it is going

Two modules, and they answer different questions.

`videoout_present` — taking a finished frame to the display path. The last stage, and therefore the
one where an upstream failure is *observed* rather than caused. A black or stale frame here is almost
never a bug here. Check that something was drawn, and that the target being presented is the target
that was drawn to, before looking for a defect in this folder.

`present_frame_rate` — the framerate the project quotes. It sits here rather than in `diagnostics/`
because it is fed by the publish path itself: every accepted publication is signed and counted, for
every title, whether or not anybody asked for a number. Its header is long on purpose and is the
thing to read first — the short version is that a *present* count reads full speed for a completely
frozen title, so this module counts publications and content-distinct publications separately, and
every consumer reports both.

Two vocabularies live here and mixing them is the recurring mistake. A **run** rate comes from
`frame_rate_since_first_publication` and carries the headline (`typical_fps`) and its qualifier
(`active_fraction`). A **window** rate comes from `frame_rate_between` and carries neither: the
interval histogram behind them is cumulative for the process and cannot be differenced. Anything
reading `active_fraction` must check `active_fraction_measured` first, because `0` otherwise means
both "this title produced nothing" and "nobody filled this in".

## Ruled out

- **Filling a window's `active_fraction` by differencing the two snapshots' `active_seconds`**
  (option 1 on #3027) — falsified on the counter itself. `active_seconds` is a whole-run
  recomputation against the *current* median interval, not an accumulator, so the cutoff moves as
  the run goes on: 201 publications at 1 fps followed by 1000 at 60 fps take it from 200.000 s to
  16.650 s, making the difference **negative** (-183.4 s across a 16.7 s window). It is also wrong
  where it stays positive, and that is the dangerous half — a window holding 6000 unbroken frames at
  60 fps differences to **1.0% active**, and one holding 100 frames at a healthy 1 fps to **0.02%**,
  both of which read as the "produced nothing" shape the metric exists to flag. Pinned by
  `a_windowed_active_share_cannot_be_differenced` in `tests/gpu/present/`.
- **Any windowed statistic separating "a static menu" from "a title that produced nothing"** — not a
  measurement problem, an information one (#3027). Within one second the two are identical by
  construction, because in both cases nothing changed during that second. The answer is a property
  of the run, so `unchanged_picture` takes the cumulative snapshot alongside the window; do not go
  looking for a cleverer window statistic.
