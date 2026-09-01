# `diagnostics` — observation only

**Nothing here is on a rendering path AT DEFAULT SETTINGS**, and that is the intended property:
with nothing armed, code in this folder can be removed or rate-limited without changing a single
rendered pixel.

**One entry breaks the stronger form of that rule and you must know about it.**
`compute_parent_walk_suspicious()` in `compute_parent_walk.cpp` reaches the site in
`execute/gpu_executor.cpp` that prints `[compute-parent-walk] DIAGNOSTIC-ONLY skip suspicious`
and then **does not run the dispatch** — grep `DIAGNOSTIC-ONLY skip suspicious`, which is unique
in source. Do not extend it with the next word: the C++ literal is wrapped across two lines
there, so the longer form matches nothing. It is env-gated, so a default boot is unaffected, but an armed run is not
merely observing. `compute_parent_walk.hpp` states this boundary; do not read the folder name as a
guarantee.

- `diagnostic_selectors` — choosing what to observe.
- `diag_ratelimit` — rate limiting. **Check a diagnostic's rate limit before quoting its volume as a
  frequency**; several phantom findings came from reading a capped count as a real one.
- `watch_list`, `compute_tree_watch`, `compute_parent_walk` — watching addresses and walking compute
  parentage.
- `shader_dump_filter` — `PROSPER_SHADER_DUMP_PROGRAM`, which narrows `PROSPER_SHADER_DUMP_SUCCESS`
  to named guest programs. It fails **open** where the skip selectors fail closed, and the header
  explains why: an empty dump directory reads as "that program never compiled".

Two standing cautions, both learned expensively:

- **Prefer instruments that detect their own invalidity.** A diagnostic that can fail silently will,
  and its silence reads as evidence.
- **A count is only as good as its population.** Know whether a diagnostic fires per dispatch, once
  per program, or only on some sub-population — several published figures were off by two orders of
  magnitude because the answer was assumed rather than read.
