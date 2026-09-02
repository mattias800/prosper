# `diagnostics` — observation only

**Nothing here is on a rendering path AT DEFAULT SETTINGS**, and that is the intended property:
with nothing armed, code in this folder can be removed or rate-limited without changing a single
rendered pixel.

**Two entries break the stronger form of that rule and you must know about them.**
`compute_parent_walk_suspicious()` in `compute_parent_walk.cpp` reaches the site in
`execute/gpu_executor.cpp` that prints `[compute-parent-walk] DIAGNOSTIC-ONLY skip suspicious`
and then **does not run the dispatch** — grep `DIAGNOSTIC-ONLY skip suspicious`, which is unique
in source. Do not extend it with the next word: the C++ literal is wrapped across two lines
there, so the longer form matches nothing. It is env-gated, so a default boot is unaffected, but an armed run is not
merely observing. `compute_parent_walk.hpp` states this boundary; do not read the folder name as a
guarantee.

`draw_program_skip` is the second, and it is deliberate rather than incidental: armed with
`PROSPER_SKIP_DRAW_PROGRAM=0xADDR`, the live renderer withholds every draw using that shader program
from the GPU. It is the graphics counterpart of `PROSPER_COMPUTE_SKIP_PROGRAM`, and it exists
because a draw that hangs the device cannot be studied any other way — the context is gone before
any other instrument reports. Unset, it costs one `empty()` test per draw and changes nothing.
`draw_program_skip.hpp` states the four limits a reader of a skipped run cannot see in the output;
read them before quoting a result. Its companion `PROSPER_DRAW_PROGRAM_CENSUS` is observation only.

- `diagnostic_selectors` — choosing what to observe.
- `geometry_probe_arming` — whether `PROSPER_GEOM_PROBE` may answer at all: does the module the
  backend is about to hand Vulkan actually declare the transform-feedback capture? It is the
  worked example of the first standing caution below. Without it the probe armed on a shader it
  could not capture, read back a zero counter, and printed "the draw produced no primitives" — a
  wrong answer rather than a missing one, on a draw that did produce primitives (#3248). Note what
  it tests: the WORDS, not the environment variable that was supposed to have caused them. The env
  var says what was asked for; the two diverged.
- `diag_ratelimit` — rate limiting. **Check a diagnostic's rate limit before quoting its volume as a
  frequency**; several phantom findings came from reading a capped count as a real one.
- `watch_list`, `compute_tree_watch`, `compute_parent_walk` — watching addresses and walking compute
  parentage.
- `draw_program_skip` — naming a graphics shader program, to census it or to decline every draw
  that uses it.
- `shader_dump_filter` — `PROSPER_SHADER_DUMP_PROGRAM`, which narrows `PROSPER_SHADER_DUMP_SUCCESS`
  to named guest programs. It fails **open** where the skip selectors fail closed, and the header
  explains why: an empty dump directory reads as "that program never compiled".
- `link_list_census` — `PROSPER_DRAW_LINKSCAN`, a CPU-side census of the linked lists a graphics
  draw's scalar buffers contain, taken from the exact bytes prosper is about to upload. Observation
  only. It is the graphics counterpart of `PROSPER_COMPUTE_PARENTSCAN`, and the one thing to know
  before reading a result is the model it walks: **an out-of-range scalar buffer load returns
  architectural zero, and zero is a link, not an exit.** So an unpopulated (all-zero) pool is not an
  empty list — it is an infinite one, a self-loop at record 0 that no trip bound can end. The
  histogram is printed beside the walk because "the buffer is all zeros" and "the walk never
  terminates" are the cause and the symptom, and only the first is actionable.

Two standing cautions, both learned expensively:

- **Prefer instruments that detect their own invalidity.** A diagnostic that can fail silently will,
  and its silence reads as evidence.
- **A count is only as good as its population.** Know whether a diagnostic fires per dispatch, once
  per program, or only on some sub-population — several published figures were off by two orders of
  magnitude because the answer was assumed rather than read.
