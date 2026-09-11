# `diagnostics` — observation only

**Nothing here is on a rendering path AT DEFAULT SETTINGS**, and that is the intended property:
with nothing armed, code in this folder can be removed or rate-limited without changing a single
rendered pixel.

**Three entries break the stronger form of that rule and you must know about them.**
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

`gpu_memory_budget` is the third, and it breaks a different half of the rule: it is **on by
default**, so unlike everything else here it executes on an ordinary boot with nothing armed, and
unlike everything else here it is wired into every shipped device allocation and free — the
renderer's included. It still changes no pixel: each wrapper is a pass-through plus a counter, and
the counters feed nothing but `fprintf`. That default is the entire point rather than an oversight:
it exists because this machine hard-froze five times under ordinary prosper GPU work and produced
three wrong published root causes, and a diagnostic that must be switched on is one nobody had
switched on for the run that mattered (#3533). `PROSPER_GPU_MEM_LOG=0` silences it.

- `diagnostic_selectors` — choosing what to observe.
- `gpu_memory_budget` / `gpu_memory_budget_vk` — how many bytes prosper itself holds on each device
  heap, against that heap's size, with a peak. The split is deliberate: the first header is free of
  Vulkan types so it can be included anywhere, and the second carries the two wrappers every shipped
  call site uses, `allocate_device_memory` and `free_device_memory`. **Never call
  `vkAllocateMemory`/`vkFreeMemory` directly in shipped code** — an unwired allocation under-reports,
  and an unwired free never gives its bytes back, so the figure climbs and reads as a leak that is
  not there. `tools/vkmem_coverage.py` finds any site that has slipped past the wrappers and runs as
  the `vkmem_coverage` ctest case, because nothing in the compiler enforces the convention.
  Read the header's caveat before quoting a number: it sees **prosper's own allocations only**, not
  other processes, the compositor, or driver overhead, so on an integrated GPU — where the desktop
  shares these heaps — "prosper holds far less than the heap" is not evidence that an allocation will
  succeed. `VK_EXT_memory_budget` is the honest upgrade and is not enabled on the device today.
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
