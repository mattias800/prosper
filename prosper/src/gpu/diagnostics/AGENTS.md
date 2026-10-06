# `diagnostics` — observation only

**Nothing here is on a rendering path AT DEFAULT SETTINGS**, and that is the intended property:
with nothing armed, code in this folder can be removed or rate-limited without changing a single
rendered pixel.

**Four entries break the stronger form of that rule and you must know about them.**
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

The fourth default-path entry is `fence_build_journal`: it samples a label at packet build
time and retains diagnostic metadata. It does not validate or complete a guest fence.

- `diagnostic_selectors` — choosing what to observe.
- `fence_build_journal` — default-on, bounded build-time label observations used by the command
  processor's diagnostics. It never rewrites a packet or completes a fence. Lookup requires a
  complete fault-safe eight-byte sample; unavailable and colliding replacements cannot expose
  a previous sample. Each direct-mapped slot has its own lock, with process-lifetime storage.
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
  succeed. `VK_EXT_memory_budget` is the honest upgrade; since #3873 it is enabled when advertised
  and the texture-cache budget (`src/gpu/memory/`) reads it, but this instrument still counts only
  prosper's own allocations.
- `memory_placement_log` — `allocate_gpu_only_memory` / `allocate_gpu_only`, the call every
  GPU-only renderer allocation makes: gpu/memory/memory_type_select's candidates in order, retried
  on `VK_ERROR_OUT_OF_DEVICE_MEMORY` (#3897), plus default-on `[mem-placement]` lines — one per
  allocation class and memory type (so a class that later lands elsewhere is visible), a `FELL
  BACK` line when VRAM ran out, an `ALLOCATION FAILED` line when every type did — and the
  `gpu-memory-off-device` perf-alarm counter. The lines are observation; the choice and the retry
  are not. `NOT DEVICE_LOCAL` there is a discrete-GPU performance problem worth an issue. Its
  gated `PROSPER_GPU_MEM_FORCE_OOM` fails device-local attempts without calling the driver — like
  `draw_program_skip`, an armed run is not merely observing. Do not allocate a GPU-only resource
  any other way: a site that calls `allocate_device_memory` directly gets no fallback.
- `geometry_probe_arming` — whether `PROSPER_GEOM_PROBE` may answer at all: does the module the
  backend is about to hand Vulkan actually declare the transform-feedback capture? It is the
  worked example of the first standing caution below. Without it the probe armed on a shader it
  could not capture, read back a zero counter, and printed "the draw produced no primitives" — a
  wrong answer rather than a missing one, on a draw that did produce primitives (#3248). Note what
  it tests: the WORDS, not the environment variable that was supposed to have caused them. The env
  var says what was asked for; the two diverged.
- `diag_ratelimit` — rate limiting. **Check a diagnostic's rate limit before quoting its volume as a
  frequency**; several phantom findings came from reading a capped count as a real one.
- `gpu_breadcrumbs` / `gpu_breadcrumbs_vk` — `PROSPER_GPU_BREADCRUMBS`: where did the GPU STOP? A
  marker is written before and after every draw and dispatch (`VK_AMD_buffer_marker`, or
  `VK_NV_device_diagnostic_checkpoints`), and after a device loss the last markers that reached
  memory name the window of sites the GPU stopped inside, beside `VK_EXT_device_fault`'s own account.
  Off by default and free when off; **an unarmed run prints a one-line "not armed" hint on a loss**,
  not silence. The core is Vulkan-free and tested on synthetic streams; the emitter takes an
  injectable dispatch table because stock lavapipe exposes neither extension, so CI proves the
  emitter/resolver contract and **cannot** prove a real driver executes the writes before a hang.
  **A verdict is a WINDOW, not a culprit**: "started" and "finished" are different moments, ids are
  assigned in recording order (not submit order), and unmarked work (transfers, barriers, presents) is
  invisible. Read the header before quoting one. It exists because a loss names the submission that
  OBSERVES it, not the one that caused it (instrument trap 170).
- `gpu_labels_vk` — `PROSPER_GPU_LABELS`: a `VK_EXT_debug_utils` label around every draw and dispatch
  whose text is `format_breadcrumb_site`, **the same spelling a breadcrumb verdict uses**, so a
  RenderDoc/RGP marker can be matched to a `[gpu-breadcrumb]` line by eye. It is the command-label
  half of naming; `vk_object_names.hpp` is the object half and still names shader modules only. Off
  by default and free when off; an absent extension makes every call a no-op. Numbers and hashes only.
  Tested against recording entry points, so what is proved is what is recorded, not how a capture
  tool displays it.
- `watch_list`, `compute_tree_watch`, `compute_parent_walk` — watching addresses and walking compute
  parentage.
- `draw_program_skip` — naming a graphics shader program, to census it or to decline every draw
  that uses it.
- `refused_shader_dump` — on by default: each distinct refused shader (vertex, pixel or compute) is written once per run, with an index line naming its first unsupported instruction, into `refused_shaders_*/` under `PROSPER_CAPTURE_DIR`. Bounded and deduplicated by code hash, so it costs nothing on a clean run. `PROSPER_SHADER_DUMP` remains the unbounded opt-in.
  A draw the **owned-wave gate** refuses is kept the same way (#4555). Nothing was recompiled for it, so the `first_bad_*` fields on its index line are only the generic coverage census and do not say why the draw was lost; the line ends in `refusal=<reason>`, which does. That program is the one to run `shader_inspect --raw-wide-proof` on: the `dropped-draws` alarm files these drops under `shader-recompile/*`, and the first question is whether the program belongs on the owned-wave path at all.
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

`draw_disposition`'s exit aggregate is a `RUN SNAPSHOT` with `quiescence=unverified` (#3973).
Its counters are loaded independently, and no reporting caller proves that CPU pass preparation
has stopped. In particular, prosper-app drains queue calls but leaves its guest thread unjoined.
A balanced snapshot does not prove completeness; `snapshot-delta` is an observed difference that
may include an in-flight pass or interleaved loads. Completed-pass `UNACCOUNTED` reports and the
`unaccounted-draws` performance alarm keep their existing accounting contract and are unchanged.

`vk_object_names.hpp` is the odd one out here and worth a line: it is not a census or a trace but a
naming shim, giving guest shader modules a `vkSetDebugUtilsObjectNameEXT` name so an external tool —
RenderDoc, RGP, the validation layer — identifies them by something a reader recognises instead of a
per-run handle. It lives in this folder because its audience is the same (somebody diagnosing a frame),
and it obeys the same two cautions: it resolves its entry point once, no-ops when the extension is
absent, and never propagates a failure, because a run that died for want of a debug name would be a
worse instrument than no names at all.
