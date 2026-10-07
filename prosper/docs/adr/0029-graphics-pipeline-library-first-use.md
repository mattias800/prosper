---
kind: adr
status: proposed
date: 2026-10-07
---

# ADR 0029: Graphics pipelines without first-use stutter, via pipeline libraries

## Context

ADR 0014 (proposed) and its rule `PERF-P7` say that a pipeline the cache lacks is compiled off the
submit thread, "from libraries where the device supports them". Neither says how the libraries are
built, when they are linked, or how the first-use cost is measured. This ADR is that refinement for
graphics pipelines; it does not replace ADR 0014's persistent-cache and pre-warm points.

What prosper does today, read from source on `origin/main` at `aa7d3b570`:

- **A graphics pipeline miss compiles inline, on the thread realizing the draw.** The live backend
  (still `tests/fixtures/render_runner.h`; ADR 0004 moves it) creates the shader modules
  (`render_runner.h:13336-13338`), then calls `vkCreateGraphicsPipelines` against the device-lifetime
  `ctx.driver_pipeline_cache` under `graphics_driver_cache_mutex()` (`:13372-13374`, mutex at
  `:1144`). Nothing is deferred: the draw waits for the driver's full compile.
- **A miss is never answered with a placeholder.** A failed creation drops the draw and counts it as
  `DrawDrop::PipelineCreation` (`:13386`); a rejected SPIR-V module drops it as
  `ShaderRejected` (`:13355`). Both are fail-visible drops, not a policy of skipping slow
  compiles.
- **The cost is already measured per miss.** `Cost::PipelineCreate` wraps exactly the creation call
  plus the cache-lock wait (`src/diagnostics/perf/perf_ledger.hpp:60-64`, scope at
  `render_runner.h:13370`), separate from `Cost::ShaderCompile`, the RDNA2 -> SPIR-V recompile
  (`perf_ledger.hpp:57-59`). The `shader-compile` alarm (`src/diagnostics/perf/perf_alarm_rules.cpp:617`)
  fires only after a sustain window (`:170`), so a single first-use hitch does not trip it; the F8
  `.prperf` capture records the individual cost.
- **Compute is separate.** Compute pipelines are created in `frontends/shared/live/live_compute.cpp`
  (`:3213`, `:11521`) under their own `pipeline_cache_mutex` (`:1279`), and have no stage split to
  exploit. This ADR is about graphics.
- **Neither extension is used.** A repository-wide search for `graphics_pipeline_library`,
  `GRAPHICS_PIPELINE_LIBRARY`, `shader_object` and `vkCreateShadersEXT` finds only ADR 0014's prose.

So `PERF-P3` holds only after warm-up, and every new state combination a title reaches costs a full
driver compile in the frame that first needs it. The recompiler output is per guest program, but the
`VkPipeline` is per program *and* per fixed-function state (blend, depth, topology, render-target
formats, vertex input), so the miss count grows with the product, not the program count.

### External reference (not verified in source)

From the maintainers' public descriptions, not from reading DXVK's code: DXVK, when
`VK_EXT_graphics_pipeline_library` (GPL) is available, compiles a **pre-rasterization** library
(vertex/geometry stages) and a **fragment-shader** library when the application creates the shader,
on worker threads. At draw time it builds or reuses small **vertex-input** and **fragment-output**
interface libraries and **fast-links** the four (no `LINK_TIME_OPTIMIZATION` flag), which is cheap
enough to do in-frame. It then queues a full, optimized pipeline for the same state on a background
thread and swaps it in when ready. Without GPL it falls back to its older path (state cache and
async compile). `VK_EXT_shader_object` is the alternative model: stages are created as independent
`VkShaderEXT` objects and nearly all state is dynamic, so there is no pipeline object to miss.
Driver support as commonly reported: NVIDIA and RADV expose both; lavapipe, which CI runs on, has
exposed both in recent Mesa releases. `CONFIDENCE: MED` on the support matrix; it is a runtime
query, and the fallback path below means no claim here is load-bearing for correctness.

## Decision

1. **Stage libraries are compiled when a guest program is recompiled, off the submit thread.**
   When the recompiler produces SPIR-V for a vertex (or ES+VS / mesh) program or a pixel program,
   a worker creates the matching GPL library -- pre-rasterization for the former, fragment-shader
   for the latter -- keyed by the recompile identity (`bd.vs_identity` / `bd.fs_identity`) plus the
   pipeline layout it was compiled against. Where the fragment library's SPIR-V depends on draw
   state (for example a lowered fragment variant), that dependency is part of its key.
2. **At first draw, prosper fast-links.** Vertex-input and fragment-output interface libraries are
   small, cached by their own state keys, and created on the realizing thread if absent. The four are
   linked without link-time optimization. The draw is recorded with the linked pipeline: no
   stutter-sized wait, **no skipped draw** (`FAIL-1`, `PERF-P7`). If a stage library is not ready
   yet, the thread waits for that library -- counted, never dropped.
3. **An optimized pipeline replaces the fast link in the background.** The same create info, with
   `LINK_TIME_OPTIMIZATION` and `RETAIN_LINK_TIME_OPTIMIZATION_INFO` on the libraries, is compiled on
   a worker into the existing `VkPipelineCache`; the persistent pipeline map swaps the handle when it
   lands, and the fast-linked pipeline is retired after the frames that reference it complete.
4. **Fallback is today's path.** Without `graphicsPipelineLibrary` (or with
   `graphicsPipelineLibraryFastLinking` false) prosper creates monolithic pipelines exactly as now.
   The selection is a *host-capability* switch: a device query, with one `PROSPER_*` override to
   force the fallback for A/B and bisection.
5. **The first-use hitch is measured, not asserted.** Instrument: `Cost::PipelineCreate` per miss
   from an F8 `.prperf` window around the first appearance of new state (`tools/perf/
   performance_capture_report.py`), plus a new count of fast links and of draws that waited on a
   library. Claim shape: max and p99 per-miss `PipelineCreate` and the worst frame time in the window,
   same binary, switch off vs on, on the reference workloads of `.claude/skills/perf-change/`:
   *Grand Theft Auto V* (`scripts/gta5/reach-performance-story.pad`), a second 3D title
   (*Outer Wilds* route), and *The Messenger* as the 2D control. Measured cold (no disk cache),
   since a warm cache hides exactly the cost in question.

Adds spec rule `PERF-P10`.

## Consequences

- **Memory.** Each program keeps a stage library alive alongside any monolithic or optimized
  pipelines, and a fast-linked pipeline lives until its optimized replacement retires it. Library
  count is bounded by recompile identities, which the recompiler cache already bounds; the eviction
  rule for the persistent pipeline map must evict libraries too, and must not destroy one a pending
  background link still references.
- **Driver variance.** Fast-link cost, and whether a fast-linked pipeline runs measurably slower than
  an optimized one, differ by driver. NVIDIA's Windows driver has already faulted in pipeline
  creation on one SPIR-V shape (`tests/gpu/test_rdna2_spirv_struct.cpp:4311`) and in its cache-load
  path (ADR 0014), so the library path must be A/B-able per device from day one.
- **Validation.** Libraries need the pipeline layout fixed at library time (or
  `INDEPENDENT_SETS`), which constrains how prosper's descriptor layouts are chosen per draw; that is
  the largest unknown. CI's lavapipe run exercises the library path only if lavapipe exposes the
  extension at the CI Mesa version; the fallback path must stay covered by a forced-fallback test.
- `PERF-P3` becomes true for first use as well as after warm-up, on capable devices.

## Alternatives considered

- **Only a larger or pre-warmed `VkPipelineCache`.** Helps second and later launches (ADR 0014,
  points 3-4) but not the first time a state combination appears, which is the stutter in question,
  and disk loading is still opt-in for a driver crash.
- **Async full compile and skip the draw until ready.** Removes the hitch by dropping guest content;
  rejected by `FAIL-1`, as ADR 0014 already records.
- **`VK_EXT_shader_object`.** No pipeline object at all, so no miss. Rejected as the first step
  because it moves every fixed-function state to dynamic state at once, touching all of draw state
  translation, and gives drivers less to optimize; it remains the candidate if GPL's layout
  constraints prove unworkable (Open questions).
- **Ubershader / generic fragment output.** Considered and not pursued: it needs a second recompiler
  output path for every program.

## Migration order

1. Expose the device query and the fallback override; count misses, fast links, library waits.
2. Record a cold baseline on the three reference workloads with the extension absent (step 5's claim).
3. Pre-rasterization and fragment libraries at recompile time, on a worker; nothing links them yet.
4. Fast link at first draw, monolithic pipeline on any library failure; A/B against step 2.
5. Background optimized link and handle swap; A/B fast-linked vs optimized steady-state frame time.
6. Once the A/Bs hold on NVIDIA and RADV, record the verdict in
   `docs/gpu/GRAPHICS_PIPELINE_CACHE.md` and on the tracking issue.

## Open questions

- Does prosper's per-draw pipeline layout selection fit GPL's layout rules, or does it need
  `INDEPENDENT_SETS` and a layout redesign first? This decides GPL vs shader object.
- Which parts of the current pipeline key are fragment-output or vertex-input state, and which leak
  into the shader stages (lowered fragment variants)? Only the former link cheaply.
- Should the `shader-compile` alarm gain a per-event rule, so a single long first-use miss is
  reported without a sustain window?
- How much of today's per-miss cost is `graphics_driver_cache_mutex` contention rather than compile?
  The ledger includes the lock wait, so the baseline must split them before attributing a gain.

## Approval

Requires the project owner's acceptance.
