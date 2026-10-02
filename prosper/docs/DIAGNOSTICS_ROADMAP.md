# Diagnostics and measurement roadmap

**Read this before building a new `PROSPER_*` diagnostic.** It is the plan for two linked tracks:

- **Triage:** make a failure on a title (device loss, crash, wrong shader value, missing guest API)
  diagnosable from local artifacts in one run, instead of by bisection.
- **Measurement:** make a performance claim a mechanical comparison against a stored baseline, with
  refusal when the run conditions differ, instead of a figure quoted from whichever harness was open.

It is a plan, not a status report. The "exists" and "missing" claims were checked against the tree on
2026-10-02 (grep over `src/`, `frontends/`, `tools/`, three corrected in review); the evidence is
stated per item. Re-verify before
building: this tree moves fast, and a stale gap list is the failure this document is written to avoid.

Companion docs: `GPU_PROFILING_EXTERNAL.md` (vendor tools that need no prosper change),
`DIAGNOSTIC_GATE_AUDIT.md`, `src/diagnostics/AGENTS.md`, `src/diagnostics/perf/AGENTS.md`,
`tools/AGENTS.md`, `RENDERER_PERFORMANCE_2026_07.md`.

## 1. Rules every item here must follow

These are the contract. An item that cannot meet them is redesigned, not waived.

1. **Free vendor tooling first.** RGP, RenderDoc, `radeontop` and the validation layer already work on
   prosper with no code change. Build a `PROSPER_*` switch only for what the guest-facing layer knows
   and the GPU vendor cannot see: guest packet, guest PC, guest shader address, HLE call.
2. **Reachable without a rebuild when it answers "where did it stop?".** A hang diagnostic that needs a
   special build is absent on the day it matters (`src/diagnostics/AGENTS.md`, the boot-phase history).
   Heavier reports may stay behind `PROSPER_DIAGNOSTICS`.
3. **Zero cost when off.** Function pointers are not loaded, no timestamps are written, no strings are
   formatted. A switch that taxes the default path is a performance regression with a label on it.
4. **Observer only.** Nothing here may change what the guest sees or what the renderer produces. A
   diagnostic whose output depends on its own presence is a defect (see trap-style findings in
   `GAME_COMPAT_ORCHESTRATION.md`).
5. **A diagnostic's lever must show it moved.** Each item ships a positive control constructed by hand,
   outside the thing that produces the measurement, and a test that fails if the instrument goes silent
   (charter: "a control drawn from the same source tests the discriminator, never the domain").
6. **Numbers and hashes only in anything shared.** Object names, labels, reports and logs that may
   reach an issue or PR carry numeric ids, hashes, module-relative offsets and counters. Never guest
   strings or game content. Captures, dumps, crash reports and minidumps stay in a gitignored local
   directory (`$HOME`, never `/tmp`: it is RAM-backed and shared; see the charter).
7. **Name the harness and the frame population.** `tools/screenshot` forces a readback and never calls
   `set_gpu_present_active`; a `fifo` run measures the display; `presented` counts re-served frames.
   Every figure states harness, present mode, resolution and `distinct` vs `presented`.
8. **Fail-visible, not fail-silent.** A diagnostic that cannot run says so once, loudly, rather than
   producing an empty report that reads as "no problem" (the trap-122 and trap-116 family).
9. **Test without a GPU where possible.** Hosted CI has no GPU and uses a software rasteriser with an
   8-lane subgroup. Logic (mapping, schema, arithmetic) is tested on synthetic inputs; extension
   behaviour is tested against a recording mock of the Vulkan dispatch, because lavapipe exposes neither
   `VK_AMD_buffer_marker` nor `VK_NV_device_diagnostic_checkpoints`. A real-driver run is a local
   step and is recorded in the PR, never claimed from CI.

## 2. What exists today (verified)

| Capability | Where | What it does not do |
| --- | --- | --- |
| Frame grab + offline replay | F9 `.prgbundle`, `tools/gpu_replay` (schedulable headless via `PROSPER_GRAB_BUNDLE_*`) | Captures rendered-frame bugs only; not CPU, logic or audio. |
| RenderDoc in-app capture | `frontends/shared/diagnostics/renderdoc_capture.hpp`, `PROSPER_RENDERDOC_AFTER_MS` / `_AT_FRAME` / `_AT_PAD_FLIP` (#3321) | Needs RenderDoc already injected; nothing bundled. |
| Bounded perf capture | F8 `.prperf`, `tools/perf/performance_capture_report.py` | Reports by time; six timestamp brackets per **compute** dispatch. |
| Always-on alarms | `[perf-alarm]`, `src/diagnostics/perf/` | Names a cost per window; no per-frame cause code. |
| Stage buckets | `PROSPER_RENDER_TIMING` | Setup/resource breakdown; not a per-frame guest/driver/GPU/wait/present split. |
| GPU submit index | `PROSPER_GPU_TIMELINE`, `tools/gpu_timeline` | Guest-submit index, **not** GPU pass timing. |
| Guest frame pacing | `tools/perf/flip_pacing_report.py` | Interval distribution of guest flips from `PROSPER_EVLOG`; names the limiter class. |
| Distinct-frame rate | `src/gpu/present/present_frame_rate.*` | Median interval, run average, active fraction. **No tail percentile and no `1% low`.** |
| Compute-dispatch GPU time | `frontends/shared/live/live_compute.cpp` | `vkCmdWriteTimestamp` around dispatches only; no graphics-pass timestamps. |
| CPU/GPU clock correlation | **missing**: no `VK_EXT_calibrated_timestamps` / `vkGetCalibratedTimestamps` use outside docs | M2 and M4 need it; add it with whichever lands first. |
| Validation layer on tests | `tools/vkval/` | Runs ctest under `VK_LAYER_KHRONOS_validation`; **not** a runtime key for a live title. |
| Vulkan object names | `src/gpu/diagnostics/vk_object_names.hpp` | Guest shader modules only. |
| VRAM budget | `src/gpu/diagnostics/gpu_memory_budget.hpp` | `VK_EXT_memory_budget` read; no per-run peak series. |
| Crash/fault handling | `src/host/image/exec_image_linux.cpp` fault handler | No flight recorder, no frame classification, no minidump. |
| Guest-side debugging | `boot_trace`, `PROSPER_HWBP`/`HWWATCH`/`PEEK`, `guest_bt`, `hang_probe`, `tools/re/xref.py` | Manual; each needs the fault already reproduced. |
| Run-to-run comparison | `tools/perf/compare_capture_pair.py`, `tools/snapshot` | Pair/shape comparison and image guards; **no baseline compare that refuses mismatched run conditions**. |

## 3. Triage track

Each item states the failure it removes the need to bisect, the design, acceptance criteria, the test
strategy, dependencies and what it deliberately does not do.

### D1. Device-loss triage: `VK_EXT_device_fault` and breadcrumbs

- **Evidence missing:** 0 hits for `device_fault` or `diagnostic_checkpoints` in `src/`, `frontends/`,
  `tools/`. (The `buffer_marker` hits are an unrelated "raw buffer marker" capture/recompiler concept.)
- **Failure removed:** a `VK_ERROR_DEVICE_LOST` today reports what was submitted, not where the GPU
  stopped. GPU hangs and stalls are among the most expensive bug classes here.
- **Design:** behind a `PROSPER_*` switch (off by default: markers cost GPU time).
  - AMD: `VK_AMD_buffer_marker` writes a marker before and after each draw and dispatch into a
    host-visible buffer.
  - NV: `VK_NV_device_diagnostic_checkpoints` sets a checkpoint per draw/dispatch and reads them back
    after the loss (`vkGetQueueCheckpointDataNV`); no buffer.
  - On loss, with `VK_EXT_device_fault` where supported: the fault report plus the last completed
    marker mapped back to **guest packet, draw index and pipeline key (hash)**, written to the log
    before the abort.
- **Acceptance:** marker/checkpoint value to packet/draw/pipeline mapping unit-tested on a synthetic
  stream with no device; emission tested against a recording mock of the Vulkan dispatch; one real AMD
  or NV loss reproduced locally and the report recorded in the PR.
- **Depends on:** a stable draw/dispatch identity (D8 shares it). **Out of scope:** vendor SDKs (Nsight
  Aftermath, Radeon GPU Detective) as linked dependencies; they are proprietary. Using them as external
  tools stays fine.

### D2. Crash report with a flight recorder

- **Evidence missing:** 0 hits for `minidump`; no ring buffer of recent diagnostic records. The fault
  handler exists; nothing classifies or symbolises what it catches.
- **Failure removed:** hand-run `guest_bt`/`xref` after every crash to learn which side faulted.
- **Design:** the handler writes one local report: exception, registers, a stack walk with each frame
  classified **guest image / HLE-prx / host** and given as module+offset, the last N ring-buffer
  records, and a minidump. A small offline `tools/symbolize.py` resolves module+offset to names from
  the module map the loader already has and from DWARF where built with `-g`; frames with no symbols
  stay module+offset. The ring buffer is fed by the structured sink (D10).
- **Acceptance:** ring wrap, frame classification and the map writer unit-tested; a forced-fault death
  test produces the report; the report carries no guest strings.
- **Depends on:** D10 for the ring feed (a minimal in-tree ring is enough to start). **Out of scope:**
  uploading reports anywhere. Minidumps contain guest memory and stay local; that is a deliberate
  decision, not an oversight.

### D3. Shader printf at a guest PC

- **Evidence missing:** 0 hits for `NonSemantic` / `DebugPrintf`.
- **Failure removed:** bisecting a wrong shader value with `PROSPER_DBG` and IR dumps.
- **Design:** `NonSemantic.DebugPrintf` (`VK_KHR_shader_non_semantic_info`, printed by the validation
  layer) injects a print of chosen registers after the instruction at a chosen guest PC, for a chosen
  shader. **Instrumented modules never enter any shader or pipeline cache** (the cache key includes the
  printf set, or the path is uncached).
- **Acceptance:** a golden test on a synthetic shader checks the injected SPIR-V and that the module is
  `spirv-val` clean; the cache-bypass has its own test.
- **Depends on:** D4 (validation layer runtime key) to surface the output. **Out of scope:** a
  source-level shader debugger. **Check first:** whether RGP/RenderDoc already answer the question
  (rule 1).

### D4. Validation layer as a runtime key for a live title

- **Evidence:** `tools/vkval/` runs ctest under the layer; there is no key to run a live title under it.
- **Failure removed:** barrier/synchronisation errors, the most common Vulkan bug class, are invisible
  on a real GPU run.
- **Design:** a switch enabling `VK_LAYER_KHRONOS_validation` with selectable sets (`core`, `sync`,
  `gpu_assisted`, `best_practices`), routing the messenger to the log and counting errors reported at
  exit. Known false positives live in **one checked-in suppression file keyed by message id, each with a
  reason**, shared with `tools/vkval`. The layer is a tool, never linked into a shipped artifact.
- **Acceptance:** a seeded invalid-usage case raises a counted error; the suppression file is applied;
  the counter appears in the exit summary.
- **Out of scope:** fixing every existing validation error (each gets its own issue).

### D5. Debug names and labels mapped to guest packets

- **Evidence:** names exist for guest shader modules only.
- **Failure removed:** anonymous images, buffers and pipelines in RenderDoc/RGP captures.
- **Design:** `VK_EXT_debug_utils` object names plus begin/end labels per submit, draw and dispatch,
  carrying queue, PM4 packet index, draw index, shader stage and hash, render-target address. **Numbers
  and hashes only, never guest strings.** Zero cost off: the function pointers are not loaded.
- **Acceptance:** a debug-messenger test checks the labels on a real (software) device; a default run
  shows no loaded debug-utils pointers.
- **Enables:** D1 (shared identity), and readable captures from the existing RenderDoc in-app trigger
  (`renderdoc_capture.hpp`, #3321).

### D6. Per-pipeline shader statistics

- **Evidence missing:** 0 hits for `pipeline_executable_properties`.
- **Design:** record VGPR/SGPR counts, spills and instruction counts per pipeline where the driver
  supports `VK_KHR_pipeline_executable_properties`; a local report lists the top pipelines by register
  count and spills (hashes only). Not part of any pass rule.
- **Why:** recompiler quality regressions become numbers before they become fps.
- **Out of scope:** vendor-specific counters (use RGP).

### D7. Survey mode for unsupported sites

- **Idea:** log each distinct unsupported draw, dispatch, shader or export **once**, skip it, count it,
  and write one capped summary at exit, so one run finds every missing piece instead of the first. A
  survey run can never be evidence of passing.
- **Prosper state:** `[recompile-reject]`, `[compute] skip` and NID census tooling exist, but verify
  whether a single run yields a deduplicated, per-kind, capped summary before building; the charter
  already treats every reject as the next thing to implement, so the value is "all of them in one run".
- **Design if missing:** a per-site key (kind + library/NID or opcode or reason + module-relative
  offset), memoised per shader hash, one write at exit, capped, no guest strings.

### D8. Stable guest-meaningful identity for draws and dispatches

The prerequisite D1, D5 and the per-pass timeline (M4) all need. Define once:
submit ordinal, packet index, draw index, pipeline key hash. Without a shared definition each
diagnostic invents its own and their outputs cannot be joined. Run-local ordinals stay run-local (the
charter: addresses and operation ordinals are run-local).

## 4. Measurement track

### M1. Frame-time percentiles and `1% low`

- **Evidence:** `present_frame_rate` has a log-bucket interval histogram, median and active fraction;
  0 hits for `1% low`. `flip_pacing_report.py` reports the guest-flip interval distribution.
- **Design:** read p90/p95/p99 from the **existing** histogram (same estimator, same documented error)
  over `distinct` frames; report `1% low`; refuse a tail the population cannot resolve (p99 needs 100
  intervals) rather than print the maximum under a percentile's name; never fill it for a differenced
  window (the histogram is cumulative).
- **Decision to make explicitly:** two `1% low` conventions are in common use:
  `1000 / mean(slowest 1% of frame times)`, and the cheapest histogram-native form, `1 / p99`. Pick one,
  state it wherever printed, and say which in every figure. They differ whenever the tail is skewed.
- **Surfaces:** `tools/screenshot` summary and manifest; later `prosper-app` exit summary.
- **Status:** in progress as a separate PR.

### M2. Per-frame breakdown: guest CPU, driver, GPU, wait, present

- **Evidence:** `PROSPER_RENDER_TIMING` and the F8 buckets give stage costs, not a per-frame series.
- **Design:** per frame, numeric `cpu_ms` (guest until submit), `rec_ms` (driver record/submit),
  `gpu_ms` (timestamp queries around the frame's submits, converted with `timestampPeriod`; omitted when
  the queue lacks timestamp support), `wait_ms` (blocked on a fence), `present_ms`; plus per-frame
  means in the report. Correlating GPU time to the CPU clock needs `VK_EXT_calibrated_timestamps`,
  which prosper does not use yet (§2).
- **Acceptance:** conversion and schema unit-tested with a synthetic clock; a software-device test that
  a submit yields a `gpu_ms`; harness-forced readback is flagged in the record (rule 7).
- **Out of scope here:** per-pass timelines (M4).

### M3. Run metadata and baseline compare with refusal

- **Evidence missing:** `tools/screenshot` records no build preset, commit, driver version or present
  mode (0 hits; resolution scale IS recorded, as `render_scale` from `PROSPER_RENDER_SCALE`), and no
  tool refuses to compare mismatched runs.
- **Design:** the runtime writes its own run conditions (build preset, short commit baked at build,
  Vulkan vendor id and driver version, present mode, resolution scale, harness id, route/scene id and
  duration, config hash). A `compare --baseline --result` mode prints per-field deltas and **refuses**
  when any condition differs or is missing, so the only things that may differ are the commit and the
  measured fields. Thresholds are proposed from measured run-to-run noise, not guessed. Reads local
  JSON only; never uploads.
- **Why it matters:** this is the mechanical form of the charter's harness rules. It makes "52 fps vs
  58 fps" impossible to quote across a `fifo`/`immediate` or readback/GPU-present boundary.
- **Out of scope:** CI-side perf gating (no GPU in hosted CI); machine model strings (personal
  hardware detail).

### M4. Per-graphics-pass GPU timeline

- **Evidence:** `vkCmdWriteTimestamp` is used for compute dispatches only (and test fixtures).
- **Design:** timestamp queries per render pass and per submit, behind a switch, calibrated to the CPU
  clock, with the identity from D8. Check RGP first (rule 1): build this only if the per-pass
  attribution is needed across titles without a GUI.
- **Acceptance:** same as M2; overhead measured and stated; off by default.

### M5. Stall attribution and invariant violation counters

- **Evidence:** `[perf-alarm]` names a cost per window, not per frame.
- **Design:** a frame over the stall threshold carries a numeric cause (compile, load/I/O, GPU wait,
  guest) derived from M2 and the counters; results report `stalls_by_cause`. Counters, counted only
  after a defined warm-up, for: unrequested CPU waits/readbacks, Vulkan object creation on the per-draw
  path, compile on the submit thread, full compare/copy of guest memory proven unchanged. Reported, not
  a pass rule.
- **Replaces:** the manual dose-response studies (e.g. the SDK-10 submit race).
- **Depends on:** M2.

### M6. Boot, compile and memory series

- **Design:** `boot_ms` (process start to first present), shader-compile and pipeline-creation times
  (totals and p99), and a per-second memory series (VRAM used/budget, guest committed, host-import and
  staging bytes) with run peaks. Builds on `boot_phase_log`, `perf_ledger` and `gpu_memory_budget`.
- **Out of scope:** eviction policy.

### M7. Present latency and jitter

- **Design:** where `VK_KHR_present_id`/`present_wait` exist, per-frame `present_to_display_ms`; report
  p50/p99 and interval jitter (p99 minus p50 of inter-present intervals) over the same non-stall
  frames. Reported, not a pass rule.
- **Depends on:** M1, M2.

### M8. Structured logging sink

- **Evidence:** ~1,491 `fprintf(stderr|stdout)` call sites and ~698 `getenv("PROSPER_*")` gates in
  `prosper/src` alone (about 2,700 and 1,100 across `src/`, `frontends/` and `tools/`).
- **Design:** a thin in-tree `LOG(subsystem, level, ...)` that **keeps the exact `[tag]` prefix** (tools
  and docs grep it), adds one `PROSPER_LOG=gpu=debug,hle=warn` knob, an in-memory ring (feeds D2) and a
  single sink. Fault and signal paths stay on raw `write(2)`: allocation and locks are not
  async-signal-safe. Migrate per folder with `tools/refactor/`, one PR per subsystem. A library
  (spdlog or similar) is not recommended: it is unsafe on those paths and changes the output contract.
- **Why last-ish:** it is a substrate; D2 and M5 can start with a minimal ring and adopt it later.

## 5. Sequencing and parallel lanes

| Lane | Items | Blocked by |
| --- | --- | --- |
| Percentiles | M1 | none |
| Validation key | D4 | none |
| Draw identity + names | D8, D5 | none |
| Shader stats | D6 | none |
| Survey audit | D7 | none |
| Device-loss triage | D1 | D8 |
| Shader printf | D3 | D4 |
| Frame breakdown | M2 | D8 (identity), none for the schema |
| Run metadata + compare | M3 | M1, M2 (to compare something worth comparing) |
| Per-pass timeline | M4 | M2, D8 |
| Stall attribution + counters | M5 | M2 |
| Boot/compile/memory, present | M6, M7 | M2 |
| Crash report | D2 | M8 (ring) |
| Logging sink | M8 | none (incremental) |

Rough order of value: D1, D2, M3 first (the most expensive failures and the least trustworthy
comparisons), then M2, D3, M5; D6, D7 and M7 when a title needs them.

## 6. What this plan deliberately does not do

- **No on-screen overlay** (`--fps` exists), no interactive debug menu that changes guest state.
- **No Windows sanitizer-preset work** here (a build-system item, unrelated to triage).
- **No uploading** of any capture, dump, report or result.
- **No proprietary SDK linked into a binary.**
- **No CI perf gating:** hosted CI has no GPU. Perf evidence is local, recorded in the PR.
- **No frame-rate "fixes":** this plan measures; optimisation is separate work that cites it.

## 7. Per-item definition of done

An item is done when: its acceptance criteria are met; a hand-constructed positive control exists and a
test fails if the instrument goes silent; the off-path cost is measured and stated; the folder's
`AGENTS.md` and, where it adds a `PROSPER_*` switch, `DIAGNOSTIC_GATE_AUDIT.md` are updated; and any
hypothesis the work falsified is recorded under `## Ruled out` below.

## Ruled out

- **"Breadcrumbs already exist because `buffer_marker` appears in the tree."** Falsified 2026-10-02:
  every hit is `is_nullable_raw_buffer_marker_candidate` or similar capture/recompiler naming; none
  reference `VK_AMD_buffer_marker` or `vkCmdWriteBufferMarkerAMD`.
- **"There are no GPU timestamp queries."** Falsified 2026-10-02: `live_compute.cpp` writes timestamps
  around compute dispatches. The gap is graphics passes, not timestamps in general.
- **"A third-party logging library is the fix for diagnosability."** Not adopted: the cost is the
  output contract (grepped `[tag]` lines) and async-signal-safety, not the formatting API (see M8).
