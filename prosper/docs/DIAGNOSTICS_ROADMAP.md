# Diagnostics roadmap — what prosper can and cannot tell you about a failure

**Read this before building a new `PROSPER_*` diagnostic.** It is a map of which failure-triage and
performance-observation capabilities exist, which are missing, and in what order they are worth
building. It is a plan, not a status report: every row was checked against the tree on 2026-10-02
(grep over `src/`, `frontends/`, `tools/`), and the evidence column says what was found.

Companion docs: `GPU_PROFILING_EXTERNAL.md` (vendor tools that need no prosper change),
`DIAGNOSTIC_GATE_AUDIT.md` (existing switches), `src/diagnostics/AGENTS.md` and
`src/diagnostics/perf/AGENTS.md` (the always-on `[perf-alarm]` layer), `tools/AGENTS.md` (tool index).

The gap list was seeded by comparing against the planning beans of the sibling PortPS5 project
(diagnostics epic and performance track). Those are **ideas to re-derive**, not code to port; see the
evidence hierarchy in the charter.

## Principles

- **Reach for free vendor tooling first** (RGP, RenderDoc, `radeontop`, the validation layer). Build a
  `PROSPER_*` switch only for something the guest-facing layer knows and the GPU vendor cannot see:
  guest packet, guest PC, guest shader address, HLE call.
- **A signal that answers "where did it stop?" must be reachable without a rebuild.** See
  `src/diagnostics/AGENTS.md` for the failure this folder already had once.
- **Observers must not change guest or renderer behaviour**, and a diagnostic's lever must show it moved
  (prefer experiments that detect their own invalidity).
- **Name the harness.** A rate from `tools/screenshot` is a forced-readback rate; a `fifo` run measures
  the display. Percentiles are over `distinct` frames, not `presented`.

## What exists today

| Capability | Where | Notes |
| --- | --- | --- |
| Frame grab + offline replay | F9 `.prgbundle`, `tools/gpu_replay` | Schedulable headless (`PROSPER_GRAB_BUNDLE_*`). |
| Bounded perf capture | F8 `.prperf`, `tools/perf/performance_capture_report.py` | Reports by time. Six timestamp brackets per compute dispatch. |
| Always-on alarms | `[perf-alarm]`, `src/diagnostics/perf/` | Names the cost, its size against the frame budget, the next instrument. No per-frame cause code. |
| Stage buckets | `PROSPER_RENDER_TIMING` | `setup_resources [buffer … copy …]` style breakdown. |
| GPU submit index | `PROSPER_GPU_TIMELINE`, `tools/gpu_timeline` | Guest-submit semantic index. **Not** GPU pass timing. |
| Per-dispatch GPU timing | `frontends/shared/live/live_compute.cpp` | `vkCmdWriteTimestamp` around compute dispatches only (perf-ledger pair, F8 brackets). |
| CPU/GPU clock correlation | calibrated timestamps | Present in 7 files. |
| Validation layer on tests | `tools/vkval/` | Runs ctest under `VK_LAYER_KHRONOS_validation`. **Tests only**, not a runtime key for a live title. |
| Vulkan object names | `src/gpu/diagnostics/vk_object_names.hpp` | Guest shader modules only; other objects anonymous in RenderDoc/RGP. |
| VRAM budget | `src/gpu/diagnostics/gpu_memory_budget.hpp` | `VK_EXT_memory_budget`. |
| Guest-side debugging | `boot_trace`, `PROSPER_HWBP`/`HWWATCH`/`PEEK`, `guest_bt`, `hang_probe`, `tools/re/xref.py` | Manual; each needs the fault already reproduced. |

## Gaps, ranked

Ranking is by how much investigation time the capability removes, weighted by how often the failure
class occurs. None of these is implemented.

| # | Capability | Evidence it is missing | Why it ranks here |
| --- | --- | --- | --- |
| 1 | **Device-loss triage**: `VK_EXT_device_fault` plus per-draw breadcrumbs (`VK_AMD_buffer_marker`, `VK_NV_device_diagnostic_checkpoints`) mapped to guest packet, draw index, pipeline key | 0 hits for `device_fault` / `diagnostic_checkpoints`; the `buffer_marker` hits in `gpu_capture_internal.hpp` and the recompiler are an unrelated "raw buffer marker" concept, not the AMD extension | A device loss today says what was submitted, not where the GPU stopped. GPU hangs and stalls are among the most expensive bugs here. |
| 2 | **Symbolised crash report + flight recorder**: fault handler writes registers, a stack walk with frames classified guest/HLE/host, the last N diagnostic records, and a minidump | 0 hits for `minidump`; no recorder of recent diagnostic lines | Replaces hand-run `guest_bt`/`xref` after every crash. Needs the structured sink from row 7. |
| 3 | **Shader printf at a guest PC** (`NonSemantic.DebugPrintf`) | 0 hits for `DebugPrintf` | Traces a wrong value to its RDNA instruction in one run; today that is `PROSPER_DBG` plus IR dumps. Instrumented modules must never enter any shader cache. |
| 4 | **Per-graphics-pass GPU timeline** (timestamp queries per render pass and per submit, calibrated to the CPU clock) | `vkCmdWriteTimestamp` exists in `live_compute.cpp` (compute only) and test fixtures; no graphics-pass use | Answers "which pass costs the 2 ms" on RADV, which `radeontop` and the F8 buckets cannot. Gate behind a `PROSPER_*` switch: timestamps cost GPU time. Check RGP first. |
| 5 | **Frame-time percentile report**: p50/p90/p95/p99 and `1% low` over `distinct` frames | 0 hits for `1% low`; percentiles exist only in unrelated tools | Quoted figures should come from one harness-labelled report rather than hand-assembled samples. |
| 6 | **Per-slow-frame cause code and invariant counters** | `[perf-alarm]` names a cost per window, not per frame | Counts unrequested readbacks, per-draw `vkCreate*`, submit-thread compiles. Would replace the manual dose-response studies. |
| 7 | **Structured logging sink** (`LOG(subsystem, level)`, one `PROSPER_LOG` knob, keeps the `[tag]` prefix) | ~1,491 `fprintf(stderr/stdout)` call sites and ~698 `getenv("PROSPER_*")` gates | Substrate for rows 2 and 6. Migrate per folder with `tools/refactor/`; leave signal/fault paths on raw `write(2)`. A thin in-tree macro, not spdlog. |
| 8 | **Per-pipeline shader statistics** (VGPRs, spills, occupancy via `pipeline_executable_properties`) | 0 hits | Useful for the FPS frontier; narrower than rows 1–4. |
| 9 | Debug names on more Vulkan objects (buffers, images, pipelines, passes) mapped to guest packets | Names exist for shader modules only | Makes RenderDoc/RGP captures readable. |

**Deliberately not planned:** an on-screen overlay (`--fps` exists) and sanitizer presets for
Windows (unrelated to triage).

## Suggested sequencing

1. Row 7 only as far as rows 2 and 6 need it; do not start a whole-tree migration.
2. Rows 1 and 3 are independent of the sink and can run in parallel.
3. Row 4 before any further renderer-performance work that claims a pass is the cost.
4. Row 5 is small and can land with any of the above.

Each row becomes its own tracker issue (`area:infra`), stating the failure it removes the need to
bisect.

## Ruled out

- **"Breadcrumbs already exist because `buffer_marker` appears in the tree."** Falsified 2026-10-02:
  every hit is `is_nullable_raw_buffer_marker_candidate` or similar capture/recompiler naming; none
  reference `VK_AMD_buffer_marker` or `vkCmdWriteBufferMarkerAMD`.
- **"There are no GPU timestamp queries."** Falsified 2026-10-02: `live_compute.cpp` writes timestamps
  around compute dispatches. The gap is graphics passes, not timestamps in general.
