# Renderer performance and tooling findings (2026-07-14)

This is the handoff for the native Windows performance work tracked in
[#702](https://github.com/mattias800/ps5ys/issues/702). It records measured results, rejected
experiments, capture-tool corrections, and the remaining architectural work. Use it with
`FRONTEND_APP.md` for the environment-variable reference and `tools/AGENTS.md` for capture/replay.

## Scope and decision

The Messenger's first level is visually correct on the native Windows frontend. The work here was
therefore profiling and general renderer improvement, not another graphics-correctness workaround.
The representative heavy frame has four compute dispatches interleaved with four graphics spans.

The current build runs that scene at roughly 24 FPS on the measured host, up from approximately
12 FPS at the start of this pass. Further Messenger-only work toward 60 FPS is deliberately paused.
The remaining cost is structural, and the next renderer changes must be tested against a 3D title
before choosing a resource-lifetime or scheduling architecture.

## Measurement method

Use a `RelWithDebInfo` native Windows build, a fresh process for each A/B, the same input route and
save data, and wait until the same first-level scene is stable. Enable rolling stage timings:

```powershell
$env:PROSPER_RENDER_TIMING = '1'
./prosper/scripts/run-windows.ps1 ./PPSA24651-app0 -NoBuild
```

For slow resource attribution, use `detail` and defer detail output until the scene under study:

```powershell
$env:PROSPER_RENDER_TIMING = 'detail'
$env:PROSPER_RENDER_TIMING_DETAIL_MIN_SUBMIT = '5000'
```

Compare rolling `[render-window]` lines, not whole-run cumulative values. App FPS is useful as the
outcome, but the stage buckets determine which subsystem actually changed.

## Results

| Change | Before | After | Main evidence |
|---|---:|---:|---|
| Avoid intermediate framebuffer copies and move final callback output | about 12.3 FPS | about 15.3 FPS | unclassified/output work fell from about 12.4 ms to 1.1 ms |
| Per-submit readable-range cache | about 15 FPS | about 21 FPS | realization fell from 15-17 ms to 3.3 ms; table/fold work from 11.6-13 ms to 1.7 ms |
| Exact-byte persistent texture decode cache | about 20 FPS | 23.7-24.8 FPS | resource construction fell from 10.1-10.5 ms to about 3.9 ms |

The readability cache is scoped to one synchronous submit and caches only whether a guest range is
mapped. It does not cache guest bytes. A representative heavy submit served about 940 of 962 checks
from the local cache and needed only about 22 `VirtualQuery` probes.

The shader fold cache retains only the scalar/resource instructions needed by dynamic descriptor
resolution. Its key includes a copy of the decoded shader bytes, so same-address guest mutation is
detected. It is bounded to 64 MiB by default and can be disabled with
`PROSPER_NO_SHADER_DECODE_CACHE=1`.

The persistent texture cache covers only guest-backed tiled `Unorm8x4` sampled textures. It excludes
render targets, storage images, and unsupported formats. Every hit copies and compares the complete
padded source before reusing decoded pixels, so address reuse and in-place mutation invalidate the
entry. In the measured scene it produced 42 persistent hits and 14 callback-local hits out of 57
texture uses, retained 101 entries / about 75 MiB, and observed no invalidation. Disable it with
`PROSPER_NO_TEXTURE_DECODE_CACHE=1`; its default 256 MiB budget is controlled by
`PROSPER_TEXTURE_DECODE_CACHE_MB`.

## Current frame budget

After the above changes, a representative renderer submit remains about 36-38 ms. Approximate costs
per frame are:

| Area | Approximate cost |
|---|---:|
| Four graphics GPU waits | 10-11 ms |
| Four graphics readbacks | 4-5 ms |
| Graphics pipeline creation/setup | about 3 ms |
| Backend resource/descriptor setup | about 2 ms |
| Graphics record/upload | about 2 ms |
| Four compute dispatches | about 3 ms |
| Remaining realization, publication, and overhead | remainder |

These numbers explain why another small CPU lookup cache is not a credible path to 60 FPS. The
renderer currently creates transient Vulkan objects, waits, and reads back at each ordered backend
boundary. A durable improvement needs persistent resources and coordinated graphics/compute
ownership, then direct presentation where screenshots/captures do not require CPU pixels.

## Capture correction and dependency evidence

Direct `PROSPER_GPU_CAPTURE` used to snapshot only draw items after execution. On a known Messenger
submit this silently produced `draws=56 computes=0 operations=56`, losing the mixed PM4 order. The
fix tracked in [#714](https://github.com/mattias800/ps5ys/issues/714) snapshots draws, computes, and
the original ordered operation list before execution, then attaches final pixels and the oracle hash
after execution.

The corrected capture reports `draws=56 computes=4 operations=60`. Its operation order is:

```text
C0, draw 0, C1, draws 1..47, C2, draws 48..54, C3, draw 55
```

`gpu_replay --graph` found four concrete edges: C0 to C1 through a 196608-byte buffer, C1 to the
first large graphics span through fragment binding 32, C1 to C2 through that buffer, and C2 to C3.
This disproves a safe blind "run all compute first" optimization. Some later graphics spans have no
captured overlap with the following dispatch, but exploiting that requires explicit dependency and
resource-ownership scheduling rather than reordering by operation type.

## Rejected experiments

- Caching `build_stage_table` from shader address and user SGPRs is incorrect. Pointed-to descriptor
  memory can mutate while those values remain unchanged. The experiment stalled Messenger on its
  initial loading screen and was removed.
- A driver `VkPipelineCache` alone made no measurable difference. The backend still creates transient
  render passes and layouts, so stable application-level pipeline keys/lifetimes are needed first.
- Reordering compute and graphics by type is not valid. The corrected capture graph proves real
  producer/consumer edges in the representative frame.

## Separate unresolved risk

Native Windows boot can still intermittently stop after exactly 75 submits while asset loading and
GC suspension continue. The same binary and route can pass on a retry. This is tracked separately in
[#712](https://github.com/mattias800/ps5ys/issues/712); do not misclassify it as a renderer-cache
regression without checking the submit count and suspend logs.

## Next renderer step

Bring up a representative Unreal/3D workload and collect the same stage timings plus a corrected
mixed-operation capture. Then design one cross-engine persistent-resource change around measured
resource identities and dependencies. Keep the exact-byte and disable-switch A/B discipline used
here, and preserve screenshot/capture correctness while reducing the synchronous boundaries.
