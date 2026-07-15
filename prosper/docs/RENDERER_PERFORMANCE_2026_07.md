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

## Dead Cells backend upload duplication (2026-07-15)

Dead Cells exposed a second duplication layer after the frontend decode caches. The frontend correctly
reused one decoded pixel allocation for repeated texture references in a graphics span, but the Vulkan
backend created a separate image, device-memory allocation, staging buffer, staging allocation, and full
upload for every draw reference. One slow control window averaged 188.8 texture references and 5.07 GiB
of uploads per backend call even though 181.3 references reused pixels the frontend had already built.

The backend now uploads each distinct `(decoded pixel pointer, width, height, depth, image dimension)`
once per synchronous backend call. Draw bindings still get separate image views and samplers, preserving
component swizzles and sampler state. The scope is deliberately one call: no guest-memory freshness or
cross-frame lifetime assumption is introduced. Set `PROSPER_NO_BACKEND_TEXTURE_SHARE=1` for the legacy
one-upload-per-reference A/B.

Native Windows / RTX 4090, same RelWithDebInfo binary, fresh saves, documented Dead Cells full-render
route, and 110-second wall-clock samples:

| Measurement | Sharing disabled | Sharing enabled |
|---|---:|---:|
| App rate in the slow section | 0.6 FPS | 7-11 FPS |
| Peak process working set | 6.80 GiB | 2.08 GiB |
| Peak process private memory | 14.28 GiB | 4.40 GiB |
| Peak dedicated GPU commit | 5.15 GiB | 0.28 GiB |
| Peak shared GPU commit | 5.18 GiB | 0.36 GiB |
| Transient pool discards | 15,615 and rising | 0 |

The enabled run advanced farther in the same wall-clock period, so the end-to-end peaks are intentionally
reported as outcomes rather than a matched-scene microbenchmark. The renderer counters establish the
mechanism directly: a representative enabled 54-reference span produced 11 uploads / 96.3 MiB instead
of 54 uploads / 460.3 MiB. Its pool stabilized at 1,865 allocations / 374.7 MiB with no discards. Process
private memory averaged 4.34 GiB at 60-90 seconds and 4.37 GiB at 90-110 seconds, never exceeding
4.40 GiB after 60 seconds.

The Vulkan regression test compares sharing on/off byte-for-byte while using separate swizzled image
views. It also renders a two-slice 3D texture twice, verifies the selected depth slice, and requires one
depth-accounted upload. This protects the general 2D and 3D paths rather than only the Dead Cells shape.

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
