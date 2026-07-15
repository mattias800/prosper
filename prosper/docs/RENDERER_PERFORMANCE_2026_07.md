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

## Dead Cells cross-callback texture residency (2026-07-15)

Per-call sharing removed duplicate uploads within one graphics span, but the same large immutable
linear atlases were still copied from guest memory and uploaded again on every callback. The steady
Dead Cells loading submit referenced 54 textures. Eleven distinct uploads remained and transferred
96.3 MiB per callback, including unchanged 64 MiB, 32 MiB, and 8 MiB linear RGBA atlases.

The frontend cache now exact-validates linear `Unorm8x4` sources as well as tiled sources. A validated
content version receives a monotonic ID; any source-byte or readable-prefix change creates a new ID.
The Vulkan backend retains only images with such an ID, under a 256 MiB / 1024-entry default bound.
Render targets, storage images, captured host backing, and unvalidated formats never receive an ID.
Per-call views and samplers remain independent, so descriptor swizzles and filtering are unchanged.

Native Windows / RTX 4090, same RelWithDebInfo binary, fresh saves, documented full-render route, and
matched 110-second samples:

| Measurement | Forced upload | Persistent images |
|---|---:|---:|
| Late app rate | about 10.9 FPS | 13.7-14.4 FPS |
| Late backend call | 36.8-37.9 ms | 17.7-19.0 ms |
| Backend resource setup | 5.8-6.1 ms | 0.42-0.44 ms |
| GPU fence wait | 21.8-22.6 ms | 8.1-9.5 ms |
| Steady texture upload | 96.3 MiB/call | 0 |
| Transient allocation pool | 375.3 MiB | 232.0 MiB |
| Peak process working set | 2.20 GiB | 2.29 GiB |
| Peak process private memory | 4.54 GiB | 4.60 GiB |

Both runs reached `Loading level PrisonStart`; neither cleared the separately tracked loading
starvation within the sample. The enabled run retained 15 frontend versions / 192.9 MiB and the same
192.9 MiB of backend images, with zero invalidations and zero transient-pool discards. Private memory
plateaued rather than growing without bound. The small residency increase is therefore the explicit
cache tradeoff, not a return of the earlier multi-gigabyte per-draw allocation growth. Disable the
backend half with `PROSPER_NO_BACKEND_PERSISTENT_TEXTURES=1`, or change its independent byte budget
with `PROSPER_BACKEND_TEXTURE_CACHE_MB`.

`test_texture_sample_render` verifies the initial miss/upload, a cross-callback hit with zero upload,
byte-identical forced-upload output, and a new content version that cannot reuse the prior image.

## Dead Cells cross-submit mapping topology cache (2026-07-15)

After persistent texture residency, Dead Cells still spent 38-44 ms per large submit in the
`tables=` phase. `PROSPER_STAGE_FOLD_PROFILE=1` showed that the scalar interpreter body took about
0.003 ms per call and shader decode validation about 0.002 ms. The dominant cost was the Windows
readability guard: the same mapped descriptor regions were checked thousands of times, but the
positive `VirtualQuery` results were discarded at the end of every synchronous submit.

Positive readable ranges from explicitly tracked kernel-HLE mappings now survive across submits while
the HLE guest-mapping generation remains unchanged. Every tracked map, unmap, or protection change
advances that generation and clears the ranges before reuse. Host-managed guest stacks, diagnostic
mappings, and other regions whose lifetime does not pass through the memory HLE remain submit-local.
The cache stores only OS mapping topology; it never stores descriptor bytes, shader fold results, or
resource tables. The existing contract that a synchronous GPU submit's guest allocations remain
mapped until the submit returns is unchanged. Use `PROSPER_NO_GUEST_READ_CACHE=1` for a control run.

Native Windows / RTX 4090, same RelWithDebInfo binary and matched 169-175-draw Dead Cells scene:

| Measurement | Submit-local ranges | Generation-retained ranges |
|---|---:|---:|
| Actual `VirtualQuery` calls per submit | about 31 | 0-2 |
| Table-fold phase | 38-44 ms | 2-5 ms |
| Total submit | 99-103 ms | 63-66 ms |
| App rate | about 9.1 FPS | 13.5-14.2 FPS |

A separate 300-second run remained stable and sustained about 17-18 FPS in a later 56-draw scene.
Its final 60 seconds averaged 4705 MiB private memory and 3681 MiB working set, with changes of about
+31 MiB and +34 MiB respectively rather than continuing multi-gigabyte growth. The fresh-save scripted
route did not reach the PrisonStart progression marker in that run, so this is renderer and memory
stability evidence, not a new progression proof.

The large process baseline has a separate source. Dead Cells explicitly allocates and maps a
`0xc0000000` (3 GiB) direct-memory arena at 2 MiB alignment. On Windows, the shared-section view
previously fell back to one eagerly committed `MEM_PRIVATE` allocation. A `VirtualQueryEx` census at
35 seconds attributed exactly 3072 MiB of roughly 4638 MiB private commit to this arena; measured
renderer persistent images and transient pools together accounted for about 441 MiB. Placeholder,
aligned shared-view, or sparse-realization work belongs to
[#697](https://github.com/mattias800/ps5ys/issues/697) because it must preserve guest query semantics,
untouched zero-page reads, aliases, and partial unmaps.

The Windows sparse large-alignment path now uses `VirtualAlloc2` address requirements plus
`MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)` and commits shared 16 KiB pages on demand. The focused test
checks 2 MiB placement, guest query state, an untouched far page, GPU-read materialization, alias
coherence, protection changes before and after first touch, and exact whole-view unmap. The modern APIs
are resolved dynamically; systems without them retain the old mapping path.

A five-minute fresh-save Dead Cells run removed the 3072 MiB private fallback and had no sparse commit
failure. In the late 54-draw loading scene, private commit averaged 1604 MiB over the final minute and
changed by about +1 MiB. Working set was 2815 MiB at exit and still gaining about 190 MiB over that
minute: repeated renderer reads were making the shared section resident even though they no longer
charged private commit. That run repeatedly scanned about 460 MiB of declared texture ranges per submit,
spent about 40 ms in resource construction, and presented at roughly 8-10 FPS. It did not reach the
`PrisonStart` marker, so it is memory/fault stability evidence rather than a progression proof. Avoiding
repeated scans or materialization of untouched resource tails is follow-up renderer work, not a reason to
restore the eager 3 GiB private allocation.

## Windows sparse page-state cache (2026-07-15)

Demand paging removed the 3 GiB private commit, but the renderer still called `VirtualQuery` for every
resource reference on every submit to prove that its sparse direct-memory pages were committed. Dead
Cells' buffers made the cost clear: a matched 170-draw scene referenced about 680 buffers containing
only 0.4 MiB in total, yet their guards took about 40 ms per submit. The bytes and copies were not the
bottleneck; repeated kernel page-state queries were.

Each sparse Windows direct-memory view now retains a compact bitmap of host-committed 16 KiB pages.
The existing HLE mapping generation invalidates the bitmap after any tracked map, unmap, or protection
change. A miss still commits the exact guest pages with the current tracked protection and falls back to
`VirtualQuery` if the commit call fails; a hit skips the OS query. The 3 GiB Dead Cells view needs only
24 KiB for this metadata. A thread-local positive mapping lookup, guarded by the same generation,
also avoids rescanning the HLE mapping vector for every resource. Set
`PROSPER_NO_SPARSE_DMEM_PAGE_CACHE=1` or `PROSPER_NO_SPARSE_DMEM_ACCESS_CACHE=1` to disable the
corresponding layer for controlled comparison.

Native Windows / RTX 4090, the same RelWithDebInfo binary, fresh processes without pad input,
115-second runs, and the last 20 matched 168-176-draw title/menu windows:

| Measurement | Page cache disabled | Page cache enabled |
|---|---:|---:|
| Frontend callback | 91.70 ms | 46.09 ms |
| Resource construction | 60.28 ms | 13.12 ms |
| Texture resource time | 18.12 ms | 9.82 ms |
| Buffer resource time | 39.53 ms | 1.04 ms |
| Late app rate | 8.75 FPS | 14.7 FPS |

The enabled run's final ten process samples averaged 1584 MiB private memory and 2034 MiB working set;
both decreased slightly over that sample rather than growing. `dmem_available` exercises first-touch
commit, a far untouched page, physical aliases, read-only rejection, and generation invalidation when a
materialized page becomes writable. The full native Windows suite passed 70 of 71 tests; the remaining
`module_loads_eboot` failure was the expected fixture mismatch because this profiling build was configured
against Dead Cells while that test pins The Messenger's import counts.

A separate 160-second presented-screenshot run used the correctly file-prefixed
`PROSPER_PAD_SCRIPT=@.../reach-first-gameplay-full-render.pad` route. It reached
`Loading level PrisonStart` and saved 32 normal 3840x2160 screenshots; all 32 had distinct source and
pixel identities. Inspected samples showed the title, update/menu flow, Prisoners' Quarters loading art,
and the later loading fade without a blank-frame collapse. In the late 54-draw scene, buffer guards took
0.40-0.43 ms, resource construction 9.56-12.27 ms, and total submits 45.21-48.06 ms. The run remained
in the existing level-loading state at 160 seconds, so this is routed output evidence rather than a new
gameplay progression proof. A matching 165-second app run sustained an 18.95 FPS median over its last
ten reports. Its last ten process samples averaged 1619 MiB private memory with a +1.1 MiB change;
working set averaged 2733 MiB and gained 119 MiB as the repeated texture scans continued making shared
pages resident.

## Shared rendered-frame publication (2026-07-15)

The selected render target previously crossed three additional full-frame CPU copies after Vulkan
readback: persistent RTT storage to the live-renderer return vector, that vector to the present layer,
and `present_readback` to the app's scratch vector. At 3840x2160 RGBA8 each copy moved about 31.6 MiB.

`RenderedFrame` now carries immutable shared ownership from the live renderer through the executor and
present layer. `prosper-app` acquires a lifetime-safe `PresentFrameLease` and uploads directly from that
storage. Compatibility readback, screenshots, captures, and replay still copy by design. Unit coverage
asserts pointer identity across renderer-to-present publication and proves a lease survives replacement
and `present_reset()`.

On native Windows / RTX 4090, the fresh-save full-render Dead Cells route improved the matched 54-draw
loading scene as follows:

| Measurement | Before | Shared publication |
|---|---:|---:|
| Frontend callback | 36-38 ms | 31-34 ms |
| Frontend output copy | 4.3-4.4 ms | 0.0 ms |
| Core submit | 46-49 ms | 40-42 ms |
| Core publication | 1.7 ms | 0.0 ms |
| App present rate | about 19 FPS | about 21-22 FPS |

The speedup was sufficient for the same 165-second route to finish Dead Cells' 71.6-second
`PrisonStart` parse and enter the next loading workload instead of remaining in the 54-draw loop. That
post-parse workload uses about 344 draws, eight dispatches, and four graphics spans per submit. It runs at about
4.3 FPS, with about 202 ms core submit time: 32 ms realization, 20 ms table work, and 170 ms ordered
backend execution. The frontend portion spends about 51 ms building resources and 89 ms in its Vulkan
backend. Full-resolution screenshots at 150 seconds still show the Prisoners' Quarters loading art, so
this is a later loading phase rather than confirmed gameplay. It is now the representative optimization target.

Private memory rose once from about 1.59 GiB to about 1.90 GiB when that workload arrived, then fluctuated
in a narrow band during the final 20 seconds. Working set settled around 3.82 GiB. Vulkan graphics and
compute pools remained bounded at 355 MiB and 47 MiB. The workload nevertheless invalidates about two
persistent textures per submit window and performs repeated uploads, so resource versioning and residency
are the next memory/performance investigation rather than further publication lookup tuning.

## Current frame budget

After shared publication, the Dead Cells post-parse loading workload is the most useful current budget:

| Area | Approximate cost |
|---|---:|
| Draw realization | 32 ms |
| Scalar table work (included above) | 20 ms |
| Frontend resource construction | 51 ms |
| Frontend Vulkan work | 89 ms |
| Ordered backend total (graphics + compute) | 170 ms |
| Frame publication | approximately 0 ms |
| Whole submit | 201-203 ms |

These numbers explain why another small CPU lookup cache is not a credible path to 60 FPS. The
renderer currently creates transient Vulkan objects, waits, and reads back at each ordered backend
boundary. A durable improvement needs persistent resources and coordinated graphics/compute ownership.
Shared CPU publication is complete; direct GPU-image presentation remains a later step because the app
and headless renderer still own separate Vulkan devices.

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
