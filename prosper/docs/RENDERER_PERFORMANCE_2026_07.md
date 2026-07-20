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

The persistent texture cache now covers guest-backed linear/tiled 2D sampled `Unorm8` textures with
1-4 components and BC1-BC7 textures. It excludes render targets, storage images, cube/volume and DCC
surfaces, captured host backing, and unsupported formats. Every hit validates the complete native,
padded-tiled, or compressed-block source range, so address reuse and in-place mutation invalidate the
entry. Its default 1 GiB budget covers Evergate's measured 835 MiB decoded working set. Disable it with
`PROSPER_NO_TEXTURE_DECODE_CACHE=1`; the budget is controlled by
`PROSPER_TEXTURE_DECODE_CACHE_MB`.

## Evergate native-render transition (2026-07-18)

Evergate's title screen and new-game transition exercise roughly 1,000 sampled-texture references and
470-490 draws per native 1920x1080 submit. On Windows / RTX 4090, full cadence and a fresh save, the
original 256 MiB `Unorm8x4`-only cache repeatedly decoded large BC atlases and fell to about 0.9 FPS.
BC/narrow-format coverage plus the 1 GiB default retains about 835 MiB of decoded pixels, removes
steady-transition decode misses, and reduces frontend resource construction from about 421 ms to
38-40 ms.

After texture decoding stopped dominating, per-draw immutable shader analysis was the next hotspot.
Shader code span, PC-relative dispatch metadata, and fragment interpolation layout are now shared from
a 64 MiB cache only after exact validation of the complete instruction/embedded-table byte span.
Concrete SGPRs, descriptor tables, addresses, and backing bytes are still rebuilt for every draw.

Matched dense windows were:

| Measurement | Texture cache only | + immutable shader analysis |
|---|---:|---:|
| Draws / submit | 469-474 | 470-488 |
| Core submit | 572-579 ms | 426-449 ms |
| Draw realization | 222-223 ms | 55-61 ms |
| Stage tables | 93-94 ms | 20-22 ms |
| Shader lookup/analysis | 110 ms | 17-19 ms |
| App rate at the dense transition | 1.7-1.9 FPS | 2.1 FPS |

The following lower-draw intro windows reached 2.6 FPS. Backend execution remains about 360-388 ms in
the dense scene and is now the dominant cost. `PROSPER_NO_SHADER_ANALYSIS_CACHE=1` restores direct
analysis for comparison; `PROSPER_SHADER_ANALYSIS_CACHE_MB=<MiB>` changes the analysis budget.

## Evergate call-local Vulkan resource sharing (2026-07-19)

After the Windows direct-memory red-zone fix made the fresh-save route deterministic, the transition
showed that a single 1080p pass recreated Vulkan buffers, image views, samplers, descriptor pools, and
layouts for roughly 450 draws. The backend already waits for a fence before returning, so immutable
objects with complete equal contracts can safely share one call-local lifetime without changing ordered
graphics/compute boundaries or adding cross-submit freshness assumptions.

Storage-buffer sharing requires the same nonzero guest address, captured size, and every captured byte.
Equal bytes at different guest addresses remain distinct because a shader-visible storage resource may be
writable. Texture bindings include the image, view type/format/swizzle, and complete sampler state.
Descriptor-set and pipeline layouts include their complete binding/layout contracts, while one descriptor
pool supplies all sets in the call. `PROSPER_NO_BACKEND_RESOURCE_SHARE=1` restores distinct keyed objects
for A/B; the call-wide descriptor pool remains in both modes.

Native Windows / RTX 4090, one RelWithDebInfo binary, fresh saves, native scale/cadence, the documented
Evergate route, and submit-aligned 25-submit windows produced the following matched late-transition result:

| Measurement | Keyed sharing disabled | Call-local sharing |
|---|---:|---:|
| Draws / submit | 475.2 | 472.0 |
| Whole submit | 409.1 ms | 354.1 ms |
| Backend execution | 285.8 ms | 218.7 ms |
| Backend resource setup | 84.0 ms | 64.6 ms |
| Pipeline-layout work | 16.9 ms | 2.0 ms |
| Backend cleanup | 103.4 ms | 72.7 ms |
| Observed transition rate | about 2.4 FPS | about 2.7 FPS |

The heaviest nearby 486-488-draw windows improved by about 23%, but the conservative matched result above
is the planning baseline. An additional experiment hashed and shared index buffers and complete descriptor
bundles. Evergate's instances were mostly unique, so hashing/key construction cost more than the avoided
objects and regressed a 476-draw window to 483.9 ms; that tranche was removed. The remaining major cost is
still structural GPU submission/fence/readback ownership, not another broad exact-hash cache.

## Evergate persistent host-buffer objects (2026-07-19)

Call-local sharing still created roughly 170-180 distinct host-visible storage buffers in each backend
call during Evergate's dense transition, then destroyed them after the fence. A frontend submit contains
about 14 such calls. Fine-grained probes showed descriptor-pool, descriptor-layout, and descriptor-update
work together below 0.1 ms/call; storage-buffer creation/upload cost 1-4 ms/call and buffer teardown another
2-5 ms/call. The repeated Vulkan buffer-object lifecycle, not descriptor management, was the actionable
resource cost.

The backend now pools persistently mapped host-coherent buffers after the existing fence wait. Buffers use
power-of-two capacities to avoid exact-size shape churn, but descriptor ranges remain the exact captured
byte count. The cache is thread-local, bounded to 4096 entries / 256 MiB by default, and contains no guest
identity or stale-content assumption: every checkout rewrites all shader-visible bytes. Set
`PROSPER_NO_BACKEND_BUFFER_POOL=1` for the transient-object A/B or
`PROSPER_BACKEND_BUFFER_POOL_MB=<MiB>` to change the byte budget.

Native Windows / RTX 4090, one final binary, separate fresh saves, native scale/cadence, the documented
route, normal renderer timing only, and matched dense 25-submit windows produced:

| Measurement | Pool disabled | Capacity-class pool |
|---|---:|---:|
| Draws / submit | 479.3 | 483.6 |
| Whole submit | 198.6 ms | 130.3 ms |
| Backend execution | 151.0 ms | 81.5 ms |
| Backend resource setup | 51.3 ms | 25.7 ms |
| Backend cleanup | 48.7 ms | 4.8 ms |
| GPU fence wait | 37.2 ms | 38.0 ms |
| Observed rate near frame 360 | 3.2 FPS | 4.1 FPS |

An initial exact-size pool was rejected: it filled the 4096-entry cap and accumulated about 20,000
evictions, leaving setup/cleanup unchanged. Capacity classes stabilized at 2464 buffers / 7.4 MiB with
zero evictions in the same transition; a detailed sample reached 203,263 hits after 2,464 compulsory
misses. The flat GPU wait confirms the measured improvement comes from CPU Vulkan object lifetime rather
than different GPU work. The next dense-scene costs are texture-binding setup and the synchronous
submit/fence/readback architecture.

## Evergate packed host-buffer arenas (2026-07-19)

The persistent buffer pool removed allocation and destruction, but dense backend calls still checked out
about 179 distinct Vulkan buffers for roughly 267 storage-buffer references. A count-only probe also found
about 80 texture references collapsing to only five unique texture binding objects, plus 2.5 descriptor-set
layouts and 1.4 pipeline layouts per call. The remaining resource setup was therefore dominated by the
large number of logical storage uploads, not texture view/sampler or layout object counts.

The backend now suballocates those logical uploads from aligned slices of a 1 MiB mapped arena. Descriptor
offsets follow `minStorageBufferOffsetAlignment`, descriptor ranges remain the exact logical byte count,
and distinct logical resources never overlap. The arena is returned to the existing bounded pool only after
the backend fence wait. `PROSPER_NO_BACKEND_BUFFER_ARENA=1` restores the per-logical-buffer pooled path for
an A/B, while `PROSPER_BACKEND_BUFFER_ARENA_KB=<KiB>` changes the target arena size.

Native Windows / RTX 4090, one final binary, separate fresh saves, native scale/cadence, the documented
route, and matched dense timing windows produced:

| Measurement | Per-logical pool | Packed arena |
|---|---:|---:|
| Draws / submit | 489.6 | 485.4 |
| Whole submit | 135.9 ms | 131.9 ms |
| Backend execution | 83.7 ms | 81.0 ms |
| Backend resource setup | 26.9 ms | 25.3 ms |
| Backend cleanup | 5.1 ms | 4.7 ms |
| GPU fence wait | 37.3 ms | 37.4 ms |
| Persistent buffer objects | 2,434 | 4 |
| Persistent mapped bytes | 6.7 MiB | 4.0 MiB |

A second nearby pair measured 139.4 to 133.6 ms whole-submit time, confirming a modest 3-4% end-to-end
gain. The flat 37 ms fence wait remains the primary structural limit; packing removes CPU object handling
and pool bookkeeping but does not change GPU work or synchronous ownership.

## Evergate intermediate scanout readback (2026-07-19)

Per-target timing showed that Evergate's dense transition renders the same 1080p VideoOut target in four
graphics spans separated by compute. The first three callbacks cannot publish a frame, but each previously
copied the whole scanout to CPU memory before the fourth and final span repeated that readback. Persistent
color-target queue order already preserves those writes, compute consumers can materialize a target lazily,
and a DMA producer is explicitly marked authoritative by the ordered executor.

Intermediate non-authoritative scanout spans now remain GPU-resident. The final callback either uses the
normal readback from a later scanout pass or materializes the cached target once if the submit ended on a
different target. Deferred scanouts are pinned against the backend's bounded LRU eviction until final
materialization, while guest writes may still invalidate their pixels.
`PROSPER_NO_INTERMEDIATE_SCANOUT_DEFER=1` restores per-span scanout readback for a same-binary comparison.

Native Windows / RTX 4090, one RelWithDebInfo binary, separate fresh saves, native scale/cadence, and the
documented Evergate route produced 104 dense submits in each run:

| Measurement | Per-span scanout readback | Intermediate defer |
|---|---:|---:|
| Median total draws / submit | 474 | 474 |
| Median backend execution | 77.5 ms | 72.0 ms |
| Median GPU fence wait | 37.2 ms | 35.3 ms |
| Median readback | 6.2 ms | 2.7 ms |
| CPU readbacks / submit | 6 | 3 |

Matched 25-submit windows near 475-482 draws improved from 127.8 ms to 124.8 ms end to end. Both runs
completed the fresh-save route at 1920x1080 and produced direct composited frames. The remaining structural
cost is the fence wait after every backend call; removing it requires deferred lifetime management for all
call-local Vulkan resources and is intentionally outside this tranche.

## Evergate persistent pipeline layouts (2026-07-19)

Dense Evergate submits still recreated the same small set of pipeline layouts in each backend call. The
backend now retains pipeline layouts by the complete ordered descriptor contract. Descriptor pools, sets,
set layouts, contents, images, and buffers remain call-local; a cache hit changes only the immutable pipeline
layout lifetime. The cache defaults to 256 entries and evicts the least-recently-used layout not referenced by
the current call. `PROSPER_PIPELINE_LAYOUT_CACHE_ENTRIES=<N>` changes the bound and
`PROSPER_NO_BACKEND_PIPELINE_LAYOUT_CACHE=1` restores call-local creation.

A native-Windows A/B/A used one final binary, separate fresh saves, native scale/cadence, the documented
route, and submit-aligned windows matched at 472-488 realized draws:

| Mode | Draws | Pipeline-layout setup | Whole submit |
|---|---:|---:|---:|
| Cache-disabled control 1 | 474-488 | 1.78-1.82 ms | 254-265 ms |
| Bounded cache | 472-484 | 0.23-0.37 ms | 199-214 ms |
| Cache-disabled control 2 | 474-487 | 1.33-1.52 ms | 180-185 ms |

The reversed control exposes substantial warm-state variance in the other timing buckets, so the whole-submit
ranges are not an FPS claim. The isolated layout bucket consistently removes about 1.1-1.5 ms from a dense
submit. This is a bounded CPU improvement; GPU fence waits and draw/resource realization remain the dominant
Evergate costs.

## Evergate backend resource references and texture residency (2026-07-19)

A deferred-submit prototype proved that removing intermediate CPU waits alone is not sufficient. It reduced
the readback-free wait bucket from roughly 33 ms to below 1 ms, but retaining complete call-local resource
graphs made the matched backend window slower (77.6 ms versus 71.9 ms) and produced fewer presented frames.
The prototype was removed. Future multi-target submission work must share command/resource ownership rather
than queueing several independently realized calls.

The next exact phase probe found avoidable work inside each independent call. A 462-draw Evergate pass spent
8.09 ms deep-copying every draw's `FrameResource` vectors into backend bookkeeping, although later recording
needed only descriptor sets, pipelines, index buffers, and scissors. Using the immutable `BackendDraw`
resources by reference and keeping descriptor-construction arrays local reduced the closest 460-draw sample's
draw setup from 43.60 ms to 22.21 ms; its GPU wait remained approximately 27-28 ms.

Texture residency was the next larger mechanism. The dense pass referenced about 960 texture bindings but
only 43-44 distinct exact image/view/sampler contracts. The old 256 MiB backend image budget thrashed that
working set. Even a 512 MiB control sat at 502.3 MiB and still evicted and reuploaded a 16 MiB image on each
dense call. The default backend image budget is therefore 1 GiB, matching the already-bounded frontend decode
budget; this is a cap, not a preallocation. A warmed 1 GiB run reported 43 persistent image hits, zero uploads,
and about 2.8 ms of texture-binding work in a 479-draw pass.

Exact image-view/sampler bindings now live with their persistent image instead of being recreated on every
callback. Each image retains at most 32 complete contracts and evicts only a binding not referenced by the
current call. `PROSPER_NO_BACKEND_PERSISTENT_TEXTURE_BINDINGS=1` restores callback-local bindings, while
`PROSPER_BACKEND_TEXTURE_CACHE_MB` still controls image residency. The Vulkan regression test covers the
initial binding miss, the next-call hit, byte-identical output, and bounded eviction without destroying the
current call's binding. Later end-to-end controls coincided with unrelated GPU saturation, so the cache and
CPU phase counters above are used for attribution rather than presenting those wall-clock rates as FPS gains.

## Evergate direct frontend buffer views (2026-07-19)

After parallel draw realization, dense Evergate submits still constructed roughly 3,300 storage-buffer
resources for only about 4 MiB of logical bytes. The frontend allocated a zeroed byte vector, copied guest
memory into it, then allocated and copied again into each `FrameResource`; the backend immediately hashed and
uploaded those immutable bytes synchronously. Reusing materializations by guest identity was rejected after
an exact A/B: only about one third of references repeated and the hash-map/ownership overhead made the buffer
bucket 2-3 ms slower.

The final path instead reuses the executor's mapping-generation-scoped readable-range cache. A complete
readable guest or capture range becomes a non-owning immutable `FrameResource` view; the backend hashes and
copies it into the mapped Vulkan upload arena before returning. Submission batching retains only the Vulkan
objects after that point. Partial or unreadable ranges keep the existing zero-filled owned-copy fallback, and
`PROSPER_NO_FRONTEND_BUFFER_VIEW=1` restores materialization for comparison.

A native Windows / RTX 4090 A/B used one final RelWithDebInfo binary, separate fresh saves, native 1920x1080
rendering on every submit, the documented read-anchored Evergate route, and six screenshots spaced over 360
rendered frames. The table averages the two closest dense 25-submit windows (about 489/479 and 490/479 draws):

| Measurement | Materialized control | Direct views |
|---|---:|---:|
| Draws / submit | 484.1 | 483.8 |
| Frontend resource construction | 46.3 ms | 34.9 ms |
| Buffer construction bucket | 13.1 ms | 5.1 ms |
| Frontend callback total | 86.8 ms | 74.4 ms |
| Backend execution | 39.1 ms | 38.5 ms |
| Complete 360-frame route | 43.43 s | 39.62 s |

The dense windows used direct views for all but roughly two of 3,300 buffer references and materialized less
than 0.05 MiB per submit. Backend time remained flat, while frontend construction fell by about 11.4 ms and
the complete route improved by 8.8%. Exact renderer snapshots remain the semantic guard; the next dominant
Windows cost is cross-submit texture validation, which still compares roughly 129 MiB of encoded source per
dense frame because page-protection write watches are not ABI-safe there.

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
The Vulkan backend retains only images with such an ID, under a bounded cache.
Render targets, storage images, captured host backing, and unvalidated formats never receive an ID.
Exact view and sampler contracts remain independent, so descriptor swizzles and filtering are unchanged.

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

## Persistent decode ownership and cache accounting (2026-07-15)

The persistent decoded-texture cache previously copied each newly decoded image out of a reusable scratch
vector. The scratch vector retained its allocation even though all future uses referenced the persistent
copy. Successful cache insertion now moves that allocation into the cache. Uncached and mutable resources
still retain reusable scratch storage so steady rendering does not allocate every frame.

Native Windows cache accounting shows the result:

| Measurement | Copy insertion | Ownership transfer |
|---|---:|---:|
| Loading-loop decode scratch | 141-160 MiB | 0 MiB |
| Post-parse decode scratch | 160 MiB | 49.8 MiB |
| Loading-loop private memory | about 1.58 GiB | about 1.42 GiB |
| Post-parse private memory | about 1.89 GiB | about 1.84 GiB |

The smaller post-parse process delta includes variation in the workload's other bounded caches: the later
run held 27 decoded textures, 15 RTT surfaces, a 359 MiB graphics pool, and a 49 MiB compute pool. Its final
45 seconds stayed near 1.84 GiB private and 3.82 GiB working set. RTT retention is not the leak hypothesis:
it stabilized at 101.8 MiB. `PROSPER_RENDER_TIMING=1` now reports RTT, decode-scratch, and validation-scratch
storage so later titles can distinguish cache growth from an unbounded process.

Detailed timing also identifies why resource construction remains expensive. Exact validation of unchanged
linear atlases costs about 7.3 ms for one 4096x4096 image, 1.8-2.1 ms for one 2048x2048 image, and about 1 ms
for one 1024x2048 image on every graphics span. `PROSPER_RENDER_TIMING=detail` labels each slow texture as a
local reuse, persistent hit/miss/invalidation, RTT source, or uncached decode. Replacing exact validation
requires write-aware invalidation; probabilistic sampling is not an acceptable correctness shortcut.

## Same-submit write-aware validation (2026-07-15)

Ordered execution now opens a backend-neutral, bounded journal for each synchronous submit. Compute buffer
and storage-image writeback already reports exact guest ranges through `notify_guest_gpu_write`; later
graphics spans query only events after the cache entry's last successful validation. An unrelated write proves
the source unchanged, an overlap forces an exact comparison, and a different submit, inactive scope, nested
execution, or 4,096-event overflow also falls back to exact comparison. DMA_DATA and WRITE_DATA currently
execute during command-buffer folding before ordered backend execution, so the first graphics span's exact
comparison observes them; they are not silently treated as between-span events.

Cross-submit validation is intentionally unchanged. Guest CPU writes need authoritative dirty-page tracking
before the exact comparison can be removed there. `PROSPER_NO_SUBMIT_TEXTURE_VALIDATION_REUSE=1` disables the
same-submit shortcut. `PROSPER_AUDIT_SUBMIT_TEXTURE_VALIDATION_REUSE=1` exercises every proposed shortcut but
still performs the old exact comparison. A 180-second native Windows Dead Cells audit reached `PrisonStart`,
completed `PARSEALL`, exercised 1,196 cumulative shortcuts, and reported zero disagreements.

The fully rendered route is animated and the compared four-span windows contained different draw counts, so
whole-submit figures are directional rather than a strict benchmark. The resource-specific reduction is
directly counted:

| Four-span window | Exact validations/submit | Validated bytes/submit | Texture resource time | Resource build |
|---|---:|---:|---:|---:|
| Audit, exact comparisons retained | 28 | 275.8 MiB | 40.2-40.4 ms | 49.9-50.3 ms |
| Write-aware shortcut enabled | 22 | 145.1 MiB | 30.5-30.7 ms | 40.8-41.2 ms |

The audit window's whole submit was about 202 ms with 348 draws; the optimized window was 196-199 ms with
386 draws. Reported app rate remained about 4.3 FPS, confirming this is a bounded frontend improvement rather
than the dominant remaining renderer cost. The optimized run ended stable near 1.81 GiB private and 3.79 GiB
working set. The next large performance work remains persistent Vulkan object reuse and fewer synchronous
graphics/compute/readback boundaries.

## Submit-aligned Vulkan timing

The original backend timing window averaged every 25 calls independently. Dead Cells currently makes four
graphics callbacks in an ordered submit, and each callback can render multiple targets, so those backend
windows could straddle scene and submit boundaries. The frontend now records the structured timing result of
every completed `render_draws_rgba` call and sums it into the same 25-submit window used by its resource and
wall-time counters. The new `backend-submit` lines report calls and draws per submit plus target setup, draw
setup, record/upload, fence wait, readback, cleanup, and the shader/fixed/resource/pipeline draw-setup split.
They also print the frontend-measured backend duration, the detailed phase sum, and the unattributed remainder.
This is the authoritative view for choosing the next backend optimization; the legacy per-call lines remain
useful for spotting an individually slow render target. `PROSPER_RTT_TIMING=1` adds a lightweight
`[rtt-timing]` record for each target group with its submit, address, dimensions, draw count, and exact phase
costs. Each record also reports its frontend-measured duration and the remainder outside those phases, so an
unattributed submit-level cost can be assigned to a concrete target call. `PROSPER_RTTLOG=1` retains its visual
pixel/draw diagnostics and also emits the timing record. Bound both
modes with `PROSPER_RTTLOG_MIN_SUBMIT` and `PROSPER_RTTLOG_MAX_SUBMIT`. The full visual mode scans rendered
pixels and dropped one Dead Cells title loop from about 20 FPS to 13-14 FPS, causing its wall-clock input route
to miss the menu; use the lightweight mode for performance attribution.
Lightweight records selected for one submit are emitted with a single stderr write while preserving their
line-oriented format, avoiding one slow Windows console operation per target.

The legacy 25-backend-call window is now separately enabled with `PROSPER_BACKEND_TIMING_WINDOWS=1`.
Printing its multiple aggregate lines from inside `render_draws_rgba` was charged to whichever target crossed
the 25-call boundary. One Dead Cells 636x420 five-draw target measured 103.15 ms outside but only 2.55 ms across
all backend phases; 100.60 ms was diagnostic output. This explained almost exactly the submit-aligned 39 ms
unattributed remainder. Submit-aligned and lightweight target timing remain enabled by
`PROSPER_RENDER_TIMING` without that legacy output.

Submit ordinals also varied enough across fresh runs that preselected ranges repeatedly missed the transition.
`PROSPER_RTT_TIMING_MIN_DRAWS=N` therefore buffers lightweight target records until the final graphics span and
emits the complete submit only when its total backend draw count reaches N. A value of 300 selects the current
357-373-draw Dead Cells workload while producing no output for the earlier 54-56-draw loop. The buffer contains
only target metadata and timing scalars; rendered pixels and backend execution are unchanged.

A 180-second native Windows fresh-save Dead Cells run reached the post-parse workload with 373 draws, eight
dispatches, and four graphics spans. Those spans rendered ten target groups per submit. The aligned backend
window was:

| Vulkan work per submit | Time |
|---|---:|
| Frontend-measured backend wall time | 90.09 ms |
| Detailed phase sum | 75.28 ms |
| Fence waits | 33.12 ms |
| Draw setup | 26.54 ms |
| Pipeline creation (inside draw setup) | 12.89 ms |
| Descriptor/resources (inside draw setup) | 8.46 ms |
| Readback | 9.43 ms |
| Record/upload | 4.59 ms |
| Target setup + cleanup | 1.59 ms |
| Unattributed backend wrapper time | 14.81 ms |

The whole ordered submit remained about 201 ms: 36 ms draw realization and 165 ms backend execution. Private
memory stepped up with the workload, then stayed near 1.81 GiB through the end; working set stayed near
3.78 GiB. This rules out unbounded growth in that window and shows that no single small setup cache can make
the workload playable. The next investigation must identify the true dependencies among the ten target calls;
persisting pipelines addresses about 13 ms, while removing unnecessary synchronous wait/readback boundaries
has the larger ceiling but requires preserving graphics/compute and RTT producer-consumer order.

## Persistent graphics pipelines

The backend now retains Vulkan graphics pipelines across target calls with an exact, bounded key. The first
prototype copied and hashed both complete SPIR-V modules on every draw; despite a 100% hit rate, that made a
54-draw loading submit slower. The accepted path assigns each entry in the exact shader-recompile cache a
process-unique identity that is never recycled, even when that cache is cleared. Live pipeline keys use those
two identities, an inline allocation-free fixed-state key, and the descriptor contract already named by the
shader identities. Externally constructed, captured, and replayed draws have identity zero and retain the full
SPIR-V plus descriptor-layout fallback. Hash collisions are benign because equality compares the exact key.
Pipeline hits also skip temporary `VkShaderModule` creation.

The cache defaults to 1024 entries and evicts the least-recently-used pipeline not referenced by the current
backend call. `PROSPER_PIPELINE_CACHE_ENTRIES` changes the bound and
`PROSPER_NO_BACKEND_PIPELINE_CACHE=1` restores transient creation. Timing output reports references, hits,
misses, bypasses, entries, and evictions at both backend-call and submit-aligned scopes.

Native Windows / RTX 4090, fresh saves, current master including #750, extended full-render input hold, and
the workload-filtered target profiler produced nearby animated heavy windows rather than identical draw
counts. The state-specific setup buckets are therefore the useful comparison:

| Mode | Draws | Shader modules | Pipeline work | Total draw setup | Backend wall time |
|---|---:|---:|---:|---:|---:|
| Cache disabled | 378 | 3.27-3.35 ms | 14.96-15.75 ms | 30.19-31.51 ms | 106.45-110.35 ms |
| Persistent cache | 359 | 0.00 ms | 10.05-10.24 ms | 20.89-20.99 ms | 95.63-97.88 ms |

After normalizing the setup buckets for the draw-count difference, retained pipelines remove about 7-8 ms
per heavy submit. The 54-draw loading loop improves by only about 1 ms because the Windows NVIDIA driver
already makes repeated transient creation cheap; the larger expected benefit on MoltenVK remains to be
measured. The enabled run held 30 pipelines in the heavy scene. Its final private-memory range was
1.77-1.80 GiB and its working-set range was 3.74-3.76 GiB, indistinguishable from the disabled control at
1.79-1.80 GiB / 3.74 GiB. Both extended routes reached the post-parse workload. Per-target logging materially
perturbs whole-submit time, so the reported application FPS from these diagnostic runs is not a normal-play
benchmark.

This closes the measured pipeline-creation tranche, not the renderer problem. Ten target calls still perform
synchronous GPU waits and readbacks, and the whole ordered submit remains hundreds of milliseconds under the
target profiler. Coordinated GPU ownership across those producer-consumer boundaries is the next architectural
step.

## Persistent GPU color targets (2026-07-15)

The first coordinated-ownership tranche retains exact RGBA8 color targets by guest identity, extent, and
format in a bounded Vulkan cache. A later graphics pass that samples the same target can bind that image
directly instead of reading it to CPU RGBA and uploading it again. Guest GPU writes invalidate overlapping
entries through the same ordered write observer used by persistent depth/stencil state. Same-target feedback,
scanout, presentation fallback, captures, replay seeds, and pixel diagnostics keep the established CPU path.

The live path is enabled by default as of 2026-07-19. Set
`PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1` for a complete frontend A/B,
`PROSPER_NO_BACKEND_PERSISTENT_COLOR_TARGETS=1` to disable backend retention independently, or change its
256 MiB budget with `PROSPER_BACKEND_TARGET_CACHE_MB`. Captures, per-target pixel diagnostics, scanout,
same-target feedback, and authoritative-readback spans retain the established CPU path. The backend unit
contract compares direct GPU producer-to-sampler output byte-for-byte with CPU readback/upload, verifies a
deferred-readback LOAD pass, and proves that invalidation uses supplied CPU pixels rather than stale GPU
contents.

Native Windows Dead Cells post-`PARSEALL` evidence used one current-master binary with submit-aligned timing.
The animated runs did not carry identical draw counts, so the phase counters establish the mechanism more
reliably than the app overlay:

| Measurement | CPU target path | Persistent GPU targets |
|---|---:|---:|
| Realized draws | 361 | 405 |
| Target writes / readbacks | 10 / 10 | 10 / 4 |
| Direct target samples | 0 | 8 |
| Record/upload | 4.5 ms | 0.5 ms |
| GPU fence waits | 26.7 ms | 16.8 ms |
| Frontend Vulkan wall time | 60.8 ms | 48.9 ms |
| Ordered backend total | 118.6 ms | 110.7 ms |
| Whole submit | 154.2 ms | 151.1 ms |

The enabled run retained 13 targets / 103.5 MiB and remained near 1.81 GiB private memory. It did more draw
work while completing slightly faster, but this is not yet a playable frame budget. Four large CPU readbacks
still cost about 9 ms, 405-draw realization costs about 40 ms, and the ordered path still executes ten target
calls plus eight compute calls. The next architectural gain requires fewer synchronous submissions/fences or
safe reuse of per-draw resource work; neither may weaken the captured graphics/compute dependency order.

The default-on decision used a paired native-Windows Evergate fresh-save route on one current-master binary
after call-local resource sharing. Both runs used native 1920x1080 targets and the same 75-second controller
script. Animated windows are not draw-identical, so the mechanism counters and ranges matter more than any
single sample:

| Measurement | CPU target path | Persistent GPU targets |
|---|---:|---:|
| Presented frames in 75 seconds | 314 | 373 |
| Observed heavy-scene rate | about 1.9 FPS | about 2.7 FPS |
| Target writes / readbacks / deferred | 0 / 14 / 0 | 14 / 6 / 8 |
| Direct target samples | 0 | 19 |
| Heavy whole-submit windows | 450-475 ms | 317-359 ms |
| Heavy frontend Vulkan work | 252-285 ms | 159-194 ms |
| Heavy GPU fence waits | 59-61 ms | 34-37 ms |
| Heavy record/upload | 13-15 ms | 1-2 ms |

This is still far from the 16.7 ms frame budget, but it removes repeated GPU-to-CPU-to-GPU ownership
round-trips from normal runs and makes the remaining per-draw realization and transient object costs clearer.

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
Shared CPU publication is complete; direct GPU-image presentation remains a later step. As of #1091
phase 1 the compute backend ADOPTS the live renderer's Vulkan device instead of creating its own, so
the two no longer own separate devices when the live renderer is registered; binding a renderer-owned
image directly to a dispatch (phase 2) is what still remains.

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

## Windows cross-submit texture write watch

The protection-fault implementation described in earlier revisions of this handoff is retired. Windows builds
an exception-dispatch frame below the interrupted stack pointer before a vectored handler runs. Unmodified PS5
code follows the SysV ABI and may keep live values anywhere in that 128-byte red zone, so even a successfully
handled read-only-page fault can silently overwrite guest locals. Evergate exposed this by saving valid output
pointers in the red zone, taking several direct-memory first-touch faults, and later reloading a null pointer.

Windows `GuestWriteWatch` therefore reports unsupported and every cross-submit texture lookup uses the exact
byte-comparison fallback. The prior audit evidence remains useful evidence that the dirty-page algorithm was
logically sound, but it cannot make Windows exception delivery ABI-safe. Re-enable protection watches only if
fault delivery itself preserves all guest red-zone bytes. Direct memory now uses a delete-on-close sparse file,
so the kernel demand-pages mapped file data without the unsafe user-mode `SEC_RESERVE` first-touch exceptions.

## Separate unresolved risk

Native Windows boot can still intermittently stop after exactly 75 submits while asset loading and
GC suspension continue. The same binary and route can pass on a retry. This is tracked separately in
[#712](https://github.com/mattias800/ps5ys/issues/712); do not misclassify it as a renderer-cache
regression without checking the submit count and suspend logs.

## Renderer-owned RTT sampling: direct image binding (#1091 phase 1 / #1095 phase 2)

The renderer and the compute backend historically created SEPARATE Vulkan devices, so a surface the
renderer owned could not be handed to a dispatch at all. Sampling one cost a full GPU->CPU readback of
the persistent color target, a per-texel conversion, a fresh `VkImage`, and a re-upload of those same
pixels to the GPU. `live_renderer.cpp` recorded the constraint directly: "Compute cannot import that
color attachment directly."

Phase 1 (#1091, merged as #1092) removed the premise: the renderer publishes its `VkDevice`, physical
device, queue and queue family through `SharedVulkanContext`, and compute ADOPTS that context when it
is present and feature-adequate. An adopted context is *borrowed* -- the consumer destroys none of it.
Compute still creates its own device when no renderer is registered, so headless compute-only use is
unaffected. Phase 1 is an enabler, not an optimization: a 3-rep A/B measured no throughput change
(7.74 ms / 45.1 fps shared vs 7.85 ms / 46.0 fps separate), which is the intended result.

Phase 2 (#1095) takes the win. **Measure the binding mix before designing the import.** Instrumentation
showed renderer-owned compute bindings run about **1000 sampled to 1 storage** -- essentially all
read-only. That deleted the two hardest requirements from the original sketch: no
`VK_IMAGE_USAGE_STORAGE_BIT` is needed (`SAMPLED_BIT` is already set on persistent targets), and there
is no guest writeback contract to preserve, because nothing is written. Every renderer-owned sampled
binding in the measured route had a single shape: `Unorm8` x4 against an rgba8 target.

Two rules keep the fast path honest, and both are load-bearing:

- **Authority.** The import is offered only while the persistent Vulkan image is the authoritative copy
  -- exactly when the existing reader would have had to materialize it (`!surface.rgba` or a size
  mismatch, and `surface.gpu_valid`). The CPU RTT copy and the GPU image are kept coherent by
  *invalidation*, not by one always being fresher, so whenever a CPU snapshot exists it may be the newer
  truth (#780) and the snapshot path must still be used. Importing unconditionally would silently
  resurrect stale pixels.
- **Exactness.** Only a sampled, non-aliased, single-layer `Unorm8` x4 view whose extent, format and
  device all match falls through. That shape's host path is a plain `memcpy` into a
  `VK_FORMAT_R8G8B8A8_UNORM` image, which is precisely what the direct bind produces, so the fast path
  is byte-identical rather than merely close. `rgba16f` targets deliberately keep the host path: the
  format selector has no RGBA16F option and converts them down to UNORM8.

The borrowed image gets its own view (preserving T# `DST_SEL` swizzle routing), is barriered into the
`GENERAL` layout the descriptors declare, and is restored to the layout its owner left it in so the
renderer's own tracking stays true. The cache entry is pinned per successful import and released per
import, including on every failure path. `PROSPER_NO_DIRECT_RTT_BIND=1` forces the host path for A/B.

Functional result, Blasphemous 2 title route: **731 direct binds, zero fallbacks to the host path,
zero validation errors** -- every renderer-owned sampled binding in that route took the fast path.

Throughput, measured on an **idle** machine through `tools/perf/ab_compute.sh`, 3 alternating reps of
150 s on `load-save-first-station.pad` (a real gameplay scene, not a menu):

| | compute, run-wide | compute, gameplay tail | gameplay frame rate |
|---|---|---|---|
| host path (`PROSPER_NO_DIRECT_RTT_BIND=1`) | 7.60 ms | 7.84 ms | 13.0 fps |
| direct bind | 5.94 ms | 5.88 ms | 14.0 fps |
| | **-21.8%** | **-25.0%** | **+7.7%** |

Every enabled run beat every disabled run on all three metrics, and the compute memory pool drops
**80.1 -> 63.7 MiB** (9 -> 7 cached allocations) as the per-dispatch staging buffer and the duplicate
image disappear.

**A 25% stage win is a 7.7% frame win, and the two must never be reported interchangeably.** Frame
time goes 76.7 -> 71.2 ms, about 5.4 ms saved per frame.

**The accounting does NOT fully close, and an earlier revision of this section wrongly claimed it did.**
The route runs 5.28 compute calls per frame, so removing 1.66 ms from each predicts an **8.5 ms** frame
saving; only **5.4 ms** (64%) materialised. Measured non-compute time per frame *rose* 36.7 -> 39.8 ms
across the arms. The most likely confound is the fixed wall clock: both arms run 150 s, so the faster
arm renders more frames (2450 vs 2298) and therefore progresses FURTHER along the route, rendering
different -- and here evidently heavier -- content. A fixed-wall-clock A/B does not compare identical
scenes. Treat the 5.4 ms as the honest observed figure and the 8.5 ms as an upper bound; closing this
gap needs a fixed-WORK route (equal frame counts or a fixed scene) rather than a fixed-duration one.

### Distance to a playable frame rate

| | current | 30 fps | 60 fps |
|---|---|---|---|
| frame time | **71.2 ms (14.0 fps)** | 33.3 ms | 16.7 ms |
| required | -- | **2.14x faster** (remove 37.9 ms) | **4.27x faster** (remove 54.5 ms) |

Composition of the current 71.2 ms frame: **compute 31.4 ms (44%)** (5.28 calls x 5.94 ms) and
**everything else 39.8 ms (56%)**.

**Compute alone is 94% of a 30 fps budget and 188% of a 60 fps budget.** Even if every non-compute cost
went to zero, compute as it stands would still miss 60 fps by roughly 2x. So 60 fps is not reachable by
optimising the non-compute remainder: the number of dispatches per frame, or their cost, has to fall by
a large multiple. That is a different class of work from the per-dispatch savings in #1091/#1095 --
which have now taken the per-call cost from 7.60 to 5.94 ms and removed the readback/re-upload
entirely, leaving the remaining per-call cost to be attacked structurally.

**Measure gameplay, not only menus.** An earlier pass measured the title-screen route and reported
about 13%. Gameplay is a different draw and dispatch mix and showed a *larger* win, but the direction
could equally have gone the other way -- a change that helps a UI-heavy scene can be neutral once real
geometry and lighting dominate. Report the gameplay portion separately rather than averaging it with
loading and menus: `ab_compute.sh` prints both the run-wide mean and the mean of the trailing
`[render-window]` rolling averages for exactly this reason (#1082).

**Record the conditions, not just the number.** The first title-route measurement was taken while an
interactive session was using the same GPU, which makes it unattributable -- performance has no
equivalent of ctest's exit code. `ab_compute.sh` therefore refuses to run when another `prosper-app`
is alive and stamps the commit, route, reps and duration onto its output.

## Next renderer step

Bring up a representative Unreal/3D workload and collect the same stage timings plus a corrected
mixed-operation capture. Then design one cross-engine persistent-resource change around measured
resource identities and dependencies. Keep the exact-byte and disable-switch A/B discipline used
here, and preserve screenshot/capture correctness while reducing the synchronous boundaries.
