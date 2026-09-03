# Renderer performance and tooling findings (2026-07-14)

> **Native-Linux continuation:** the current Evergate performance frontier, reproducible dense-route
> baseline, rejected experiments, and ranked continuation plan are recorded in
> [`EVERGATE_PERFORMANCE_HANDOFF_2026_07.md`](EVERGATE_PERFORMANCE_HANDOFF_2026_07.md).

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

> **These figures are stale, and not because they were measured badly.** They were taken on
> 2026-07-14 with `prosper-app` (see *Measurement method* below), which was the shipped path at the
> time. #1270 replaced it ten days later, on 2026-07-24: the app now blits the renderer's front-buffer
> image straight to the swapchain and the renderer skips the CPU readback entirely, so every number
> above describes a renderer that no longer exists. A windowed, uncapped run of the same title on
> 2026-08-27 (`--present-mode immediate`) reports a median of **156 fps** over 468 samples, two
> thirds of them between 100 and 160 — a *presented* rate, so an upper bound on new frames rather
> than the frame rate itself (#3083). The *conclusion* of this pass is unaffected: the stop decision
> was about where the remaining cost is structural, not about the absolute rate.

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

The persistent texture cache now covers guest-backed linear/tiled 2D sampled `Unorm8`, `Float16`, and
`Float32` textures with supported component counts, plus BC1-BC7 textures. It excludes render targets,
storage images, cube/volume and DCC surfaces, captured host backing, and unsupported formats. Every hit
validates the complete native, padded-tiled, or compressed-block source range, so address reuse and
in-place mutation invalidate the entry. Its memory-aware default is one eighth of host physical
memory, clamped to 1-2 GiB. Disable it with `PROSPER_NO_TEXTURE_DECODE_CACHE=1`; the budget is controlled by
`PROSPER_TEXTURE_DECODE_CACHE_MB`.

## Blue Prince 3D submit decomposition, re-measured (2026-08-02, #1284)

Re-measured on `c79f742e` — i.e. **after** #1292 (demand-driven readbacks) and #1703 (submit-scoped
decode identity) — with `PROSPER_RENDER_TIMING=1` on the scripted Blue Prince fresh-save route into
the manor, RADV / Radeon 8060S (STRIX_HALO, integrated), native 1920x1080, no snapshot acceleration.
35 peer-free heavy `[render-window]` samples, median:

| term | med ms | share | July (#1284 body) |
|---|---:|---:|---:|
| `measured` (whole backend submit) | 165.18 | 100 % | 263.33 |
| `draw_setup` | 126.81 | **76.8 %** | 119.87 |
| — `resources` | **89.66** | **54.3 %** | 95.42 |
| — `fixed` | 24.40 | 14.8 % | 16.43 |
| — unattributed inside `draw_setup` | 6.17 | 3.7 % | — |
| — `pipeline` | 1.43 | 0.9 % | 1.22 |
| `readback` | 14.71 | 8.9 % | 79.16 |
| `cleanup` | 13.19 | 8.0 % | 11.03 |
| `gpu_wait` | 6.71 | 4.1 % | 29.29 |
| — `gpu_device` (real GPU time) | **4.31** | **2.6 %** | — |
| `record_upload` | 2.73 | 1.7 % | 22.58 |
| `fence_waits` per submit | 16.3 | — | 61 |

The scene is now **2,109 draws/submit**, not July's 1,516, so `resources` improved more per draw than
the totals show. **Every term the original decomposition named has collapsed except `draw_setup`,**
which is unchanged and is now 77 % of the submit. `gpu_device` 4.31 ms inside a 165 ms submit means
**2.6 % GPU utilisation**: this title is not GPU-, driver-, or shader-bound, it is bound in prosper's
own per-draw CPU path.

### `draw_setup.resources` is 93.7 % storage-buffer upload

`resources` never measured descriptor setup. The interval spans the whole per-draw resource block,
which also builds every texture upload and every storage-buffer upload. The sub-attribution added in
#1284 (`[render-window] backend-submit resources avg_ms: …`) splits it live:

| sub-term | med ms | share of `resources` |
|---|---:|---:|
| **`res.buffer`** | **103.51** | **93.7 %** |
| `res.other` | 3.01 | 2.7 % |
| `res.descriptor` | 2.25 | 2.0 % |
| `res.texture` (upload 0.75 / bind 0.18 / lookup 1.15) | 2.01 | 1.8 % |

`res_buffer` splits 4:1 into `copy` (the staging `memcpy`) **83.80 ms** over `acquire` (pool/arena
acquisition, which on a pool miss is create + allocate + bind + map) **13.64 ms**, remainder 7.64 ms
— the transient fallback path, which allocates and copies together and is attributed to neither, plus
key building and lookups.

**Do not read that split as "allocation churn is the small half".** `acquire` measures only the
allocation *calls*; most of the churn cost lands in `copy`, because a freshly `vkAllocateMemory`'d
and `vkMapMemory`'d staging buffer has non-resident pages and the memcpy into it faults, while a
recycled pool buffer is already resident. The sub-attribution draws its boundary at the API call, and
reading that boundary as the mechanism produced a wrong conclusion here once already.

Mechanism, from two independent counters in the same run. **Volume:** the frontend resolves every
binding to a zero-copy direct guest view and reports `buffers=34,747 logical=1,764.4 MiB
materialized=0.0 MiB` per submit — and the *backend* then `memcpy`s the deduplicated set into
host-visible staging anyway. The Evergate "direct guest backing" win was only ever realised on the
frontend side of that boundary. **Churn:** `backend_buffer_pool` sits pegged at its 256 MiB default
ceiling with evictions exactly equal to misses (+7,960 / +7,960 over 9 s, 45 % miss rate) — a capacity
thrash in which one buffer is destroyed for every one created, with the eviction victim taken as
`available.begin()` on an `unordered_map`, i.e. an arbitrary bucket rather than an LRU.

### The buffer pool's 256 MiB default is ~35 % of the frame

`render_host_buffer_pool_limit()` defaults to a **fixed 256 MiB**. Blue Prince's real host-staging
working set is **974 MiB**, so on a 3D title the pool degenerates into an allocator: evictions equal
misses (~216k each per arm) and one buffer is destroyed for every one created.

Four alternating arms, one binary, `PROSPER_BACKEND_BUFFER_POOL_MB` the only lever (256 / 2048 /
256 / 2048). Witness: the 256 arms peak at 256.0 MiB with 215,366 and 217,205 misses; the 2048 arms
settle at 974.0 MiB / 503 buffers with **503 misses and zero evictions**.

| term | 256 MiB | 2048 MiB | delta | within-treatment | between |
|---|---:|---:|---:|---:|---:|
| `measured` | 203.06 | 125.98 | -38.0 % raw | 10.62 | 77.09 |
| `res_buffer` | 113.10 | 41.50 | -63.3 % | 16.76 | 71.60 |
| `copy` | 92.54 | 33.27 | -64.0 % | 18.19 | 59.27 |
| `acquire` | 13.65 | 1.44 | -89.5 % | 0.95 | 12.22 |
| `cleanup` | 14.77 | 2.95 | -80.1 % | 0.14 | 11.82 |
| draws | 2218.15 | 2109.47 | +4.9 % | 35.10 | 108.67 |

Every timing term separates far beyond its within-treatment spread. **The draws row does not**: the
2048 arms sat at 2109/2110 draws against the 256 arms' 2201/2236, a systematic ~4.9 % lighter scene
(an expected selection effect — the faster arm covers more of a fixed-duration route and samples
different moments). So the raw -38.0 % overstates it. **Normalised per draw, 0.0915 -> 0.0597 ms,
the honest figure is -34.8 %** — quote that one.

Controls: per-submit byte volume is identical between arms (`logical=1,644.0 / 1,764.4 MiB`,
`materialized=0.0 MiB`), so the 2048 arm copies the same bytes faster. Minor faults sampled from
`/proc/<pid>/stat` (no ptrace, no perturbation) run at 150,911/s for the 256 arm against 48,827/s
and 97,398/s for the two 2048 arms — roughly 31,400 versus 6,200 and 12,200 per submit. The direction
holds, but **the two 2048 arms differ from each other by 2x**, nearly the size of the gap being
attributed, and explaining the whole `copy` delta by faults alone would need ~3 us per fault, well
above a generic anonymous-page fault. First-touch faulting on freshly mapped staging memory is
therefore recorded as the **leading hypothesis, not a finding**; page-table/TLB churn from ~216k
map/unmap pairs per run is the other candidate. The fix does not depend on which it is.

**Caveat: all four arms ran contended (one peer throughout).** Absolute milliseconds are inflated;
the comparison is internally controlled but the magnitude needs a clean-box confirmation.

The fix is bounded and **cannot change rendered output** — the same bytes are uploaded from the same
sources, only host staging recycling changes: make the default memory-aware (mirroring
`texture_decode_cache_limit_bytes`, with the `*_MB` override winning) and give eviction an LRU victim
instead of `available.begin()` on an `unordered_map`. The risk is memory footprint, not correctness.

This does **not** retire the copy, and the difference matters for what anyone expects next: the fix
stops the pool *thrashing while it copies*, it does not stop the copying. Even at 2048 MiB, `copy` is
33.27 ms and `res_buffer` 41.50 ms of a 125.98 ms submit — about **8 backend submits per second
against a bar of 30.** Removing the copy entirely is tracked as **#1733**: import guest pages as
Vulkan memory (`VK_EXT_external_memory_host`; prosper uses it nowhere today, there is no
`vkGetMemoryHostPointerPropertiesEXT` / `VkImportMemoryHostPointerInfoEXT` in the tree), or add a
versioned guest-identity buffer cache. The second reuses the #1703 machinery but carries the
#611/#780 stale-data risk; the first does not, because the GPU would read the guest bytes rather than
a snapshot of them.

Frame math, so nobody expects one term to finish the job: at the clean 165.18 ms submit, removing
100 % of `copy` gives ~95 ms (10.5 submits/s); removing `copy` **and** `acquire` **and** `fixed`
still leaves ~60 ms (16/s). `readback` + `cleanup` + `gpu_wait` + `record_upload` are 37 ms combined
and `gpu_device` is 4.31 ms, so **no single term reaches 30 fps on this title.**

### Ruled out

- **Per-draw descriptor setup is not the term.** `res.descriptor` — layout lookup,
  `vkAllocateDescriptorSets`, `vkUpdateDescriptorSets` — is 2.25 ms across 2,152 draws (~1 µs/draw),
  2.0 % of `resources`. This is the measured reason #1284 round 1's descriptor-set-reuse experiment
  came back neutral: it was optimising 2 % of the term. Do not re-open it.
- **Texture handling is not the term live.** `res.texture` is 2.01 ms; #1691/#1703's caches work.
  Note the trap: the *identical instrument* on an offline `.prgcap` replay of the same title reports
  `res.texture` = 762.72 ms, because replay disables the persistent decode cache by construction
  (`texture_decode_cache_candidate()` requires `!has_captured_host_data`). **A replay texture number
  is not evidence about live texture cost, in either direction.**
- **Un-cached `getenv` in the per-draw loop is not a term.** The loop makes 11 un-cached `getenv`
  calls (`PROSPER_NO_CULL`, `NO_BLEND` x3, `DSLOG`, `STENCILLOG`, `STENCIL_MIRROR`,
  `STENCIL_REPLACE`, `NO_DEPTH_BIAS`, `FLIP_FRONT_FACE`, and `NO_SWIZZLE` which is per *texture
  reference*), sitting next to three neighbours that already cache into `static const`. Measured
  directly: glibc `getenv` on a miss costs 19.7 ns (10-var environment) to 25.6 ns (94-var), so the
  whole per-submit volume is **≈0.5 ms of 165 ms — 0.3 %**. Tidy if touched for other reasons; not a
  performance fix.

  Recorded at length because of how it *looked*: 11 uncached environment lookups in the hottest loop
  in the renderer, one of them per texture reference, sitting beside three neighbours that already
  cache into `static const` — an obvious oversight with an obvious fix, and it was minutes from being
  "fixed". A two-minute microbenchmark priced it at nothing. glibc's `getenv` compares the first
  character before `strncmp`, so it is tens of nanoseconds, not the microseconds the shape of the
  code suggests. **A plausible optimisation that measures to nothing is worth writing down precisely
  because the next reader will find the same 11 calls and draw the same conclusion.** Price a hot-loop
  call before removing it, not after.
- **The #1268 small-buffer content hash is not the term.** Live `hash-stats`: 414,633 hash calls over
  168.1 MiB against 3.5 M references — ~1 % of the buffer term.
- **"Texture handling is not the term live" is true of *The Messenger*'s shapes and NOT of a title
  that binds a large mutable surface.** Same instrument, Stray at its title screen (2026-09-04,
  `fc21d46ca`): the frontend texture leaf is **1110.1 ms of a 5020 ms window**, the largest single
  leaf in the capture, against `gpu-device` at 232.5 ms. The row above is not withdrawn — `res.texture`
  is the BACKEND's bucket and it is still small (206.7 ms here) — but it has been read as "textures
  are solved", and the frontend materializer is a different layer with a different answer. What
  separated them was printing the frontend's own cache-outcome classes, which the renderer had
  recorded since #2250 and the report never printed: 930.8 of the 1110.1 ms was in a single unnamed
  residual. **Before concluding a leaf is small, check which layer's leaf you are reading, and check
  the report is printing the sub-classes that exist.**

## Plucky Squire maximum-size atlas retention (2026-07-29)

Plucky Squire exposed a capacity cliff rather than a decode-kernel regression. The title first holds
about 590 MiB of decoded textures, then introduces a valid 14,208x13,552 BC3 atlas. Its compressed
source plus RGBA8 decode occupies about 930 MiB, so the former fixed 1 GiB frontend budget could not
retain both sets. It repeatedly decoded 3.1-3.5 GiB of texture references per submit, spent
0.7-2.9 seconds in frontend resource building, and recreated write watches over millions of pages.
The app fell from 15-20 FPS to 0.4-0.8 FPS despite zero render-target readbacks.

Matched native-cadence A/B runs showed that a 2 GiB frontend ceiling retained the measured
1.52-1.64 GiB hot set: after one cold decode, frontend resource building returned to about 5-7 ms.
Raising the independent backend sampled-image ceiling to 2 GiB then reduced steady backend resource
setup from about 10-20 ms to 1.2-1.4 ms, producing roughly 17-20 FPS in the same scene. These are caps,
not preallocations. Both defaults now use one eighth of their relevant physical/device-local memory,
clamped to 1-2 GiB, so an 8 GiB host or GPU retains the historical 1 GiB behavior and both explicit
MiB overrides retain precedence.

A follow-up 4 GiB frontend A/B showed that the remaining roughly 300 reported cold misses per submit
were not capacity misses: fewer than 3,000 actual decodes occurred over 800 frames. They were sampled
depth attachments whose authoritative pixels already lived in retained Vulkan images. The frontend
was probing the guest-byte decode cache and copying DCC metadata before the sampled-depth bridge won
later in the same resource path. Resolving that bridge first reduced the false misses to effectively
zero and lowered steady texture resource handling from about 5.6-6.3 ms to 3.7-4.2 ms per submit,
without changing the cache budget or depth-image binding contract.

## Plucky Squire pipeline working set (2026-07-29)

Plucky's desk-loading scene uses about 2,500 distinct exact graphics-pipeline contracts. The former
1,024-entry backend limit continuously evicted hot pipelines after the scene crossed that boundary:
matched 25-submit windows reported 2-10 evictions per submit and recurring multi-millisecond pipeline
setup spikes. This was cache-capacity churn, not shader recompilation or a correctness failure.

A matched scripted A/B with a 4,096-entry limit grew to 2,482 resident pipelines with zero evictions.
Once the compulsory creations completed, pipeline setup fell to about 0.07 ms per submit window; the
loading scene improved from roughly 3-4 FPS to 4.5-5.6 FPS on the Radeon 8060S test laptop. The larger
limit retains exact keys and LRU behavior, allocates nothing until a distinct contract is encountered,
and remains configurable through `PROSPER_PIPELINE_CACHE_ENTRIES`.

## Plucky Squire adaptive write watches and sampled volumes (2026-07-29)

The desk-loading route initially registered page-protection watches for every large texture and compute
source as soon as it entered a persistent cache. By the first MainLevel frames this had cumulatively
registered millions of pages and reduced presentation to about 1.8 FPS. A hard 1 MiB watch ceiling proved
the diagnosis (roughly 4.6 FPS) but made every larger stable source pay an exact comparison forever.

Large sources now start with exact validation and promote only after three unchanged comparisons, under an
8 MiB promotion budget per graphics/compute call; Linux also coalesces adjacent `mprotect` ranges. On the
same route, registered pages fell from about 4.4 million to 263 thousand at the matched checkpoint, menu
cadence rose to 16-17 FPS, and the first MainLevel windows reached about 4.8-5.2 FPS. The policy is generic,
keeps exact comparison as the unsupported/failed-watch fallback, and exposes the promotion thresholds for
A/B through the frontend variables documented above.

Decode-reason accounting (`PROSPER_DETILE_STATS=1`) then identified four immutable 48x48x48 RGBA16F volume
lookup textures and one 128x32 two-channel UNORM16 input being decoded essentially once per callback. The
volume detiler already handled their exact source layout; only cross-submit retention excluded them.
Extending the exact-byte cache to supported volume spans and UNORM16 removed those five repeated misses.
Matched menu windows reduced texture construction from roughly 14-16 ms to 11-12 ms per callback and
progressed around 2-3 presented frames per second faster. A six-submit full-resolution GPU replay remained
visually correct. A separate Float16 cube-retention experiment was rejected: the late loading route churned
the 2 GiB cache, grew from 218 to 411 cumulative write watches, and fell from 3.8 to 1.7 FPS in a matched
2,640-frame A/B. Non-block-compressed cubes therefore remain on their dedicated uncached layer-aware
decode path. The supported block-compressed exception below is narrower: it retains an expensive decoded
result against the exact complete six-face source span and does not admit the rejected Float16 class.

Deferring every texture watch at or above 1 MiB was also rejected. It finished with essentially the same
watch footprint and 3.8 FPS late checkpoint as the mixed policy, while cumulative exact validation rose
from 69.7 to 79.2 GiB. Stable 1-8 MiB sources therefore continue to arm immediately; larger sources retain
the exact-first promotion rule.

Once a source has a working write watch, retaining its complete encoded bytes duplicates information:
Unchanged proves reuse, while Dirty/Unknown can conservatively invalidate. Releasing only those protected
snapshots (never audit baselines or sources that are also the decoded pixels) saved 267 MiB at the end of a
matched 2,650-frame route. The cache held 485 decoded entries instead of 253 under the same 2 GiB ceiling.
Average frontend resource construction fell from 24.1 to 21.5 ms per submit and total frontend time from
50.3 to 46.6 ms per submit. Exact validation traffic changed only slightly (76.8 to 75.7 GiB) after GPU/DMA
writes were correctly wired into persistent-watch invalidation, and late checkpoint FPS was mixed rather
than a repeatable speedup. This is therefore primarily a memory/capacity improvement, not a validation or
frame-rate claim. Dirty or unknown watches deliberately re-decode rather than using a hash, and
`PROSPER_KEEP_TEXTURE_SOURCE_SNAPSHOTS=1` restores the control behavior for audit/performance comparisons.

## Compute write-watch promotion census (2026-09-02, #3155 / #3156)

The policy the section above describes has **two** levers, and the compute path only ever had one of
them. `should_promote_write_watch(source_bytes, stability, defer_min_bytes, promote_after)` promotes
when the source is *smaller* than `defer_min_bytes` **or** the stability ladder has been climbed. The
renderer's texture cache passes a real `defer_min_bytes` (8 MiB, `PROSPER_TEXTURE_WRITE_WATCH_DEFER_MIN_KB`),
which is what implements "stable 1-8 MiB sources continue to arm immediately" above. The compute call site
has always passed a literal **1**, making `source_bytes < defer_min_bytes` false for every real source. So
on that path the size exemption is unreachable and every source, at any size, must accumulate consecutive
successful full compares.

That is exactly the arm the section above **measured and rejected for textures**: "Deferring every texture
watch at or above 1 MiB was also rejected … cumulative exact validation rose from 69.7 to 79.2 GiB." It has
never been measured for compute, where sources include storage *results* that change every frame and arming
an unproven watch is wasted `mprotect`/fault/rearm work. **Both readings are live; do not change the literal
without production numbers.**

An in-tree census now produces those numbers. `PROSPER_WATCH_PROMOTE_CENSUS=1` reports, every 256 submits
and at exit, how many validated acquisitions each of the three proofs decided (submit journal / armed page
watch / full byte compare), the bytes each cost, and the stability counter seen at every promotion decision.
`PROSPER_COMPUTE_WATCH_DEFER_MIN_KB` exposes the defer minimum so the A/B is an environment sweep rather
than a patch; **its default is the historical 1 byte and changes nothing.**

Harness measurement (`test_game_compute`, 46 validated acquisitions — a tiny workload, quoted for its
*shape*, not its magnitude), one run per arm:

| arm | decisions | stability 0 | threshold met | granted | watch-decided | exact compares |
| --- | --- | --- | --- | --- | --- | --- |
| `PROMOTE_HITS=1` | 45 | 38 (84.4%) | 45 | 45 | 3 | 42 |
| `PROMOTE_HITS=2` | 50 | 39 (78.0%) | 11 | 11 | 1 | 44 |
| `PROMOTE_HITS=3` (default) | 51 | 39 (76.5%) | 4 | 4 | **0** | 45 |
| `PROMOTE_HITS=4` | 51 | 39 (76.5%) | 0 | 0 | 0 | 45 |
| `DEFER_MIN_KB=8192` | 45 | 38 (84.4%) | 45 | 45 | 3 | 42 |

Three things follow, and the first two match the production census posted on #3155 (94.8% of decisions
taken at stability 0) closely enough to be the same phenomenon at a different scale:

* **The stability counter is at zero for the large majority of decisions**, so `hits` values of 2 and above
  are gated on a ladder the dominant population never climbs — which is why they measured
  indistinguishable. `hits=1` differs in *kind*, not degree: it removes the counter from the decision.
* **At the default, write-watch promotion decides zero acquisitions here.** Every one of the 46 is settled
  by the submit journal (1) or a full compare (45).
* **`DEFER_MIN_KB=8192` reproduces `hits=1`'s census exactly** while leaving the ladder's precondition
  intact for genuinely large sources. So the effect attributed to `hits=1` is reachable through the size
  exemption instead of by discarding the proof requirement. (On this harness every source is under 8 MiB,
  so `=8192` and `=0` are the same arm; that will not hold on a title.)

Read the counts, not the bytes: every arm reports `5.0 MiB compared`, because the three compares
promotion removes here are small ones and the difference is below the report's one-decimal MiB
resolution. A **byte** claim needs a title, which is the production sweep.

The suite passes all 449 assertions at `hits=1`, at `DEFER_MIN_KB=0` and at `=8192`, which is evidence —
not proof, the harness is not a title — that arming at stability 0 does not break the exactness contracts
it checks. `hits >= 4` fails one arm ("validated typed-storage result can be leased by an exact sampled
descriptor"): that region dispatches four times, so the counter reaches 3 and no more, and the lease needs
a watch that was actually armed. That failure is the suite depending on promotion firing, which is itself
evidence the harness now hosts the mechanism.

### Ruled out

| Hypothesis | Verdict and evidence | Source |
|---|---|---|
| `PROSPER_COMPUTE_WRITE_WATCH_PROMOTE_HITS=1` is a user-facing trap: a documented switch value that deterministically crashes the emulator | **Falsified.** It crashes exactly one binary, `test_game_compute`, which asserted `guest_write_watch_set_fault_onstack(true)` in seven places while installing no SIGSEGV handler, and compensated only at the write sites it knew about. At `hits=1` the cache arms a watch the test has no hook for and the test's own store into that page is unserviced. The faulting instruction is `notl 0x0(%r13)` — `large_result[0] ^= 0xffffffffu;` — at a page-aligned mmap address. Production installs the handler. With the handler installed in the test, `hits=1` passes 449/449. | #3156, instrument trap 250 |
| The promotion **threshold** (`PROMOTE_HITS`) is the knob that decides whether compute promotion fires, so retuning it 3→2 is the fix | **Falsified twice, by two different measurements.** #3155's own re-measurement found 2 and 3 indistinguishable (63-65 GB either way) and closed PR #3158 for changing a default with no measured benefit; the census above shows why — 76-84% of decisions are taken at stability 0, where every threshold ≥ 2 refuses identically. The threshold is not the reachable lever; `defer_min_bytes` is, and it is hard-coded to 1. | #3155, #3158 |
| The 8 MiB per-submit promotion **budget** is what refuses these promotions | **Falsified.** `budget_refused = 0` in all five harness arms, and #3155's production census measured 7 refusals against 58 stability-eligible decisions (12%) with a 1 GB budget changing nothing measurable. | #3155 |
| Compute buffer/image residency "uses the same exact-first rule" as the texture cache | **Falsified — it was a documentation error in `FRONTEND_APP.md`.** The two paths share `should_promote_write_watch` but not its arguments: the texture path exempts sources under 8 MiB, the compute path exempts nothing. Corrected in the same PR. | #3155 |

## Syberia block-compressed cube retention (2026-08-03)

Syberia's save/profile screen repeatedly decoded one 2048x2048x6 BC6H cube. A five-second sampling
profile attributed 33.70% of CPU samples to `decode_bc6h_block`; adjacent captures kept the same cube
descriptor and content hash, and a bounded live miss trace observed address ordinals 1–8, 16 and 32
with one unchanged key and `reason=unsupported-candidate`. The 33,570,816-byte footprint covers the
selected mip in six independently-strided face chains. Each miss decoded 1,572,864 BC6H blocks, and
presentation fell from about 3.5 fps to 2.0 fps when this cube appeared.

The decoded-texture cache now admits a cube only when its format has a supported BC block decoder.
Its validation range is the conservative contiguous descriptor footprint, including face-chain gaps;
zero, host-size-overflowing and address-overflowing ranges stay ineligible. Reuse continues through the
existing exact comparison, ordered GPU-write journal, deferred write-watch promotion and memory-aware
LRU. This is intentionally not the rejected Plucky Float16 experiment: non-BC cubes remain excluded.
CPU regression coverage asserts the exact Syberia candidate/range shape and the nearby exclusions.

One post-fix full-resolution diagnostic reached the same profile-screen phase. The 2048 cube emitted
one `cold-or-evicted` miss with `candidate=1`, `eligible=1` and the exact 33,570,816-byte source, then
no later miss through t=225.1 s / frame 611. That arm removed the old repeated-miss signature but did
not independently prove a same-identity hit: aggregate cache hits are not identity evidence, the
detail logger suppresses resources below 0.5 ms, and an approved debugger breakpoint raced process
exit. `PROSPER_DETILE_STATS` therefore gained a bounded first-hit witness carrying the key, source span
and persistent ID/version.

A subsequent short run proved the missing identity link. The run-local 2048x2048x6 BC6H resource at
key `0x9dc5b36201047cea` first reported one exact 33,570,816-byte `cold-or-evicted` miss, then reported
`cache=persistent-hit` with the same key, footprint and source span, persistent ID 195/version 1, and
`validation=exact validated=33570816`. Both the submit journal and write-watch query returned clean
result 2. The process was stopped immediately after this witness. The expensive decoded cube is thus
being reused from the persistent cache; the earlier absence of repeated misses was not merely the
texture disappearing from the workload.

The profile output retained the expected full-width, overexposed scene without new cube corruption.
Through the post-cube t=125.057→225.102 s window, presents advanced 419→614: **1.949 fps**, no
measurable improvement over the earlier ~2.0 fps observation. This is not a clean negative A/B because
the post arm additionally enabled verbose render timing. It does show that removing the repeated miss
does not by itself make Syberia fast under that diagnostic load; the remaining renderer/compute costs
still dominate. The screenshot command's exit 1 was self-inflicted apparatus failure — a 48×5 s
schedule cannot complete under `--timeout 230` — after 45 source- and pixel-distinct samples, not a
game stall.

## Plucky Squire unused descriptor materialization (2026-07-29)

A later MainLevel probe found repeated 16-24 ms buffer-resource spikes even though the backend never
consumed the affected binding. The guest fold had conservatively retained a broad original V# with an
absurd 0xffffffff-byte declaration, while the final recompiler emitted separate pc-specific scalar
bindings for every actual access. SPIR-V reflection consequently reported binding 32 as an unused runtime
extra and bindings 33-52 as the shader's statically used interface. The frontend nevertheless built every
runtime table entry, probing and zero-allocating the full 64 MiB corrupt-descriptor safety ceiling once per
draw. Other invalid candidates at addresses zero and four took the same fallback.

The live renderer now materializes only bindings present in the final reflected SPIR-V interface. This is
not a size heuristic: used dynamic buffers still retain their complete declared range, including the
multi-megabyte Blue Prince streams from #1427. A definitely unmapped used buffer keeps the established
all-zero semantics in a small reflection-sized robust buffer instead of allocating its corrupt declaration.
The same scripted diagnostic still recovered the pathological runtime descriptors, but emitted zero slow
buffer materializations after the change. A full-resolution Plucky replay retained output hash
`1c5241d9475e752d`, and its before/after BMP files were byte-identical.

## Plucky Squire texture-validation attribution (2026-07-29)

After unused buffer materialization was removed, `PROSPER_RENDER_TIMING=detail` still attributed the
late route's largest frontend spikes to full-resolution sampled textures. Detail output now separates
exact-validation time and copied/expected byte counts from total resource time, and reports both the
submit-local GPU-write query and cross-submit page-watch state. This distinguishes an expensive proof
of unchanged bytes from an actual decode without changing the cache policy.

The resulting live trace identified two different costs. One 3840x2160 packed-float source was last
written by compute program `0x3015770000` around submit 1116 and remained unchanged while consumers
continued beyond submit 2989. Its page watch had disabled during the earlier dynamic phase, so each
later callback compared the complete 33,423,360-byte source in roughly 3.5-5.8 ms. In contrast,
same-frame outputs from programs `0x30180d0000` and `0x3015770000` were still being written at their
consumer submits. Their conservative invalidations caused 64-90 ms BC/packed-float decode paths;
some packed-float validations copied zero of the expected 33,423,360 bytes because the guest range
was no longer completely mapped.

A disabled-watch recovery experiment remains rejected after byte-preserving buffer notifications were
fixed. A conservative 64-match rerun recovered six smaller sources, but the two 66.8 MiB gameplay sources
reached only 13 and six matches before their scene lifetime ended. A deliberate 16-match stress run then
forced recovery: several 3 MiB sources were dirtied, disabled, and recovered a second time within two
reporting windows, while both 66.8 MiB sources also entered the expensive recovery path. This proves that
an exact-equality streak still cannot establish that every producer has stopped; the corrected buffer path
removed one false invalidation source, not real later writes or other producer classes. A longer delay only
reduces the frequency of the cycle. The next optimization must therefore carry exact cross-submit GPU
write/result evidence or bind a retained compute result directly; equality of guest bytes alone is not a
sound trigger for restoring page protection.

## Cobra Float32 sampled-texture retention (2026-07-25)

Cobra's title and cinematic repeatedly sample large guest-backed Float32 post-process inputs. The live
renderer narrowed each one to RGBA16F on every callback even when its exact source bytes were unchanged;
these were the largest frontend CPU cost and also forced a fresh backend image upload. Float32 2D inputs
now use the existing exact-content cache. The key retains the complete descriptor shape, validation counts
the 4-byte source components, and a cache hit restores the native RGBA16F backend format. Existing guest
CPU/GPU write tracking and exact-byte fallback still invalidate changed versions.

One Linux/Radeon 8060S binary, the same scripted route, native 1920x1080 output, audio enabled, and rolling
25-submit windows produced:

| Measurement | Float32 uncached | Float32 retained |
|---|---:|---:|
| Title frames reached at 45 seconds | 744 (16.5 FPS) | 1,178 (26.2 FPS) |
| Matched cinematic draws / submit | 28.7 | 30.7 |
| Cinematic whole submit | 142.8 ms | 46.5 ms |
| Cinematic resource construction | 63.5 ms | 4.5 ms |
| Matched dense draws / submit | 374.8 | 380.8 |
| Dense whole submit | 418.2 ms | 279.1 ms |
| Dense resource construction | 133.1 ms | 82.8 ms |

The dense route retained 141 decoded versions / 910.8 MiB, below the existing 1 GiB bound. Backend draw
setup and synchronous color-target readbacks remain the largest gameplay costs; this tranche does not
claim a playable frame budget.

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
- **A dispatch ring cannot overlap anything WITHIN a batch, because the end-of-batch drain is
  mandatory** (2026-09-01, [#3157](https://github.com/mattias800/prosper/issues/3157),
  implementation [#3161](https://github.com/mattias800/prosper/pull/3161), closed unmerged). The
  hypothesis was well motivated and the sizing was real: an `LD_PRELOAD` interposer put
  `vkWaitForFences` at **19.6-37.9% of wall** across three UE titles (Stray `PPSA02101` 37.9%,
  Little Nightmares III `PPSA05143` 33.0%, Dragon Quest VII `PPSA17942` 19.6%), with `vkQueueSubmit`
  at ~1% everywhere -- so batching *submits* is dead, and ~160 us per dispatch is scheduling latency
  with the GPU idle. An alias-guarded ring was built and is correct: 315/315 at depths 1, 3 and 8,
  zero faults across five interleaved 55 s Stray runs.
  It does not deliver the sized win, and the reason is structural rather than a tuning failure. On
  this route the guest submits roughly **one dispatch per batch** -- **23,294 dispatches across
  ~23,400 batches** on Stray -- and the drain at the end of each batch is **mandatory**, because the
  guest may read its memory as soon as the submit returns. Every slot is therefore drained
  immediately after it is filled, so the ring never overlaps anything. Measured throughput: medians
  67.5 -> 68.5 fps, within noise; the only clear effect is **reduced variance**, and what little
  gain exists comes from per-slot command pools, not from pipelining.
  **What this does NOT rule out:** deferring **across** batches. That is where the sized win still
  lives, and it is untried -- it needs a guest-visibility contract the ring does not have. Do not
  read this row as "pipelining the compute path is dead"; read it as "a within-batch ring is dead,
  and the dispatches-per-batch ratio is the number to measure before building the next one."
  That ratio comes from `PROSPER_RENDER_TIMING`, which counts dispatches and batches; the alias
  census (`PROSPER_COMPUTE_ALIAS_CENSUS`) answers a different question -- whether consecutive
  dispatches alias through guest memory -- and has no call or batch counter at all.


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

Throughput, measured on an **idle** machine through `tools/perf/ab_compute.sh`, **4 reps with the arm
order alternating by rep parity**, 150 s each on `load-save-first-station.pad` (a real gameplay scene,
not a menu):

| | compute, gameplay tail | gameplay frame rate |
|---|---|---|
| host path (`PROSPER_NO_DIRECT_RTT_BIND=1`) | 7.75 ms | 13.3 fps |
| direct bind | 5.97 ms | 16.6 fps |
| | **-23.0%** | **+24.3%** |

Split by order, to show the warm-up confound is not producing the result: OFF-first gives -23.7% /
+22.3%, ON-first gives -22.3% / +26.6%. The compute memory pool drops **80.1 -> 63.7 MiB** (9 -> 7
cached allocations) as the per-dispatch staging buffer and the duplicate image disappear.

**Always alternate the arm order.** An earlier 3-rep sweep ran OFF before ON every time, handing the
second arm every warm-up benefit there is -- on-disk pipeline caches, GPU clock ramp, page cache -- and
that arm was the one being advocated. Repetition does not detect a systematic bias; only alternating or
randomising the order does. Here the bias turned out to be small, but it was not knowable in advance.

**The compute delta reproduces; its frame-rate translation does not.** Across two independent sessions
the compute saving was stable (-25.0% then -23.0%), while the measured frame-rate benefit ranged from
**+7.7% to +24.3%**, with the host-path arm steady near 13 fps and the direct-bind arm varying between
14.0 and 17.5 fps. The fast path is conditional by design -- it is taken only while the persistent
image is the authoritative copy -- so how often it applies can legitimately vary between runs, and
absolute frame rates from different sessions are not comparable. Quote within-session deltas, and
treat any single frame-rate figure here as indicative rather than exact.

### Distance to a playable frame rate

| | current | 30 fps | 60 fps |
|---|---|---|---|
| frame time | **71.2 ms (14.0 fps)** | 33.3 ms | 16.7 ms |
| required | -- | **2.14x faster** (remove 37.9 ms) | **4.27x faster** (remove 54.5 ms) |

Composition of the current frame, from the earlier session's 71.2 ms measurement: **compute 31.4 ms
(44%)** (5.28 calls x 5.94 ms) and **everything else 39.8 ms (56%)**. A later session measured the same
build at 60 ms (16.6 fps); see the variance note above, and re-derive the split rather than reusing
these absolutes.

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

## Frame decomposition of Blasphemous 2 gameplay (#1101)

Following #1095, #1101 decomposed a ~68 ms Blasphemous 2 gameplay frame with
`PROSPER_COMPUTE_PHASE_TIMING` + `PROSPER_RENDER_TIMING` on the save-backed route. **Actual GPU
execution is ~5% of the frame.** The rest is host-side pixel round-trips and per-dispatch state:

| cost | ms/frame | what it is |
|---|---|---|
| compute image setup (create+upload) | 15.6 | per dispatch; the VkImage-reuse contract, #1106 |
| graphics readback | 15.6 | GPU->CPU color-target copies |
| compute writeback (pack+layout) | 13.0 | CPU tiling conversion of storage results |
| GPU execution + waits | 3.1 | ~5% |

The methodology lesson: the plausible bottleneck was guessed wrong twice -- submit-and-wait at 5.3%,
SPIR-V validation at 0% (`setup_validate_ms` proved the 2.76 ms setup is entirely image create+upload).
**Measure the decomposition before choosing the optimization.**

This section documents the two proven-correct outcomes; the largest opportunity (graphics readback) and
the image-setup cost are tracked as follow-ups.

### Range-specialized storage-image pack (landed)

Compute writeback pack was a non-inlined `storage_pack_texel` call per texel re-entering a format switch
per component -- the exact shape #1092's `storage_unpack_range` removed on the unpack side.
`storage_pack_range` hoists the format dispatch out of the loop; packed formats keep the general
per-texel path. Bit-identical by construction; `PROSPER_VERIFY_PACK=1` proved it on 1,327 real-workload
dispatches with 0 mismatches. `PROSPER_NO_PACK_RANGE=1` for A/B. It has **no measurable frame impact**
(pack is ~1.3 ms of a 53 ms frame), so it is a correctness-neutral hot-loop cleanup rather than a win;
the commit's lasting value is the `setup_validate_ms`/`setup_buffers_ms` sub-timers that made the
decomposition and #1106's scoping possible.

### Deferred as follow-ups

- **Graphics readback (15.6 ms/frame, #1101):** deferring offscreen color-target readbacks measured
  +19% fps at scale 1 but broke the scale-4 `messenger-scene` snapshot (black). The cause was not
  conclusively isolated between render-timing fragility of the blind capture route and a subtle
  correctness gap, so it is held on branch `perf/issue-1101-frame-decomp` pending a content-based
  capture window or deeper root-cause. See #1101.
- **Image setup (15.6 ms/frame, #1106):** reusing the VkImage across dispatches. Needs a correctness
  contract for storage-image writeback and invisible guest texture writes; deferred to a review-gated
  change.

## Plucky Squire: repeated raw 3D storage results

Plucky's program `0x3013930000` dispatches a 48x48x48 lighting-grid kernel every frame and writes four
tiled RGBA16F volumes. The Vulkan dispatch itself is small; the raw storage fallback previously copied
four RGBA32_UINT interchange images to the CPU, packed them to FP16, and retiled them even when the
result was byte-for-byte unchanged. In the measured title/game route this one dispatch had a **3.74 ms
median total**, of which **3.21 ms** was writeback (22,264 samples from the baseline timing run).

The retained-result contract now also covers raw storage images when the existing seed-skip proof says
the shader is both write-only and full-coverage. That restriction is essential: raw RGBA32_UINT values
are not canonical guest-format values and cannot generally be reused as a later image-load seed. For a
full writer the previous image is unobservable, so the backend can safely retain it and compare the
next transfer to an exact GPU-side baseline. The CPU receives only a four-byte changed flag. Guest
write tracking remains authoritative: an external change disables the skip and forces ordinary exact
pack, retile, notification, and baseline repair.

On the same machine and route, the updated live run reduced the program to a **0.68 ms median total**
and **0.00 ms writeback** (81 samples). This is a targeted before/after rather than an alternating
whole-route benchmark, so it supports the dispatch-local result but not a standalone frame-rate claim.
The realized capture `/tmp/plucky-393-3281244e.prgcap` records the exact four-volume shape and shader,
but replay materializes resources as owned `host_data`; by design that does not enter the live
cross-frame guest-memory cache. Correctness is therefore pinned by the raw 3D production-backend test:
it proves full coverage, establishes an exact baseline, skips an identical third write, then mutates
the guest mirror and requires one exact repair write and invalidation.

## Plucky Squire: repeated large writable buffers

A clean live profile (phase timing only; no `PROSPER_COMPUTELOG` hashing) found two consecutive
lighting kernels spending about **1.53 ms/frame** in buffer writeback: program `0x30179b0000` had a
0.98 ms median and `0x30179d0000` had a 0.55 ms median. Guest relocation names those same programs
`0x3017970000` and `0x3017990000` in the corresponding capture and live A/B. The capture shows the
first kernel's writable result is a 33,423,360-byte buffer that is byte-identical on repeated
dispatches. The old path still mapped and compared every byte on the CPU before learning that no guest
copy was needed.

Large persistent writable buffers now retain a second exact storage buffer as their previous result.
When guest GPU-write provenance or the host write watch proves the guest mirror unchanged, the shared
GPU comparator checks current versus baseline and returns one changed flag. An equal flag omits the
large host mapping/comparison but preserves the architectural GPU-write notification. If guest memory
changed externally, source validation disables the shortcut; the ordinary CPU comparison repairs the
guest exactly, while a transfer advances the GPU baseline to the new result. This is the same
collision-free contract used by retained storage images, not a content hash.

An 80-repeat isolated replay of capture compute 22 measured the following medians after discarding the
first five iterations:

| | exact GPU result comparison | CPU result comparison |
|---|---:|---:|
| writeback | 0.02 ms | 0.77 ms |
| dispatch (including comparator) | 0.65 ms | 0.27 ms |
| total | **1.66 ms** | **1.96 ms** |

The replay owns resource bytes as `host_data`, so its repeated setup scan is not representative of the
live guest-write-provenance fast path; the comparison isolates the result-side tradeoff. The adjacent
8,355,840-byte result was neutral (2.02 versus 2.04 ms total), so GPU result baselines default to a
measured **16 MiB crossover** instead of doubling every persistent buffer. Set
`PROSPER_COMPUTE_BUFFER_RESULT_MIN_MB` to retune it, or
`PROSPER_NO_PERSISTENT_COMPUTE_BUFFER_RESULTS=1` for an exact A/B. The production-backend test runs an
idempotent one-MiB buffer with the test threshold lowered, requires the normal invalidation on the
GPU-identical repeat, mutates guest memory, and requires byte-exact repair on the next dispatch.

A subsequent live matched-switch check confirmed the dispatch-local result over 780+ samples per arm:
the 33 MiB kernel changed from **1.27 to 0.85 ms total** (writeback 0.99 -> 0.09 ms, dispatch including
the comparator 0.13 -> 0.59 ms). The adjacent 8 MiB kernel stayed **0.82 versus 0.81 ms**, confirming
the crossover gate. Whole-route FPS varied by more than this sub-millisecond saving between the two
sequential runs, so these numbers support the local optimization only, not a frame-rate claim.

## Plucky Squire: small repeated storage mip results

Plucky runs the same mip-generation compute program twice per frame. The first invocation writes a
large chain and already qualifies for persistent image residency. The second writes 32 KiB, 8 KiB,
2 KiB, and 512-byte native storage results. The historical one-MiB image threshold left all four on
the transient path, so a roughly 0.2 ms dispatch still paid about 1.5 ms to map, pack, and retile its
small outputs every frame.

Sampled inputs and writable outputs have different retention economics. Tiny sampled images are
numerous and cheap to upload, while a repeated storage image pays a writeback/layout boundary even
when its result is unchanged. The cache policy therefore keeps the sampled default at 1 MiB and gives
storage images an independent one-page (4 KiB) crossover. Existing
`PROSPER_COMPUTE_IMAGE_CACHE_MIN_KB` overrides remain compatible for both roles;
`PROSPER_COMPUTE_STORAGE_IMAGE_CACHE_MIN_KB` retunes only storage residency.

Matched 35-second live sweeps measured the second invocation after submit 100:

| storage crossover | samples | median writeback | median layout | median total |
|---|---:|---:|---:|---:|
| 1 MiB (historical control) | 477 | 1.51 ms | 1.16 ms | 2.11 ms |
| 32 KiB | 498 | 0.70 ms | 0.35 ms | 1.45 ms |
| 4 KiB | 514 | 0.16 ms | 0.08 ms | **0.83 ms** |
| 1 KiB | 481 | 0.16 ms | 0.02 ms | 0.89 ms |

The 4 KiB and 1 KiB totals are within run noise; one page is the narrower general policy because it
does not retain sub-page Vulkan objects. Frame progression stayed healthy across the matched control
and broad candidate runs (527 and 529 presented frames). These were sequential dispatch-local sweeps,
not an alternating whole-route benchmark, so they support the retained-output crossover rather than a
standalone FPS claim. GPU replay cannot exercise this live residency path because replay resources are
owned `host_data`; the production-backend test instead pins the temporal contract, and the pure policy
checks pin both exact default boundaries.

## Plucky native-subgroup replay fidelity

Plucky's two remaining heavy compute programs exposed a profiling artifact: live rendering selected
the exact wave32/wave64 subgroup path, while arming capture forced the translator back to its portable
lane-emulation shell. The resulting capsule was internally replayable but did not contain the shader
whose live performance was under investigation. Capture v37 now retains the required subgroup size
beside the stored SPIR-V and replay recreates the required-size/full-subgroups pipeline contract.
Pre-v37 captures remain readable with `subgroup=0`; unsupported replay devices reject a native contract
instead of silently executing it with a different subgroup width.

Radeon GPU Analyzer also ruled out generic SPIR-V dead-code cleanup as the next optimization. For the
`0x3015770000` program, original and DCE variants compiled to byte-identical gfx1151 ISA. For
`0x30180d0000`, DCE only reordered two independent scalar moves while leaving the 85-VGPR and 7,176-byte
ISA totals unchanged. Rewrites that reduced reported VGPR use to 62 regressed measured replay dispatch
time, so they are retained as negative evidence rather than production changes. A fresh v37 capsule
then retained the live `0x30180d0000` and `0x3015770000` modules with `subgroup=64`; compute-only replay
created and ran both exact pipeline contracts. The standalone startup capsule still has six unrealized
operations and no temporal RTT seeds, so its final pixel oracle and live compute result do not match.
It is valid evidence for module/pipeline analysis, not for a full-frame correctness claim.

A matched live switch test first separated the two remaining kernels. Native lowering reduced
`0x30180d0000` from a 5.65 ms to 3.22 ms median GPU dispatch, but the four-wave, LDS/barrier-heavy
`0x3015770000` kernel moved from 5.21 ms to 5.62 ms. The reason was structural rather than an inherent
multi-wave cost: its exact-subgroup module still contained the portable structured-control-flow wave
votes, including 20 synthetic workgroup barriers in addition to the program's three real barriers.
Every other eligible Plucky dispatch in that route used exactly one 64-lane guest wave; those were
neutral or faster, including a repeated mip kernel at 0.44 ms native versus 0.86 ms portable. The
interim policy therefore defaulted to exact single-wave workgroups and retained
`PROSPER_NATIVE_COMPUTE_MULTIWAVE=1` as an experiment.

The follow-up on 2026-07-30 made the exact-subgroup contract consistent across all translator paths.
Straight-line MBCNT now uses subgroup exclusive scans, as the CFG dispatcher already did, and the
structured compute path uses subgroup-any instead of the portable workgroup vote. The captured
`0x3015770000` module fell from 32,076 to 8,026 SPIR-V words and now contains ten subgroup votes plus
only the three guest barriers. A matched 55-second full-resolution live A/B measured steady submits
after submit 100:

| program | portable median | native median | samples (portable/native) |
|---|---:|---:|---:|
| `0x30180d0000` | 6.230 ms | 3.615 ms | 726 / 800 |
| `0x3015770000` | 5.880 ms | 1.630 ms | 724 / 799 |

The default remains conservative and title-independent: multi-wave workgroups are admitted
automatically only when the decoded shader contains the canonical MBCNT low/high pair or at least four
VCC/EXEC branches accepted as top-level regions by the structured compute emitter alongside top-level
guest workgroup barriers. That structural proof guarantees portable lowering really emits the repeated
scratch-emulated votes which justify the exact subgroup shell; raw branch opcodes alone do not qualify.
Other multi-wave shapes still require `PROSPER_NATIVE_COMPUTE_MULTIWAVE=1`, and
`PROSPER_NO_NATIVE_COMPUTE_SUBGROUP=1` remains the complete portable control. A default-policy run
selected subgroup64 for both heavy programs, presented 265 frames, and produced no renderer or dispatch
failure. A fresh four-present bundle replayed all 12 submits and 56/56 operations in the heavy submit;
its 3840x2160 output was pixel-identical to the independently scheduled live screenshot (infinite PSNR).
Frame counts remain supporting route progression rather than a standalone FPS claim.

## Plucky split no-GS NGG producer coverage

The same startup submit exposed three real draws that both live rendering and capture previously dropped.
They share a split no-GS NGG vertex program: a fetch/vertex producer writes one 28-byte LDS record, while a
compiler-generated wrapper compacts the guest wave and exports that record. The existing structural
passthrough proof already recognized wrapper addresses expressed as
`v_mad_u32_u24(stride, exporter, constant)`. This compiler emitted one equivalent field address as
`v_mul_u32_u24(stride, exporter)` plus a DS-read immediate, so the otherwise-proven wrapper fell back to
unsupported cross-lane execution.

The recognizer now accepts both exact unsigned-u24 address forms and still requires a single stride/index,
direct terminal LDS loads, matching producer fields, the allocation/primitive-export shape, and complete
POS0. In the proven producer only, canonical all-ones `v_mbcnt_lo/hi` derives the guest lane from the
flattened Vulkan vertex/instance invocation. Other masks remain rejected because their peer-lane state is
not available in a vertex invocation. Synthetic coverage renders the MAD and multiply-plus-DS forms,
checks the resulting geometry, and verifies that replacing the all-ones mask with a partial mask fails
closed.

A fresh live capture of the same 43-draw / 27-dispatch submit moved from **37 to 40 realized draws** and
reported the generic proof (`base=v5`, stride 28, one parameter export). The three new operations are the
two 48x48 targets at semantic draws 5/6 and the 32x32 target at draw 35. Exact replay pipeline statistics
showed real geometry and fragment work:

| semantic draw | vertices | primitives / after clip | fragment invocations / passed samples |
|---:|---:|---:|---:|
| 5 | 192 | 96 / 96 | 110,592 / 110,592 |
| 6 | 192 | 96 / 96 | 110,592 / 110,592 |
| 35 | 128 | 64 / 64 | 32,768 / 32,768 |

The only remaining unrealized entries are two decoded no-effect draws and one draw with no fragment
program whose captured pipeline also has no color/depth/stencil write state. This closes the concrete
shader-translation gap; it does not turn the startup capsule into a full-frame oracle.

An initial seven-submit bundle confirmed why: although all 78 realized compute/graphics operations execute,
the window starts at process submit 1 and still has 39 read-before-write image versions with neither a prior
GPU producer nor a serialized RTT seed. Its selected 3840x2160 replay is black, and final-capsule export
correctly refuses to invent the missing seed. The exact module, pipeline, and per-draw statistics above are
valid; the bundle image is explicitly not correctness evidence.

## Plucky title replay closure and scalar RTT snapshots

An older six-submit title bundle exposed a separate capture-tool failure before a later live comparison could
be trusted. The live RTT cache could retain CPU bytes from an earlier extent or format after a GPU-only render,
and capture silently described unsupported scalar `R8_UNORM` and `R32_UINT` surfaces as RGBA8. Capture v38
adds their native seed formats, validates every snapshot against its current identity, reads back an
authoritative GPU-only surface when necessary, and skips only entries that have no current authoritative
contents.

Fixing the snapshot exposed three false edges in the bundle graph. It had treated every compute binding as
both read and written, matched distinct images by interior byte overlap, and counted programmed color targets
whose effective write mask was zero. The graph now uses reflected per-binding read/write facts, exact guest
bases for image-to-image dependencies, byte overlap only where buffers or DMA require it, and non-zero target
write masks.

With those generic rules, the old bundle exports a final capsule with 39 color and three depth/stencil seeds.
The bundle and standalone capsule both render 33,177,600 bytes at 3840x2160 with output hash
`92f7a0c3b31a909b`; their BMP files are byte-identical with SHA-256
`3b627f822192a0152c26d864f0ac9ffe22087f98916699d4745439e80c7dccf6`. The image is the complete Plucky title
screen. This proves deterministic bundle-to-capsule closure, not native correctness: two earlier 48x48 volume
reads remain unresolved because that old bundle skipped their then-unsupported typed-storage dispatches.

## Byte-preserving compute-buffer notifications

The retained compute-buffer path already had two independent exact proofs before suppressing a
writeback: source validation proved that guest RAM still matched the retained input, and the GPU
comparator proved that the dispatch output matched the retained result byte for byte. Despite writing
no guest bytes, that fast path still used the ordinary GPU-write notification. It unnecessarily dirtied
cross-submit guest-byte watches and entered the range in the submit-local mutation journal.
The exact CPU-comparison fallback had the same issue whenever it proved equality and skipped its host copy.

The byte-preserving notification now reaches only the renderer alias observer. Color/depth targets and
CPU RTT snapshots overlapping the range are still invalidated because they can diverge from guest RAM,
while decoded guest-byte caches, compute source watches, and the submit journal remain valid. A Linux
production-backend regression checks both paths: a sub-threshold buffer that takes the CPU comparison and
a retained one-MiB buffer that takes the exact GPU comparator. Restoring either old notifier makes its
corresponding assertion fail; a later real guest mutation still defeats the comparator shortcut and is
repaired normally.

This is a generic coherency correction, not a measured Plucky speedup. A targeted 145-second full-resolution
route recorded zero buffer-result skips in the selected window. It did record 1,981 storage-image skips,
including 1,003 exact 33,177,600-byte skips for the repeated 3840x2160 packed-float output. Storage-image
skips already avoid guest-write notification, so they are deliberately unaffected by this change. That
trace instead keeps the disabled texture-watch and incomplete guest-range cases below as the active Plucky
bottleneck.

## Plucky direct compute-image sampling prototype (issue #1477)

The remaining large sampled-texture spikes had an exact GPU authority that the renderer did not consume.
A successful typed-storage compute dispatch already retained its native Vulkan image on the graphics
device, then mirrored the same result into tiled guest memory. A later graphics span nevertheless validated,
detiled, and uploaded those guest bytes again. The observed full-resolution outputs are 33,423,360-byte
R11G11B10 and 66,846,720-byte RGBA16F surfaces; cold or invalidated fallback references have taken roughly
64-90 ms to materialize.

The issue #1477 prototype publishes only successful native typed-storage results created with sampled usage.
The graphics import reconstructs the complete compute-cache identity and borrows the image only while the
current-submit GPU-write journal or a clean cross-submit page watch proves that no later architectural writer
overlapped its guest mirror. Every attempted storage dispatch revokes the previous authority before command
recording, and only complete dispatch/writeback/layout success republishes it. A shared lease pins the cache
entry until the graphics fence cleanup; graphics transitions the borrowed image from `GENERAL` to shader-read
and restores `GENERAL` without destroying the image. Raw-uvec4 interchange images, replay `host_data`, depth,
sRGB, mip-tail/array/volume views, and DCC surfaces without an exact all-`0xff` uncompressed proof remain on
the existing path. `PROSPER_NO_DIRECT_COMPUTE_IMAGE_BIND=1` is the control switch.

Focused production-backend coverage proves cross-submit watch authorization, rejects an import after an
architectural GPU/DMA write dirties the watched guest range, and holds the lease through submission cleanup.
The renderer execution test compares borrowed native FP16 and R11G11B10 images against the existing CPU
readback/upload route byte for byte, then samples the owner again to prove its layout and lifetime were
restored. Restoring the fallback importer or omitting either native-format bridge makes those tests fail.

A 135-second full-resolution, normal-cadence Plucky route exercised the bridge on the real 4K outputs. The
explicit detail log reached its 250-line cap with direct hits; individual frontend resource decisions took
approximately 0.00-0.11 ms and copied or validated zero guest bytes. At submit 1,100, cumulative texture
resource construction was 8.22 ms per submit with the bridge and 9.20 ms in the sequential disabled control;
cumulative frontend time was 34.49 versus 35.04 ms. The enabled run produced a clean native 3840x2160 Play
Style screenshot and completed 1,508 presents, versus 1,367 in the control. The scheduled control screenshot
landed ten guest presents earlier during the menu's black fade, so it is not a pixel oracle and the route
progression difference is supporting evidence rather than a standalone FPS claim. GPU replay cannot exercise
this temporal residency boundary because replay resources intentionally own `host_data`; the production tests,
live disable switch, and the unchanged replay fallback divide that verification responsibility explicitly.

After merging current master, a fresh 140-second exact-head A/B repeated the result with identical build,
route, full-resolution, audio, input, and timing settings. The enabled run logged 250 direct decisions,
including a 3840x2160 RGBA16F result at 0.09 ms, and reached 2,220 presents; the single-switch disabled run
logged no direct decisions and reached 2,040. Their final 25-submit texture-resource windows were 6.45 ms and
10.62 ms respectively. Both native 3840x2160 checkpoints were visually clean. They armed at the same guest
present (1,058) but asynchronous readback completed on different later presents and captured different menu
states, so they remain independent correctness/progression evidence rather than a pixel-equality oracle.

### Raw compute replay closes the portable-capture profiling gap

The next measured Plucky hotspot was program `0x30180d0000`, binding 30: a 3840x2160 RGBA16F
storage result whose captured portable module expands 66,846,720 guest bytes into a 132,710,400-byte
RGBA32_UINT interchange image. Inspection proved that this is not necessarily the live steady-state path.
The renderer device advertises RGBA16F storage support and ordinary uncaptured execution already emits the
native-width float image, but capture-bound translation intentionally stores portable SPIR-V. Realized
compute captures retained neither raw RDNA2 nor the semantic launch ABI, so replay could not regenerate the
current/device-specific module and its timing overstated live cost.

Capture v39 now retains that missing source and ABI state. `gpu_replay --recompile-raw` rebuilds compute as
well as graphics, substitutes only successful modules, restores captured push constants, and selects typed
storage/subgroup capabilities from the initialized replay device when rendering. Tests cover v39 roundtrip,
owned materialization, current-SPIR-V substitution, missing-source fallback, malformed state, and pre-v39
compatibility. This makes a fresh capture an honest stored-portable versus current-live A/B for the remaining
Plucky compute work.

A fresh targeted v39 capsule of submit 7 then retained raw state for all 27 realized dispatches. The current
replay rebuilt all 27 on the Radeon device (`device-formats=0x3ff`), and both generated BMPs were byte-identical
(`sha256 9e2733eda3327c11ae662cd59708b9d5b344fa59f2b694009c9854755a59745e`). For `0x30180d0000`
binding 30, validated SPIR-V changed from a `%uint` portable storage image to a `%float` native storage image;
staging fell from 132,710,400 to 66,355,200 bytes and setup from 68.69 ms to 53.77 ms. The same A/B exposed a
larger aggregate win in the `0x30194e0000` mip dispatch: 49.10 ms to 32.74 ms. The single-submit replay still
selects a 512x512 fallback instead of the capture's 3840x2160 live oracle because this early endpoint has no RTT
seed window, so the equal BMPs are deterministic stored/current regression evidence, not a claim that this
capsule reproduces the complete live frame. A rolling producer window remains the next correctness oracle.

## Plucky storage-result validation snapshots

Profiling the full-resolution gameplay route found another synchronous CPU copy outside the Vulkan
dispatch itself. After every successful persistent storage-image writeback, compute retained an exact
copy of the complete guest target as its next source-validation baseline. This is necessary for a
read/modify/write storage image, but it duplicated 30-70 MiB every frame even for a proven-full
write-only target whose seed is unobservable.

Directly replacing every result snapshot with a Linux page-protection watch was not a valid fix. It
regressed changing targets through repeated protection churn: an early version raised
`0x3017be0000` writeback from about 3 ms to 50-61 ms. A later same-binary A/B that promoted only stable
results still raised `0x30180d0000` setup from 1.02 to 5.27 ms median because querying a large watch
walked its host-page records. It also failed to reduce total snapshot traffic materially.

The final policy uses no new page watches. For a proven-full write-only target, the existing exact GPU
result comparison runs even when no source baseline is available. A changing result is written back
normally and releases the unobservable source snapshot. The first repeated result repairs the guest
mirror and retains one exact byte baseline; later GPU-identical skips neither rewrite the guest nor
recopy that baseline. Read/modify/write, partial, and replay-owned targets keep the old exact-snapshot
contract. `PROSPER_NO_ADAPTIVE_STORAGE_RESULT_VALIDATION=1` restores the complete prior policy.

The production-backend regression covers all three boundaries: identical skipped output does not
recopy its baseline, an external guest mutation still forces exact repair, and alternating proven-full
writers carry no result snapshots. A second invocation under the recovery switch proves the disabled
path preserves old snapshot behavior without adding copies to identical skips. The existing injected
post-submit readback failure continues to prove that a GPU baseline cannot publish stale guest bytes.

A matched 180-second native-resolution Plucky A/B used the same binary, route, audio/input frontends,
and timing settings; the control changed only the recovery switch. Storage-result snapshot traffic fell
from 33,478.7 MiB to 30,125.3 MiB (10.0%). Representative complete-dispatch timings were:

| program | control median | adaptive median | samples (control/adaptive) |
|---|---:|---:|---:|
| `0x3017be0000` | 4.75 ms | 3.73 ms | 124 / 124 |
| `0x30180d0000` | 4.05 ms | 4.24 ms | 2,058 / 2,032 |
| `0x3017bb0000` | 10.54 ms | 9.16 ms | 121 / 121 |

The dominant stable title dispatch therefore remains neutral instead of paying page-watch query cost,
while the changing 1920x1080 writer improves by 21.5% at the median. Its 124 adaptive samples remained
bounded at 5.92 ms p95 and 15.47 ms maximum, with no recurrence of the experimental 50-61 ms state.
Different submit counts reflect real route progression, so these are supporting live measurements rather
than a standalone FPS or pixel-equality claim.

## Plucky cold image-cache snapshot copies

Per-image timing on the full-resolution gameplay route localized the remaining
`0x30180d0000` writeback cost more precisely. Its 3840x2160 RGBA16F output spent only about
8-10 ms in the required tiled guest layout, while cache admission spent a 41.61 ms median copying
66,846,720 bytes. The two large sampled inputs had a second copy after the fence as well: setup had
already captured each exact guest source before transfer, but `retain_image` copied that owned vector
again instead of adopting it.

Two generic lifetime fixes remove those copies. A cold proven-full write-only guest target at least
16 MiB defers its source snapshot until a later use actually needs exact source authority; the old seed
is unobservable, and a later external guest write still forces ordinary exact repair before an identical
GPU result may skip writeback. An aligned exact storage result also goes directly into its retained GPU
comparison buffer instead of first filling a host vector that the successful ownership transfer
immediately clears. If that ownership transfer fails, any older host result fallback is invalidated so
it cannot suppress a later changed writeback. Separately, sampled-image admission now moves its
already-snapshotted source into the cache. Small, partial/readable, replay-owned, failed-recovery, and
non-GPU-comparable paths retain their prior snapshots. `PROSPER_COLD_STORAGE_SNAPSHOT_MIN_MB` tunes the
cold crossover, while
`PROSPER_NO_ADAPTIVE_STORAGE_RESULT_VALIDATION=1` remains the complete storage-source control.

Matched 160-second native-resolution runs used the same build, route, audio/input frontends, and timing
settings. Filtering to the gameplay form of `0x30180d0000` (`setup_ms > 50`) avoids mixing its much
smaller title-screen use:

| state | setup median | writeback median | total median | samples |
|---|---:|---:|---:|---:|
| merged baseline | 88.01 ms | 113.17 ms | 206.61 ms | 103 |
| cold-result copies removed | 95.19 ms | 68.98 ms | 168.42 ms | 103 |
| sampled snapshots moved too | 91.53 ms | 12.79 ms | 110.25 ms | 115 |

These measurements predate the adaptive image-cache sizing merged in #1519. That change raises the
historical 512 MiB default to one eighth of the largest device-local heap, capped at 2 GiB, and should
reduce how often the same Plucky identities become cold. The table therefore measures the cost of an
observed cold dispatch, not the frequency or whole-route impact on the current combined renderer; those
must be measured again on the rebased exact head.

The final dispatch-local total is 46.6% below the merged baseline; writeback is 88.7% lower. Presented
frame totals varied with route progression (2,430 / 2,209 / 2,544), so they are recorded only as a
progression sanity check. Both normal and disabled adaptive production-backend variants pass, including
the failure-repair, external-mutation, direct-import, partial-write, and raw-volume contracts. Timing now
also attributes map, result preparation, guest-watch, notification, cache, and host result-fallback
snapshot costs, so later cache churn cannot hide inside the aggregate writeback phase again.

The production-backend suite also runs a reduced cold target with the real crossover set to zero. It
proves first admission omits the source copy, the first invalidated repeat repairs and establishes exact
authority, an external mutation forces writeback, and the disable switch preserves immediate snapshots.
An alternating A/B/A regression injects GPU-baseline ownership failure after B and proves an older host A
fallback cannot leave B in guest memory when A returns.

## Plucky imported-depth guest preparation

The cold-image measurements above predated the adaptive compute-image cache sizing from #1519. A fresh
160-second comparison on the combined renderer confirmed that capacity now controls how often the old
`0x30180d0000` cold path occurs: a forced 512 MiB cache produced 113 calls with `setup_ms > 50` and
49,522 MiB of source snapshots, while the adaptive 2 GiB default produced three such calls and
21,796 MiB. The ordinary (not cold-filtered) program median was neutral at 2.87 versus 2.89 ms. This closes
the repeated cold-snapshot bottleneck as a priority; it does not make the whole title fast.

The next per-image trace instead exposed a renderer-authority error in sampled compute inputs. A persistent
Vulkan depth image can be imported directly even though it deliberately is not registered as a color render
target. The import populated the descriptor's `VkImage`, but the sampled-source condition checked only the
color `renderer_owned` flag. It therefore fell through to guest validation, cache lookup, and sometimes
conversion of raw depth backing that was stale and never consumed. A cache hit was also allowed to replace
the imported `VkImage` while the binding remained marked imported, making the redundant path a potential
correctness and lifetime hazard as well as CPU work.

Sampled guest preparation now runs only when neither renderer authority path supplied the image. The policy
is expressed as a small independently tested helper and covers guest-backed, color-renderer-owned,
direct-depth-imported, storage, and recovery-switch cases. `PROSPER_NO_IMPORTED_IMAGE_GUEST_BYPASS=1`
restores the previous fall-through for a same-binary diagnostic. Per-binding timing now separates renderer
query/import, cache lookup, staging, source preparation, allocation, view, and sampler costs;
`PROSPER_COMPUTE_TIMING_CODE=0x...` limits those records to one exact program without enabling compute trace
hashing.

Two 160-second full-resolution runs of the same binary and scripted route compared that recovery switch.
For `0x3017580000` binding 14, the imported-depth samples spent a 1.519 ms median in cache lookup and
1.524 ms total under the old path (282 imported samples of 283). With the bypass, all 98 observed samples
were imported and spent 0.000 ms in cache lookup / 0.003 ms total. Whole-program medians and aggregate
snapshot totals are not treated as an A/B result: the game reached different resource-authority states,
with binding 15 imported in 276 control samples but guest-backed in all fixed samples. That difference
changed the program mix despite similar presented-frame totals (1,403 / 1,350). The binding-14 comparison
isolates the fixed branch because its imported contract is identical in both runs.

An uninstrumented 90-second route presented 1,348 frames (about 15 FPS), confirming that the remaining broad
slowdown is real rather than entirely caused by diagnostics. A heavily logged run showed white flashes over
the studio/publisher intro; intrusive output clearly worsened cadence, but the flashes have not yet been
confirmed or disproved in a clean capture. Treat that visual report as an open correctness item, not as a
known timing artifact.

## Astro Bot compute decomposition (#1732) — the 3D workload this document asked for

This document's July stop decision deferred "the remaining synchronous graphics/compute boundaries"
until they could be evaluated against a **3D** workload. Astro Bot (`PPSA21564`) is that workload, and
it is a far sharper instrument than the 2D titles the earlier passes used: here the compute boundary is
worth roughly 20x rather than a few percent.

Measured on `a1f3b05c`, headless `boot_trace`, route
`scripts/astrobot/reach-worldmap-boot-trace.pad`, RADV STRIX_HALO, no other GPU consumer at either
boundary, no render acceleration.

### Compute execution is ~93 % of headless wall time

`PROSPER_RENDER_TIMING=1` reports the backend's own cumulative cost inside the compute-on run, divided
by the process's own `/usr/bin/time` wall clock: **45,900 dispatches at 9.39 ms = 431.0 s of 461.62 s
(93.4 %)**, and on a second run **139,150 dispatches at 5.70 ms = 793.2 s of 852.95 s (93.0 %)**, stable
across every sample of both. This
is a *single-arm* attribution, so unlike the `PROSPER_NO_COMPUTE=1` ratio it cannot be confounded by a
guest-path change. The backend is invoked **once per dispatch** (`calls == dispatches`), each invocation
one `vkQueueSubmit` + `vkWaitForFences` on the guest's own submit thread.

The `PROSPER_NO_COMPUTE=1` comparison is fair on the axis that matters, but its **multiplier is not a
constant**. That switch installs a no-op success backend that never mutates guest GPU resources, so the
guest could in principle take a cheaper path; it does not. `PROSPER_PROGRESS=5` reads guest
submits/draws/flips from the HLE AGC layer, upstream of any backend, and at matched flip numbers
cumulative **submits/flip converges to identical values in the two arms** (7.29 vs 7.29 at flip 1767).
The route being *flip-anchored* is what makes the arms comparable at all — a wall-time route presses its
buttons at different game states in the two arms.

Two things the ratio does **not** establish, both of which it has been read as:

- **It varies from 92x to 30x across one route** (92x at flip 323, 38x at flip 1200, 30x at flip 1767
  and still falling). A multiplier quoted without naming the flip means nothing.
- **draws/flip diverges at matched flips** — 138.6 with compute on versus 68.0 with it off at flip
  1636. Both arms reach the same content, but the content ramp is *wall-clock*-anchored, so the fast arm
  has simply not streamed the scene in yet. The wall-time ratio therefore mixes our compute cost with
  "the fast arm is rendering less"; prefer the single-arm number above.

And 67 flips/s is the rate with the work deleted. It bounds the cost, not the achievable ceiling.

### The decomposition

`tools/perf/compute_phase_report.py` aggregates the existing `[compute-phase]` and `[compute-image]`
records. 18,933 **succeeded** dispatches, 1,581 guest submits, 421.8 s:

| phase | share | ms | contents |
|---|---:|---:|---|
| dispatch | 75.4 % | 317,908 | command recording, `vkCmdCopyBufferToImage` per image binding, kernel, readback, submit/fence wait |
| setup | 17.4 % | 73,287 | descriptor validation 484 ms, buffer binding 1,264 ms, **image binding 71,539 ms** |
| writeback | 6.8 % | 28,509 | pack 10,139 ms, retile (`layout_ms`) 10,112 ms |
| pipeline + cleanup | 0.5 % | 2,112 | |

**Every percentage in that table is a share of succeeded-dispatch time (421,827 ms).** The image
figure is setup's `unattributed` row, i.e. the image-binding loop, on the same base. Do not compare it
against the tool's `setup image bindings` line: `[compute-image]` records carry no ok flag, so that
line covers **all** dispatches (388,600 ms, 50.2 % of the 774,834 ms every dispatch consumed including
the failures below) and is a strictly larger population. Mixing the two makes the child look **5.3x**
its parent (388,600 / 73,287) — exactly what trap 47 in `GAME_COMPAT_ORCHESTRATION.md` says an
aggregator must never allow.

Three instrumentation properties that generalise past this title, each of which produced a wrong table
before it was handled:

- **`setup_ms` spans the image-binding loop, which has no sub-timer on the `[compute-phase]` line.**
  With `PROSPER_COMPUTE_PHASE_TIMING` alone, setup's named children explain 0.4 % of a 17.4 % phase; the
  rest only becomes visible with `PROSPER_COMPUTE_IMAGE_TIMING` as well. The report prints an explicit
  `unattributed` row under every parent rather than dropping the gap.
- **A failed dispatch leaves `execute_item` early, so `phase_dispatch`/`phase_writeback` are never
  advanced.** The phase spanning the break is then computed backwards and prints **negative** —
  `dispatch_ms` for a break inside the dispatch window, `pipeline_ms` or `writeback_ms` for one either
  side, and no negative at all for a break before setup ends — while `cleanup_ms`
  absorbs the whole record. Summing failed records into the phase table produced
  `dispatch (GPU) = -20,532 ms` and moved 46 % of the run into `cleanup`. They are excluded and counted
  separately.
- **`[compute-image]` records carry no ok flag**, so they span failed dispatches too and must be
  denominated against all-dispatch wall time rather than the succeeded-only total.

The switches are cheap enough to trust: `avg_ms` at five matched dispatch ordinals against a run with
neither gives **+1.5 % to +1.9 %**, measured rather than assumed.

### Root cause: `0x500571000`'s dispatch size never resets

This subsection records the pre-#1819 defect. #1819 corrected GDS append/consume addressing and a
current-master run now produces bounded, nonmonotonic per-frame counts without device loss; the clean
post-fix frontier is the world-map image round-trip measured below.

One program is **75.2 % of succeeded-dispatch time** (317,271 ms of 421,827 ms, 323 dispatches, 982 ms
mean, 93 % of it in the submit-and-wait) in the 853 s routed run.
`PROSPER_COMPUTELOG_CODE=0x500571000` prints the derived geometry: `local` is a constant 16x16x1 and
`groups_x` grows by **exactly 3,345 every dispatch, monotonically, and never decreases across the whole
observed route** — 1 at dispatch 1, 3,366 at 23, 327,831 at 120, **752,646 (192.7 M invocations) at 247,
still climbing when tracing stopped**. A per-frame quantity is being **accumulated rather than
consumed**: the count reaching the backend is a running total where the guest plausibly means a
per-frame value.

That last sentence is the *hypothesis*, not the observation. `gpu_executor.cpp` copies
`groups_x = d.threads_x` verbatim from the packet when `USE_THREAD_DIMENSIONS` is clear, and neither
that path nor the deriving branch beside it accumulates anything — so where the running total comes
from is **not yet located**, and it may be a guest-side count we are reading at the wrong time rather
than a derivation defect. What is certain is the shape: this program cost 235 ms, then 562, 982,
1,492 ms across one run, and the title becomes monotonically slower the longer it runs.

**Every aggregate for this program is a function of route duration**, precisely because its cost grows.
Every figure in this section comes from the one 852.95 s decomposition run described above; an earlier
partial read of the same log, at 13,195 dispatches, gave 123,762 ms for the same program. Name the run
and the point in it whenever quoting an absolute cost here.

### Post-#1819 exact-hash world-map decomposition (2026-08-03)

A clean interactive F8 run on exact revision `b7a56cc2` reached `worldmap` without device loss at about
3.58 guest flips/s and identified stable SPIR-V hash `0x05975892ababe69f` as the largest current compute
cost. A following selected timing arm proved its own hash selector through one accepted banner, one
first-match witness and one terminal `seen=53507 matched=69 verdict=matched` summary. All 69 selected
dispatches succeeded; the normal app shut down after 540 presented frames.

The selected program consumed **31.285 ms/dispatch**, but only **3.450 ms** was the GPU dispatch:

| top-level phase | total | mean/dispatch | share |
|---|---:|---:|---:|
| setup | 1,216.7 ms | **17.633 ms** | 56.4 % |
| pipeline | 66.7 ms | 0.967 ms | 3.1 % |
| GPU dispatch | 238.1 ms | **3.450 ms** | 11.0 % |
| writeback | 630.5 ms | **9.138 ms** | 29.2 % |
| cleanup | 6.7 ms | 0.097 ms | 0.3 % |

Three 3840x2160 bindings explain 98.9 % of the 1,103 ms real image-setup population: storage binding
73 costs **10.505 ms/dispatch**, while sampled bindings 44 and 41 cost **2.683** and **2.633 ms**.
Binding 73 writes back on 67 of 69 matches, alternating two guest ranges: 51 records at 66,846,720
bytes cost 9.862 ms/write and 16 at 51,118,080 bytes cost 7.803 ms/write. Across them, layout/retile is
4.340 ms/write, retained-result cache work 3.533 ms/write and write-watch notification 1.436 ms/write.

This run also exposed an instrument defect. Storage binding 73 reported 10.486 ms/dispatch of
`prepare_ms` and 3.708 ms/dispatch of `cache_ms`, but the storage cache clock starts **after** the
prepare clock. They are nested, not siblings: exclusive seed preparation is about **6.777
ms/dispatch**. The old flat report added both, printed an impossible -252.5 ms residual, and was wrong
only in its image sub-table; the top-level phase table above remains valid. The report now prints
storage cache as an included child of preparation, checks the hierarchy per record, and groups real
bindings by stable shader hash + binding + class. Selected image records also carry address,
persistence and upload-skip state so one later exact-hash arm can distinguish real source changes from
avoidable cache/provenance invalidation without a broad compute log (#1732).

### Ruled out

- **"The current selected shader kernel itself costs about 30 ms."** False. On the valid post-#1819
  exact-hash arm the whole dispatch costs 31.285 ms, but the GPU interval is only **3.450 ms**;
  setup plus writeback consumes 26.771 ms and localises the current problem to the host image
  round-trip (#1732).

- **"The dominant dispatch cost is image upload volume."** False, and one matched pair settles it.
  Over the full 853 s run, counting only succeeded dispatches and non-alias bindings:

  | program | dispatches | staged | dispatch_ms |
  |---|---:|---:|---:|
  | `0x500571000` | 323 | 160.40 GiB | **295,879** |
  | `0x50059cd00` | 322 | 160.40 GiB | 2,364 |
  | `0x5005cc100` | 322 | 166.60 GiB | 2,354 |
  | `0x5005fdb00` | 322 | 166.60 GiB | 2,347 |

  `0x500571000` and `0x50059cd00` stage the **identical 160.40 GiB** across the same number of
  dispatches, and one takes **125x** longer. (A GiB/s column is deliberately omitted: dividing the
  outlier's bytes by its time yields "0.54 GiB/s", which reads as *its uploads are slow* — the exact
  conclusion this bullet exists to falsify. Its time is dominated by a 192 M-invocation kernel, and
  the bytes are incidental to it.) The per-dispatch image round-trip is a real and large
  cost elsewhere in this decomposition, but it does not explain the dominant term. The two columns are
  drawn from different record types, so the populations were checked rather than assumed: these four
  programs fail **once in 1,290 dispatches**, so their `[compute-phase]` and `[compute-image]` records
  cover the same work. Ranking phases by size selects uploads; asking what predicts the *outlier*
  rejects them (#1732).
- **"The geometry is fine, the 128,005-word SPIR-V module is the problem."** False, and the way it was
  nearly believed is the lesson: the first three traced dispatches genuinely read `groups=1x1x1`,
  because the growth only begins at dispatch 23. **Three samples of a monotonic series are not a sample
  of it** — when a value is suspected of drifting, plot the series before quoting any element of it
  (#1732).

### Separately: 63 % of dispatches fail, silently

**87,551 dispatches fail** — every one from submit 2299 (the world-map load) onward — consuming
**353 s, 46 % of all compute wall time, on work that is then discarded**. Mind the denominator:
106,484 dispatches reach `execute_item` and emit a `[compute-phase]` record, so the failure rate
*there* is 82 %, but a further **32,667 take the CPU fast path and return before `execute_item`**,
emitting no record. Against every dispatch the backend saw (106,484 + 32,667 = 139,151, consistent with
`[render-timing] compute calls=139,150`, which is printed only every 25 calls) the rate is
**62.9 %**. Read
`[render-timing] compute_cpu_fast fills=N` before quoting any rate from `[compute-phase]` counts.

At default verbosity the failures emit **nothing**: every failure path inside the dispatch body of
`execute_item` returns false behind an `if (trace)` guard, so a run log shows only the two registration
lines. (Two rejects before that body — an unsupported descriptor kind and a missing storage-image
device feature — do print unconditionally, and neither fires here.) This is a fatal-gap class under
`CLAUDE.md` and a likely mechanism for #1459 (Astro Bot's world map renders only its backdrop). Any
future compute-performance number for this title is measuring two different populations until it is
split by `ok=`, which `compute_phase_report.py` now does.

`CONFIDENCE: HIGH` on the decomposition, the ~93 % attribution, the fairness verdict, the monotonic
`groups_x` growth across the observed route, the failure census and both falsifications.
`CONFIDENCE: LOW` on *where* the accumulation happens — the obvious derivation paths do not accumulate,
so it is not yet localised and may be a guest-side count read at the wrong time.

## Next renderer step

First reproduce the reported white intro flashes with lightweight capture or a clean live run. Heavy
per-image/per-dispatch logging changes the title's already-slow timing and must not be the visual oracle.
If a capture retains the affected producer history, use GPU replay to separate a deterministic renderer
defect from presentation/cadence behavior.

For performance, a fresh adaptive-cache trace ranked `0x3017580000` and `0x30181a0000` at roughly 16.00 and
15.41 ms median. The imported-depth work above removes one dead input path, but a guest-backed 3840x2160
Float16 input in the fixed run still spent about 6.9 ms in source preparation. Measure a generic raw-FP16
upload/GPU conversion or exact native-sampling alternative against the known RADV regression that originally
selected RGBA8; do not simply re-enable native FP16 globally. In parallel, capture a fresh v39 rolling
temporal window around the Plucky title/gameplay transition. It must include the producers for the two
48x48 volumes and close with zero unresolved leaves, providing a native pixel oracle as well as exact compute
pipeline contracts. Keep the exact-byte and disable-switch A/B discipline used here, and preserve
screenshot/capture correctness while reducing the synchronous boundaries.
