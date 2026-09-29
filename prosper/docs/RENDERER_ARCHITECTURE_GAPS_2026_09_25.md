# Renderer architecture gaps (2026-09-25)

A structural review of what prosper's render path COSTS per frame. It began from the question "why are heavy
3D titles single-digit fps?" -- the correctness half of that framing is withdrawn in § 0. It is deliberately
*not* another leaf profile: #3770 and `PERFORMANCE_ROADMAP_HANDOFF_2026_09_23.md` already show
leaves being shaved for single-digit percentages, and two of those attempts measured *backwards*.
The claim here is that a handful of costs are paid by EVERY title on EVERY path, and that removing
one of those is the only kind of performance work that compounds across the corpus.

Every prosper-side figure below is from prosper's own code, profile, or status docs, and every
`file:line` was verified against the tree on 2026-09-25. The designs named — buffer device
addresses, timeline semaphores, page-granular write tracking — are published Khronos/OS
capabilities, not anyone's proprietary work.

## 0. WITHDRAWN: "the addressing model is the shared root cause"

The first two revisions of this document led with a claim that prosper's CPU-side static address
resolution (`src/gpu/recompiler/indirect/rdna2_indirect_pointer_descriptor_range.cpp`, 1,106 lines
of VGPR/SGPR live-range analysis) set *both* the performance and the correctness ceiling, and cited
GTA V's absent world as the correctness evidence.

**The correctness half is false and is withdrawn.** It was written from a `CLAUDE.md` in a shared
checkout 397 commits behind `origin/main`. Current `COMPATIBILITY.md` records GTA V at **rung 3** —
the prologue bank heist rendering in full colour on a default launch — and *Sonic Frontiers* with
the world rendering and 84% of the frame lit. prosper's static resolution demonstrably does handle
the shapes those titles use. Any future argument that prosper cannot express a resource-addressing
shape must be made from a specific failing draw, not from this document.

Two things follow, and the second is the useful one:

- The remaining per-draw cost of CPU-side resolution (`resolve_dynamic_fetch` 4.69%,
  `find_spirv_descriptor_binding` 1.91%) is an ordinary single-digit-percent perf item. It is
  recorded in § 4 and is **not** a frontier.
- **The comparison this document came from is now a controlled one, which makes it sharper rather
  than weaker.** Both prosper and the implementations it was compared against produce correct
  rendering on these titles. The output is therefore held constant and the only remaining variable
  is what the render path *costs*. That is exactly the comparison §§ 1-3 are about, and none of
  those three findings depended on the withdrawn claim: each is measured from prosper's own code,
  its own profile, or its own status docs.

Recorded rather than deleted so the next reader does not re-derive it. The general lesson is the
charter's own: a status quoted from a stale checkout reads exactly like a current one.

## 1. The GPU is idle inside long submits — barriers, not shading

`BLUE_PRINCE_STATUS.md:656` records `gpu_device` at **30–45 ms/submit** against a 16.7 ms budget
and concludes 60 fps is unreachable by CPU work alone. That conclusion is right, but the natural
reading of it — "so it is shading cost" — is contradicted by the same title's own measurement:
`radeontop` puts the GPU at **4.17% busy** against a `vkcube` control at 56.31%
(`docs/GPU_PROFILING_EXTERNAL.md`). A 4%-busy GPU is not computing for 30–45 ms; it is stalling.

The backend has **40 `vkCmdPipelineBarrier` sites**, a number of them
`VK_PIPELINE_STAGE_ALL_COMMANDS_BIT -> ALL_COMMANDS` (`render_runner.h:2386`, `:2537`, `:6299`,
`:6547`) — the heaviest barrier expressible, a full pipeline drain on both sides. Per draw or per
target transition, that serialises the whole GPU.

This is listed first among the performance items because it is the largest unexplained quantity in
the project: a 26x gap between GPU time charged and GPU work done.

CONFIDENCE: HIGH on the barrier inventory and the utilisation figures; MED that barriers are the
dominant cause of the stall rather than, say, layout transitions or WAW hazards on render targets.

## 2. The CPU blocks on the GPU at every submission

`render_runner.h:2609` `submit_and_wait()` is the only submission path: create a fresh `VkFence`,
`vkQueueSubmit`, `vkWaitForFences`, and on timeout fall through to `vkQueueWaitIdle`. prosper's own
comment at `render_runner.h:1305` records the frequency — **15.68 queue submits and 15.68 fence
waits per Blue Prince frame**. Sixteen full drains; the CPU and GPU never overlap.

`grep -rn 'timelineSemaphore\|VkSemaphoreTypeCreateInfo'` returns nothing: prosper uses no timeline
semaphores, though they are core since Vulkan 1.2. The standard shape is one device-wide timeline
semaphore; submit takes the next monotonic tick, signals it, passes a **null fence**, and returns
without waiting. Completion becomes a non-blocking `known_tick >= tick` query and resource lifetime
becomes "retire what has passed", drained opportunistically. Real CPU waits then survive only where
a consumer needs bytes back — which `flush_now` (`:12048`) already classifies and counts.

CONFIDENCE: HIGH — submission path, absence of timeline semaphores, and the 15.68 figure are all
direct reads of the tree.

## 3. "Did this buffer change?" is answered by reading every byte

`render_runner.h:4398` decides residency reuse with
`std::memcmp(entry->owner->snapshot.get(), source, key.bytes)`. The **hit** path therefore reads
`2 x bytes` to conclude nothing changed, and prosper keeps a second full CPU copy of every resident
buffer purely to compare against. A miss adds two more full copies (`:4414`–`:4415`).

A write-watch exists and is wired in, but is gated into irrelevance (`:4333`–`:4365`): two full
equality validations before it arms, and **two dirty queries disable it permanently**
(`watch_disabled = true`). Buffers that change occasionally — the common case — fall back to
unconditional `memcmp` forever.

The 2026-09-23 GTA profile is consistent: `__memmove_avx512` 11.63% self, `__memcmp_evex_movbe`
4.51% self, renderer exact source validation 91,021 calls / **1.63 GB in 30 s**, renderer upload
batches 40,009 calls / **2.84 GB in 30 s**.

Page-granular write-protect tracking makes a clean buffer cost **zero** bytes to validate instead
of `2 x bytes`. prosper already owns the hard half — `src/host/memory/guest_write_watch.cpp` plus
the fault handler in `exec_image_linux.cpp` — so this is a policy change, not new machinery.

CONFIDENCE: HIGH on mechanism and gating; MED on attributing the profile shares specifically to it,
since those are sampled CPU on one TID and `memmove` has other callers.

## 4. Smaller: per-pass submission granularity, interpreted per-draw resolution

Submission is flushed at pass boundaries (`:12048`), which is a drain while § 2 holds. Once
submission is non-blocking the natural policy inverts: hold one command buffer across passes and
flush on a draw count plus the genuine sync points.

Separately, the per-draw resource resolution in § 0 is re-derived every draw for something fixed
per *shader*. Compiling each shader's resolution plan once into a flat ordered op list and
replaying it is the standard fix — worth real single-digit percentages, and listed last because
§§ 0–2 are worth multiples.

## 5. What this predicts, and how to falsify it

If this is right, #3770's result is expected rather than surprising: descriptor-set reuse measured
backwards because the frame is not descriptor-bound. It is bound by a stalled GPU, sixteen pipeline
drains, and reading guest memory twice to decide it had not changed.

Three independent discriminators, cheapest first. Each is falsifiable and none requires the others:

1. **Barriers (§ 1).** Instrument the existing 40 sites with a per-frame count and a device
   timestamp either side. If the `ALL_COMMANDS` barriers do not account for the gap between 4% GPU
   utilisation and 30–45 ms/submit, § 1 is wrong and the stall is elsewhere.
2. **Submission (§ 2).** Change *only* `submit_and_wait` to a timeline tick plus deferred
   retirement, keeping every existing copy and compare. If frame time does not move, § 2 is wrong.
3. **Residency (§ 3).** Arm the existing write-watch permanently for one buffer class instead of
   disabling it after two dirty queries, and compare `buffer_resident_compared_bytes` against the
   frame time. If removing 1.63 GB / 30 s of comparison does not move the frame, § 3 is wrong.

**A positive control is mandatory and must not come from the same route** (charter, same-source
control rule): a title already known to be GPU-bound rather than submit-bound should *not* improve
under (2). If everything improves uniformly, the instrument is measuring itself.

**Each of these is PATH-shaped, and that is the selection criterion.** A fix that makes one title
faster by steering it onto a different branch does not compound: the next title lands on a
different branch and arrives at the same frame rate. Measured 2026-09-25, the GPU/render layer
reads **501** distinct `PROSPER_*` switches — 77 diagnostics, 108 `PROSPER_NO_*` A/B opt-outs, and
**316 behaviour-changing**. That is the size of the space a title-shaped fix searches, and the
reason per-title performance results have not transferred. Prefer a change that deletes a cost
every path pays over one that finds a cheaper path.

## Findings from the 2026-09-25/26 measurement pass

Recorded here rather than only in commits, because two of the three are negative and a negative
result that lives in a commit body is one the next agent re-derives at full cost.

**LANDED: the CPU RTT snapshot pool matched on exact size instead of capacity.**
`CpuRttSnapshotPool::copy` reused a retained buffer only when its `size()` equalled the request, so
every buffer that was merely big enough was refused and the publication fell to `assign()`, which
allocates. Nothing counted whether the pool worked -- `CpuRttSnapshot::reused` was read only by its
own unit test. Measured on Astro Bot, headless, ~120 s, all arms inside a verified-clean window:

| | copies | hit | allocated |
|---|---:|---:|---:|
| before | 14,089 | 38.5% | 56,543 MiB |
| after | 15,519 | 99.5% | 1,096 MiB |

Throughput over the same wall clock went 10,323 passes (before, n=1 at matched instrumentation) to
10,906 / 11,075 / 11,313 (after, n=3). Non-overlapping, so read the direction as established and
the magnitude as roughly 5-9%.

### Where Astro Bot's time actually goes, and the next frontier

Established by measurement on 2026-09-26, so the next reader starts from facts rather than from
§§ 1-4's hypotheses:

- **Not the renderer.** 7.8% of wall clock inside `render_draw_pass_rgba`.
- **Not GPU-bound.** Of 89 threads, exactly ONE sits in `drm_syncobj_array_wait_timeout`.
- **Not idle, and not parallel.** 147.5 CPU-seconds in 51.8 elapsed -- about 2.85 of 16 cores.
- **Not blocked on prosper.** 86 of 89 threads park in `futex_do_wait`, and their stacks are
  overwhelmingly GUEST primitives: `k_sce_cond_wait`, `k_sema_wait`, `k_posix_sem_wait`,
  `k_ef_wait`. That is Astro Bot's own worker pool idling, not a prosper defect. Do not re-derive
  this from the thread count: 89 threads and 96.8% "blocked" looks alarming and means nothing.
- **Not thread creation.** Ruled out above on an interleaved A/B.

What is left, with numbers, is the guest write-watch. Over ~120 s:

```
query=152,436  unchanged=145,008 (95.1%)  dirty=7,131  faults=5,684
rearms=73,260  host_write=53,008 notifies
pages_hit=50,816,105        (~959 watched pages unarmed PER notify)
page_protect=28,801 calls / 140,038 MiB of range re-protected
```

Read those two halves in opposite directions. The **query** side is working: 95.1% of queries
answer "unchanged" and avoid a copy, which is the mechanism § 3 argues for extending. The
**notification** side is where the cost is -- `guest_write_watch_notify_host_write` unarms about a
thousand pages per call, 53,008 times, and `mprotect` appeared in 4 of 14 active profile samples.
With 89 live threads each `mprotect` also carries a TLB shootdown to every core, so its cost scales
with the guest's thread count rather than with the range.

**This is measured volume, NOT a demonstrated defect.** A notify's page count is proportional to
the range the host actually wrote, so much of it may be irreducible. Before optimising it, show
that the unarm path is on the critical path -- the same discriminator this document demanded for
pass batching, and the one that killed it.


### WITHDRAWN: "§§ 1-3 have one root cause: readback"

This section claimed readback was 68.5% of a backend call on the shipped frontend and forced one of
every two flushes, making it the reason submits block. **Both figures came from runs where GPU
present was INACTIVE, and the conclusion is withdrawn.**

`tools/screenshot` never calls `set_gpu_present_active` — the charter says so, and it is still true.
The run that was supposed to be the control, `prosper-app` under `SDL_VIDEODRIVER=offscreen`, logged

```
[app] GPU present: surface on shared instance failed
      (VK_EXT_headless_surface extension is not enabled); using own device
```

and so never reached `set_gpu_present_active(true)` either. Two different harnesses, the same
forced-readback path, and the agreement between them was read as evidence that the shipped path
behaved the same way. It does not.

**Measured on the real windowed path, GPU present confirmed adopted:**

| title | raw policy-positive slots (before caller gate) | renderer share of wall clock | rate |
|---|---|---:|---|
| The Messenger | **1 of 36,897** (100.0% not-wanted) | 6.7% | 59.9 fps, 8,880 frames |
| Astro Bot | 293 of 19,233 (98.5% not-wanted), 8.9 GiB potential extent | 9.4% | single-digit |

So readback is **not** the render path's dominant cost, the submit does **not** block because of it,
and § 2's timeline-semaphore idea is neither blocked by it nor rescued by it. The historical
"Astro Bot copies 8.9 GiB under `no-color-target`" conclusion is also withdrawn: that census
classified the raw target policy but did not apply the backend's independent
`want_color_readback` gate. The 8.9 GiB was potential extent, not proven traffic.

The missed gate was exposed on 2026-09-28 by the same-binary Outer Wilds pair for #3914: the old census
reported 542,982.6 MiB "copied" under `no-color-target` while the live caller passes no target
for depth-only passes and sets `want_color_readback=false`. The backend gates both allocation and
`vkCmdCopyImageToBuffer` on that value. A split segment or a disabled live-target path can still
request bytes with no target, so the corrected census must count the *effective* request rather
than assume the whole historical bucket is zero. The historical figures alone did not establish
actual readback volume; the corrected runs below resolve the sampled shipped paths.

The corrected, GPU-present Outer Wilds route classified 206,786 slot-0 checks: 206,785
`not-wanted`, zero `no-color-target`, and one explicit request for 31.6 MiB of planned
extent. Its F8 window measured 0.2 ms of readback. These are not equal-scene before/after
throughput measurements; they show that the old 542,982.6 MiB line was an instrument error,
not an optimization target. The requested extent is still not a completed-transfer count.
On the same corrected binary, Astro Bot's opening classified 7,483/7,483 slots as
`not-wanted` and its F8 window measured 0.1 ms of readback. The historical 8.9 GiB
bucket was likewise not a measured copy cost. Compute occupied 3,745.8 ms of that
five-second F8 window's retained components; that is the next measured lane, with
storage materialization, RTT snapshots and detile still separately substantial.

**The general lesson is the expensive half.** A harness artifact was ruled out by comparing two
harnesses, and both had the same artifact. *Agreement between instruments is not independence.*
Check the mechanism directly — one `grep` for the log line that says the path activated — rather
than inferring it from two measurements that agree.

**What this does NOT withdraw:** the CPU RTT snapshot-pool fix, which was re-measured on the
windowed path and stands. Astro Bot performs **24,991** pool copies there at a **99.5%** hit rate
with 113 misses; before the capacity-matching change the same workload missed 61.5%. The pool is
genuinely exercised on the shipped path, so that fix is real.

### Historical raw-policy readback investigation — conclusions withdrawn

The following 2026-09-26 slot-0 census classified target policy before the caller gate.
Its percentages cannot rank effective readbacks and the conclusions below are preserved only
to show the path to the later corrections. They must not guide a new optimization:

| title | not-wanted | explicit-request | bound-non-persistent |
|---|---:|---:|---:|
| The Messenger | 49.4% | **50.6%** | **0** |
| Astro Bot | 97.4% | 0 | **0** |

**`bound-non-persistent` never fires.** So "make more render targets persistent", which is the
obvious first move and the one this document would otherwise have recommended, buys exactly
nothing. That is the whole reason the census was built before the fix.

Tracing the remaining half is the useful part, and it ends somewhere specific:

1. `BackendColorTarget::readback` **defaults to `true`** (`render_runner.h:362`), and the live
   renderer sets it `false` in exactly one narrow volume case. So "explicit request" is mostly the
   *default* surviving, not a caller asking.
2. But slot 0 is not un-deferred: `live_renderer.cpp:10675` builds the target with
   `base != 0 && !defer_readback`, and `defer_readback` (`:10547`) goes through
   `can_defer_scanout_readback`.
3. Both gates on that path are **default-on** -- `live_gpu_targets` and `defer_rtt_readback` are
   each a chain of `!PROSPER_*` opt-OUTs. Nothing is switched off.
4. What defeats the deferral is `cpu_needed_same_batch`: a consumer later in the SAME submit needs
   authoritative bytes, so deferring would make it observe the previous submit.

So the readback that remains is not a missing switch, a stale default, or a residency gap. It is
the architecture: **a render target that a later draw in the same submit samples is served from CPU
bytes**, so the pixels must exist on the CPU before that draw records. That is what
`CpuRttSnapshotPool` feeds and what forces the flush.

**CORRECTION, from the windowed measurement above.** The paragraph that followed named
`cpu_needed_same_batch` as what defeats the deferral. It is not: `PROSPER_READBACK_WHY=1` reports
`same_batch_cpu=0` and `same_batch_reasons=storage:0,dimension:0,extent:0,feedback:0` on BOTH
frontends. What actually fills the bucket is `vo_final` — 53,399 of 53,400 under `tools/screenshot`
and 11,599 of 11,600 under offscreen `prosper-app` — and `vo_final` fires precisely because
`final_gpu_present` requires `gpu_present_active()`, which neither of those runs had. On the real
windowed path the whole bucket collapses to 1 slot in 36,897.

So there is no sampled-render-target round trip to remove on the shipped path, and the sentence
below is withdrawn with the section above. It is kept, struck, because the reasoning was sound and
only the input was wrong — the trap was reading agreement between two harnesses as independence.

~~**The frontier is therefore GPU-side resolution of sampled render targets, not readback tuning.**~~
Partial machinery already exists (`rtt_gpu_seed_import_extent_compatible`,
`retain_gpu_result_baseline`, `renderer_result_retained`). The measurable target is
`cpu_needed_same_batch`: every pass where it is true is a pass that cannot defer, and counting
*why* a consumer is CPU-needed is the next census, not the next fix.


## Ruled out

**RULED OUT (2026-09-30): an absent decode-budget announcement proves its environment value has
not been read yet.** The original `live_renderer_registration_statics` harness passed with the
budget read moved above the warmup return and its report left below it: both callbacks saw the
same 77 MiB setting. Changing the setting from 55 to 77 MiB between the skipped and in-window
callbacks makes that mutation fail one assertion while the restored renderer passes. The report's
placement alone cannot pin the read's placement. This guard samples the process gates and budget;
it does not directly prove decoded-cache, texture-identity or thread-local-cache preservation.
See [#3968](https://github.com/mattias800/prosper/pull/3968) and
[#3892](https://github.com/mattias800/prosper/issues/3892).

**RULED OUT as the first visibility boundary for Astro Bot's observed opening route (2026-09-29):
the CPU read of the next indirect dispatch's argument triplet.** An exact argument-read observer
and binding-60 output selector found the first read at `0x5074063e0+12`, disjoint from the pending
result at `0x514080000+16711680`. In the bounded trace, none of 306 retained selected results were
*first materialized while pending* by a raw-buffer read; all 306 first materializations were
ordered memory effects. The first sampled overlap was the next dispatch's image output to the same
guest range, across a parser stall. The census no longer checks overlap after an earlier event
materializes a result, so this says nothing about later raw-buffer reads. The process exited
mid-submit, making the summary verdict invalid. This is a complete-prefix observation, not a
deferral authorization. Physical aliases and image ownership remain unproved. See
[#3157](https://github.com/mattias800/prosper/issues/3157).

**FALSIFIED: "small render passes are what make heavy titles slow."** This document's own § 1-2
motivated a pass-batching change. The pass-cost census measured it instead: Astro Bot spends
**7.8% of its wall clock inside `render_draw_pass_rgba` at all**, so collapsing passes could not
have moved it whatever the pass count. The Messenger, which runs well, spends 63.8%. Pass length is
real (Astro Bot's passes hold a mean of 1.00 draws, 100% single-draw) and it is not this title's
frontier. Do not restart pass batching without first showing the renderer is a large share of the
target title's wall clock.

**RULED OUT (2026-09-27): recycling per-pass Vulkan objects to buy frame time.** Kernel buffer-object
churn looked like ~9% of the GTA V render thread in sampled profiles, so descriptor pools (per render
pass and per GPU-detile upload), per-batch fences and per-batch timestamp query pools were put on
bounded free lists behind same-binary opt-outs. The mechanism moved exactly as intended: strace of
the render thread over 10 s of gameplay showed `AMDGPU_GEM_CREATE` 13,649 → 7,396 (−46%),
`SYNCOBJ_CREATE` 3,765 → 739 (−80%) and all ioctls −35%. The frame rate did not: interleaved pairs
8.47/8.46, 8.45/8.21 and 7.97/7.89 fps (OFF/ON), with ON never ahead of its OFF neighbour. Halving
BO churn is not a frame-time lever on RADV, whatever its sample share says. Branch
`perf/descriptor-pool-reuse`, #3873. Instrument trap found on the way: `PROSPER_NO_DESCRIPTOR_POOL_REUSE`
is already read by the compute context's own pool reuse (`live_compute.cpp`), so an A/B that reuses
that name disables both; the contaminated A/B showed a phantom +3%.

**RULED OUT (2026-09-27): an early exit in `index_buffer_is_unannounced_32bit`.** Exact (the verdict
is a conjunction, so it can return at the first disqualifying word), but a GTA V render-thread
profile showed no drop in the function's self share (below 0.36% before, 0.58% after, single 15 s
windows). GTA's unannounced index buffers are most likely genuinely 32-bit, which need the full scan
either way. Not shipped. #3873.

**RULED OUT: replacing per-call worker-thread creation with a persistent pool.**
`parallel_compute_texels` and `parallel_rows` each create a fresh set of `std::jthread`s per call
and join them. Measured volume on Astro Bot: **19,738 calls creating 200,809 OS threads** in ~120 s,
with `pthread_create` at 11.4% of the busiest thread's non-idle samples and `mmap`/`munmap`
alongside. That looks like an obvious win and is not one.

A `WorkerPool` was built (TSan-clean, seven-arm contract test) and adopted behind a runtime gate so
both arms lived in ONE binary. Interleaved A/B on a freed box, three rounds each:

| arm | runs (passes per ~120 s) | mean |
|---|---|---:|
| per-call jthreads | 11,019 / 8,984 / 10,088 | 10,030 |
| pooled dispatch | 10,345 / 10,039 / 10,625 | 10,336 |

The ranges overlap heavily -- the jthread arm holds both the best and the worst run, and its worst
started at load 7.70. **No difference is detectable at n=3 against this variance**, so the pool was
removed rather than shipped as a 317th switch for no measured benefit. The mechanism is worth
knowing: glibc caches thread stacks, so a warm `pthread_create` costs about what a condvar wakeup
costs, and a pool's shared queue is then pure added contention. **Do not restart this from the
thread count alone** -- 200,809 creations is a real number and it buys nothing.

**A sequential A/B of the same question first measured "~50% slower", and that was an artifact.**
A peer agent's `prosper-app` started mid-session holding ~387% CPU; every pooled arm landed after
it started, and the reverted build then re-measured at 4,451-4,825 passes against a 10,906-11,313
baseline taken before it -- the "regression" reproduced with the change absent. It had already been
written into a code comment as established fact before the revert failed to restore the baseline
and exposed it. **Interleave A/B arms rather than running them in sequence**, and record the load
at each run's start: this whole episode is visible in one column of the results table and was
invisible without it.

The shipped helpers are byte-identical to main (git diff shows additions only); only the
volume census was kept, because the volume is the input to any future batching question.


- **"The recompiler or shader quality is the frontier."** Not for this cost class: the profile's
  top self shares are `memmove`, `malloc` and `memcmp`, none of which is shader work.
- **"It is shading cost."** Falsified for Blue Prince by `radeontop`: 4.17% GPU busy against a
  56.31% control, and `PROSPER_RENDER_SCALE=2` did not reduce `gpu_device` (43.47 vs 45.24 at
  matched draw counts) while pixel-proportional readback halved as expected (#2276).
- **"Descriptor-set reuse is the win."** Falsified by measurement in #3770: 22.886% exact
  pass-local repeats, and both the targeted leaf and the enclosing renderer moved the wrong way.
  Do not revive it from repeat counts alone.
- **"prosper is naive about dynamic state."** It is not: `render_runner.h:9979`–`9995` enables 16
  dynamic states including extended-dynamic-state depth/stencil/cull/topology. Pipeline-key
  explosion from baked dynamic state is not a live hypothesis here.
- **"Per-reference texture resolution needs a memo" (#3873 plan item 2).** Not what the cost was.
  `PROSPER_TEXREF_CENSUS=1` on GTA V gameplay: ~158k texture references per 5 s, ~158 per submit
  against 41.6 distinct identities per submit (74% are exact in-submit repeats), and **no repeat
  resolved to a different outcome** (0 of ~117k per 5 s window, in every window of the run) -- so a
  memo would have looked safe. But the
  census also timed the chain by stage, and 4.3 us of the 4.8 us an ordinary in-submit reuse cost
  was ONE step: the volume-alias check walking the whole RTT cache (~850 entries, 1 volume) on every
  reference once any volume target had existed. With that walk replaced by an exact candidate index
  (`shared/rtt/volume_target_index.hpp`), an in-submit reuse costs 0.59 us and the chain averages
  2.5 us/ref; the most a memo could still save is ~0.5 us x ~120k refs per 5 s, about 1% of wall,
  for an invalidation surface of the #611/#780 class. Do not build a resolution memo without first
  showing a chain stage that is both slow and repeat-invariant -- the census reports exactly that.
- **"The expensive texture-reference classes from the first census are still there" (re-measured
  2026-09-28, after #3877/#3882/#3883/#3886).** Two are gone, one was not what it looked like. Sonic
  Frontiers' `rtt` class (73-117 us/ref, all in the `depth` stage) is now 0.8-1.0 us/ref: #3882's GPU
  gather removed it. GTA V's `persist_submit` class no longer appears, and the whole GTA chain is
  2.5-5.8 us/ref per class, with no stage above 4 us. What remained was Sonic's `other` class, ~250
  refs per 5 s at 2.4 ms each (~12% of wall on the title and menu screens). It was ONE texture: a
  3840x2160 RGBA16F dim-5 view whose live RTT entry was an identity-only shell (no valid image, no CPU
  snapshot, no uniform colour), so renderer authority kept it out of the decode cache and it was
  re-read and re-detiled on every submit. Retaining that shell's decode behind exact validation took
  the menu chain from ~640 to ~33 ms per 5 s and the menu from 24-28 to 30 flips/s, the guest's own
  cap (`live_rtt_base_slice_blocks_decode_cache`, #3873). **Two classes remain, neither of them a
  resolution-chain defect.** (a) The opening movie's `persist_invalid` refs (~290 per 5 s at 4.4 ms)
  are the video's Unorm8 planes, which really do change every frame. Their cost is the CPU detile plus
  a full-size exact compare that is certain to fail. (b) Sonic gameplay's `other_nocand` refs (~300
  per 5 s at 0.6-1.1 ms, 4-7% of wall) are non-BC Float16 cubes, deliberately excluded from retention
  since the Plucky Squire regression (see `RENDERER_PERFORMANCE_2026_07.md`). Reopen (b) only with a
  per-title A/B that shows that churn is absent.

**RULED OUT (2026-09-27): "after #3877 another single exact O(n)-per-draw hotspot remains on the
GTA V render thread."** A source-line profile of the submitting thread (Release with `-g1` line
tables, plain `perf record -F 499` on the whole process in a settled gameplay window, samples
histogrammed by innermost *prosper* source line and by inlined call site; library leaves such as
`memmove`/`malloc` attributed to their prosper caller through AMD LBR `-j any_call`) found no source
region above ~1.3% of the thread outside the known architectural items (compute result write-back,
`parallel_compute_texels`, buffer-range copies, readback snapshots). The largest remaining exact
candidates are flat and small: `RegisterFile` lookups ~2-3% spread over every register read,
`ShaderCompileKey` hash + compare ~2%, `std::set` node allocation for `srt_seen` in
`build_stage_table` ~1%, the FNV hash over every SPIR-V word in
`validate_spirv_descriptor_interface` ~1%. None is individually measurable against GTA's ~2-3%
run-to-run spread. The one exact item above that line was per-dispatch compute program analysis
(the native-multiwave probe, re-run on every compute dispatch), fixed by `compute_program_facts`
(+3-6% Sonic Frontiers / GTA V, #3873). Outer Wilds' largest single site is the backend buffer-range
copy (`parallel_render_memcpy_batch`, ~11% of user samples) -- § 3's "did this buffer change?"
problem, not a missing index. On Sonic Frontiers (profiled before #3882) 4.3% of the thread was the
retained depth-array float expansion loop, which #3882's GPU gather removes rather than speeds up.

## The frontier, found 2026-09-26: compute-item execution, not the renderer

Everything above measures the render path. It was the wrong place to look, and the census that
settled it is in this PR's own instrumentation.

**Measured on the shipped windowed frontend, GPU present confirmed adopted:**

| | compute-items | renderer (pass-cost) | rate |
|---|---|---:|---|
| Astro Bot | **74.9% of wall**, 66,028 calls, **1.917 ms/call** | 8.8% | ~5 fps |
| The Messenger | 0.1% of wall, 20,055 calls, **0.005 ms/call** | 4.3% | ~59 fps |

Comparable call counts, **383x the per-call cost**. The renderer is a single-digit share on both, and
the *faster* title spends a smaller absolute share there. Two earlier instruments disagreed about
the render path's share -- 58% from the per-submit report against 8.5% from pass-cost -- and the gap
was this: compute items reach the backend through the **ACB submit path**, which the per-submit
render report does not count. The largest consumer of the frame had no instrument at all.

### What the 1.9 ms is

A 39 s run that never leaves the static "Sony Interactive Entertainment" splash:

```
[compute-items] calls=15115 in-compute=29526ms (75.9% of span) mean=1.953ms/call
[worker-spawn]  compute-texels ~9,712 calls at 20-35 MiB each; detile-rows 2,419 calls
205 presented frames -> 5.3 fps
```

**About 240 GiB of CPU texel conversion in 39 seconds -- roughly 6 GiB/s -- on a screen that is not
changing.** CPU detiling is only 0.7% of bytes moved; the conversion dominates. The path is
`execute_item -> storage_unpack_range -> parallel_compute_texels`, which materialises storage-image
texels into RGBA32 quads on the CPU before a dispatch.

**The magnitude was already documented in the code** and nobody had connected it to a frame rate.
`live_compute.cpp`'s comment above `storage_unpack_range` records **~56 ms per image-bearing
dispatch against 0.39 ms of actual GPU dispatch** -- a ~143x ratio -- measured on a different title.

### Why the GPU path is not taken, from the gate's own census

`PROSPER_COMPUTE_STORAGE_GATE_CENSUS=1`, same splash, shows a mip/bloom pyramid evaluated hundreds
of times per geometry:

```
  1x1   bpe=16  evaluated=219   persistent=0     CANDIDATE=0
  7x4   bpe=4   evaluated=186   persistent=0     CANDIDATE=0
 15x8   bpe=4   evaluated=186   persistent=0     CANDIDATE=0
 30x16  bpe=4   evaluated=371   persistent=0     CANDIDATE=0
240x135 bpe=8   evaluated=1027  persistent=1027  CANDIDATE=660
```

Every geometry with `persistent=0` is `CANDIDATE=0`: it can never take the GPU path and always
materialises on the CPU. `dcc_safe`, `exact` and `write_only` are satisfied almost everywhere, so
**persistence is the binding constraint**, not correctness of the shape.

### Why this is the compounding target

It is shared code with no title in it, and the cost is per-call rather than per-pixel -- which is
exactly the shape the project owner reported from the keyboard: single-digit fps from the very
first frame, identical on a static splash and on an FMV. Scene complexity does not behave that way;
a per-call constant does.

**The question to answer next is why these images are not persistent**, since that is the one gate
term that fails. A secondary, independently useful item: `GpuRetilePipeline::bind` creates AND
destroys a `VkDescriptorPool` per call (31,262 times on this route, ~43 per frame, each a kernel
round trip), and the GPU retile census shows 1,606 of 4,818 images declining with
`mip-tail-or-offset` and falling back to CPU detiling.

**Do not read this as "fix Astro Bot".** The Messenger runs the same code at 0.005 ms/call. What
differs is how much of its work lands on a path that materialises CPU-side, and that path is
general.

### Inside the 74.9%: two size thresholds tested, both NOT the cost

The gate census named `persistent` as the one failing term, so the obvious reading is that a
residency size threshold excludes the common case. Two such thresholds exist and **both were
tested and neither is the cost.** Recorded because each looks compelling and each is wrong.

**Storage images, 4 KiB threshold (`live_compute.hpp:56`) -- FALSIFIED on an interleaved A/B.**
`persistent_compute_image_enabled` returns `bytes >= 4 KiB`, and the pyramid levels below that
(1x1, 7x4, 15x8, 30x16) report `persistent=0 CANDIDATE=0`, so they can never take the GPU path.
`PROSPER_COMPUTE_STORAGE_IMAGE_CACHE_MIN_KB=0` is a same-binary control. Three interleaved rounds:

| arm | frames | compute-items share |
|---|---|---|
| default (4 KiB) | 321 / 313 / 305 | 76.1% / 75.7% / 76.3% |
| MIN_KB=0 | 311 / 301 / 306 | 75.8% / 76.0% / 76.0% |

**The lever demonstrably moved** -- `7x4` and `15x8` went `persistent=0 CANDIDATE=0` to
`persistent=186 CANDIDATE=186`, verified by re-running the gate census under both arms -- and
nothing changed. So candidacy is not what forces the CPU work, and those images are genuinely
cheap. This is a negative result with a verified lever, not a void one.

**Compute buffers, 1 MiB threshold (`live_compute.cpp:1139`) -- real, but an order of magnitude too
small.** `persistent_compute_buffer_enabled` returns `bytes >= 1 MiB`, above a comment reading
*"Small bindings are cheap and numerous."* Measured over ~45 s with
`PROSPER_COMPUTE_BUFFER_TIMING=1`:

```
31,280 binding records, 89.6% below 1 MiB, median binding 192 B
compared 2.12 GiB total, 1.10 GiB (51.7%) from sub-1MiB
uploaded 0.73 GiB total, 0.42 GiB (57.6%) from sub-1MiB
```

Sub-threshold bindings really are the majority of compare traffic, and every one reports
`cache=ineligible persistent=0 total-watch-chunks=0 upload-skipped=0` -- no write-watch, so a full
compare and a full upload on every dispatch even when nothing changed. But **1.10 GiB over 45 s is
24 MiB/s**, and the thing to explain is ~3 GiB/s. Worth fixing on its own terms; not the frontier.

### So the cost is the unpacking itself

What remains, and what the profile pointed at from the start, is
`execute_item -> storage_unpack_range -> parallel_compute_texels`: **storage-image texels
materialised into RGBA32 quads on the CPU before a dispatch.** The worker-spawn census puts that at
roughly 240 GiB of `count x (src_stride + 16)` work in 39 s -- about 120 GiB of real texel data,
~3 GiB/s, on a screen that is not changing.

That is not a threshold to retune. It is the compute path binding storage images by materialising
them host-side rather than as GPU storage images, and `live_compute.cpp`'s own comment prices it at
**~56 ms per image-bearing dispatch against 0.39 ms of actual GPU dispatch**.

**Next question, and it is the architectural one:** under what conditions does a compute dispatch
take the native storage-image path rather than materialising, and what fraction of dispatches on a
real title can take it? The gate census answers the first half already; nobody has asked the
second.

### The whole chain, measured end to end (2026-09-26)

Every step below is a measurement on the shipped windowed frontend with GPU present confirmed
adopted, not an inference from a profile leaf.

```
1. compute-item execution        74.9% of wall   (renderer: 8.8%)
2. host-copy transfer pressure   2,058 MiB/s     (a title at 59 fps: 4 MiB/s)
3. largest category              storage-materialize, 39.4 GiB in 49 s
4. of those copies               8,993 copies from 84 DISTINCT SOURCES
5. why the cache did not skip    not-persistent 6,430 (76%) / persistent-but-watch-dirty 1,984 (24%)
```

**~99% of the largest host-copy category is re-copying images that did not change.** 39.4 GiB of
copying carries at most ~378 MiB of distinct data (84 sources averaging 4.5 MiB).

**The machinery to avoid it already exists and is already respected.** `acquire_cached_image` holds
a `GuestWriteWatch` and returns `upload_skipped`, and the copy site guards on
`!(bi.persistent && bi.upload_skipped)`. So this is not a missing mechanism; it is a cache that is
not admitting the working set. **76% of the copies are of images that are not persistent at all**,
which is an admission question -- key, eligibility, or capacity -- and only 24% are images the
cache holds but whose watch reports dirty or unknown.

Those two terms have completely different fixes, which is why the split is the useful number and
the aggregate was not.

**What this is NOT.** It is not a residency *threshold* problem: the images here average 4.5 MiB,
three orders of magnitude above the 4 KiB eligibility floor, and the A/B on that floor is recorded
above as falsified with a verified lever. Do not start there.

**The next question, and it is one measurement away:** for the 6,430 non-persistent copies, does
admission fail on the cache key, on an eligibility predicate other than size, or on capacity and
eviction? `acquire_cached_image`'s early `image_cache.end()` return distinguishes the first from
the other two.

**Why this generalises.** The same shape appears at three independent sites, which is what makes it
architectural rather than a bug: `rtt-snapshot` (34.7 GiB, fixed for the exact-size case earlier
today and still the second largest), `detile` (26.2 GiB), and compute buffer bindings, every one of
which reports `cache=ineligible total-watch-chunks=0` -- no write-watch at all, so a full compare
and a full upload on every dispatch. prosper copies guest data host-side without dirty tracking, in
several places, and each site was found separately at full cost. That is the thing to fix once.

### The bottom of the chain, 2026-09-26: a GPU→CPU→GPU round trip, and it is a LATCH

The step above ended on a question about cache admission. That question was the wrong one, and the
answer is better: **nothing here is a cache miss.** The copies are the CPU leg of a round trip
between two compute dispatches that run on the same device.

#### The mechanism, traced end to end

At `live_compute.cpp:9079` the guest pointer `src` is set to `nullptr` whenever `renderer_owned`,
and the copy at `:9249` reads `live_target.pixels` instead — the renderer's **CPU-side render-target
snapshot**. That snapshot is published by `CpuRttSnapshotPool::copy` at `:12827` from
`layout_source`, which is the **mapped staging buffer holding a compute dispatch's own result**,
read off the GPU at `:12678`. So:

```
dispatch N    (GPU)  writes a storage image
                     -> vkCmdCopyImageToBuffer into host-visible staging      GPU -> CPU
                     -> map, pack/detile, tile into the guest mirror          (legitimate writeback)
                     -> publish a CPU render-target snapshot                  [rtt-snapshot]
dispatch N+1  (GPU)  binds the same address; renderer-owned, no GPU seed
                     -> memcpy out of that CPU snapshot into staging          [storage-materialize]
                     -> vkCmdCopyBufferToImage                                CPU -> GPU
```

The two largest host-copy categories are the two legs of that trip. The cache gate's
`!renderer_owned` term is **correct** and should not be widened: the guest range is not the
authority for these bytes, so neither the guest write-watch nor the submit journal can validate
them.

#### Why it never recovers

Three gates, each measured at the site where it fires, on a 74 s Astro Bot run (shipped windowed
frontend, GPU present confirmed adopted, idle box):

```
[gpu-seed-refused]         cpu-only-authority=28051.9MiB/7885c   view-ineligible=944.8MiB/1184c
[rtt-destination]          candidates=11660 borrowed=3363 recorded=3363 published=3362 failed=0
[rtt-destination-refused]  no-persistent-image=24291.1MiB/6103c  no-rtt-entry=4504.7MiB/750c
                           extent-mismatch=481.2MiB/10c          format-mismatch=152.3MiB/6c
```

1. **`cpu-only-authority` — 87% of the renderer-owned copies, 97% of their bytes.** The GPU seed
   import (`live_renderer.cpp:1948`) requires `live_rtt_gpu_importable`, which requires
   `surface.gpu_valid`. These targets have a correctly-sized CPU snapshot and no valid GPU image.
2. **`gpu_valid` is cleared by the publication itself.** `live_renderer.cpp:2100`: when a compute
   result is published with `linear_pixels`, the notifier sets `published.gpu_valid = false`. Its
   own comment says why — *"A linear snapshot means no device mirror received this result"*.
3. **The only path back is refused.** The mirror-back that sets `gpu_valid = true` (`:2120`)
   requires `bi.mirror_result_to_imported`, which is set only alongside `bi.seed_from_imported`
   (`live_compute.cpp:8060`) — i.e. only if the dispatch imported the image in the first place,
   which step 1 just refused. The destination borrow at `:11140` is the escape hatch, and it is
   refused for **8,297 of 11,660 candidates**, overwhelmingly with `no-persistent-image`.

**So it is a latch.** One transition to CPU-only authority is permanent for that target: no
dispatch can import it, so no dispatch can mirror back, so it never regains GPU authority, and
every subsequent dispatch pays a full round trip. That is consistent with Astro Bot running at
~5 fps from the very first splash screen rather than degrading into it.

#### The root, and it is one predicate

`no-persistent-image` is `find_persistent_color_target(addr, w, h, format, /*require_valid=*/false)`
returning null, or an entry whose layout is `VK_IMAGE_LAYOUT_UNDEFINED`. The cache
(`tests/fixtures/render_runner.h:3070`) is keyed on the exact 5-tuple
`{id, width, height, format, volume_depth}` and bounded by
`persistent_color_target_limit()` — **4096 MiB on this machine** (heap/4, clamped;
`[render] persistent color-target residency budget = 4096 MiB (device-local heap 44197 MiB)`).

#### Every decline on this path used to be a bare `return false`

This is the reason the chain took a day rather than an hour, and it is the generalisable part. The
importer had twelve declines and the destination borrower six, all spelled `return false`, so a
consumer could report only *"the renderer would not hand it over"* — while its fallback for any of
them is a full CPU round trip of the image, per dispatch. The `[rtt-destination]` counters existed
but printed only under an active F8 perf capture, every sixteenth candidate, and the census's
`failed` counter read **0** while 8,297 candidates were being dropped by a `continue` placed before
it. A zero that means "not counted" is indistinguishable from a zero that means "did not happen".

Both paths now carry a `LiveTargetImageImport::Refusal` on the out-param they already fill, and
both report at exit by default.

#### It is not capacity — and the obvious A/B could not have told us

The natural experiment is `PROSPER_BACKEND_TARGET_CACHE_MB`. It was run, interleaved, two rounds,
and it is **VOID rather than negative**: the only line prosper prints about this budget comes from
`init_persistent_color_target_device_budget`, which never consults the variable, so it read
`4096 MiB` in both arms. The arms' numbers were also indistinguishable — `no-persistent-image` of
5,758 / 5,585 / 5,499 / 5,344 in run order A, B, A, B, monotone in *time* and unmoved by *arm* —
but with no way to show the lever moved, that is not evidence.

Residency answers it directly, with no lever at all:

```
[persistent-targets] PEAK residency 97 of 256 entries, 1604.1 of 4096.0 MiB (39.2% of budget)
                     eviction-attempts=0 evicted=0 (0.0 MiB)
```

**Peaked far below both bounds and evicted nothing.** Read the peaks, not the eviction counter:
the admission paths skip the eviction loop entirely while a submission batch is pending
(`avoid_cache_eviction` / `eviction_deferred`), so a target *can* be refused for capacity with
every eviction counter still reading zero (#3872 review). And the gate has **two** bounds — a byte
budget and a 256-entry count — either of which refuses a target on its own, so clearing one proves
nothing. At a high-water mark of 39.2% of the byte budget and 97 of 256 entries, neither could
have bound.

So `no-persistent-image` means the entry for `{addr, width, height, format, volume_depth}` was
**never created** — no graphics pass ever made a persistent device image at that key, and the
compute path does not make one either.

#### The fix this implies

> When a compute dispatch publishes a result for a render-target address with no persistent device
> image, **create one and write the result into it**, instead of publishing CPU pixels and latching
> the target to CPU-only authority.

That converts the publication at `live_renderer.cpp:2090` from the `linear_pixels` branch
(`gpu_valid = false`) to the mirrored branch (`gpu_valid = true`), which unlatches the target: the
next dispatch's seed import succeeds, it keeps the mirror-back obligation, and both legs of the
round trip disappear for every later dispatch on that address. The budget and eviction policy it
would allocate into already exist and are 62% idle.

**Why it compounds.** Nothing above is title-specific — no format, no extent, no program hash, no
title id. It is one predicate on the publication path that every title's compute stage crosses.
The same architecture is already recorded independently on two other titles by two other lanes:
#3683 measures the producer leg on *Sonic Frontiers* (6.5 s of `retile_copy` plus 8.0 s of
write-watch on one route) and #3450 measures the consumer leg on *Grand Theft Auto V* (88.5% of
per-dispatch GPU time in prosper's own storage copy, crossing PCIe on a discrete GPU). Neither had
the middle of the chain, which is why each read as a separate cost.

**Independent corroboration of #3157 from a third title**: the compute-items census reports
`calls=30276 items=30276` — exactly one item per call over 30,276 calls, so the batching that
issue's correction says does not happen, measurably does not happen here either.

#### After #3914: a read-only source alias still blocks publication

#3914 implemented the compute-created destination described above. In Astro Bot's opening route,
the remaining CPU RTT snapshots were counted by producer: 4,653 publications and 37.85 GiB of
linear pixels over 82 seconds. One 3840×2160 RGBA16F producer accounted for 24.23 GiB, and a
recurring 1920×1080 RGBA16F producer for 5.99 GiB. These are copied-byte counts, not elapsed
critical-path time or an FPS estimate.

The exact destination probe found the 4K and 1080p storage results eligible and their renderer
images borrowable. Publication was then refused because each dispatch also had a read-only sampled
binding to that same renderer image. Source pixels are sampled before the completed result copy,
but the existing collision rule treats any other imported binding as a lifetime conflict. The
sampled-first RGBA16F route already has a separate GPU mirror; the costly route prepares the
storage output first, then imports the sampled binding. The narrow admission allows an exact
read-only imported alias after an independently pinned self-seed. The fixture checks that the
renderer receives *changed* output pixels in both binding orders, and focused strict synchronization
validation reports no new hazard. Two same-binary opening-route pairs reduced CPU RTT snapshot
volume per compute item from 1.183–1.195 MiB to 0.371 MiB. Selected dispatch timings varied in
both directions, so that is a copied-byte result, not a supported latency or FPS gain.
`PROSPER_COMPUTE_DEST_TRACE_ADDR` traces one or more `0x` guest addresses through this destination
decision without enabling the RTT debug mode that disables GPU residency.
