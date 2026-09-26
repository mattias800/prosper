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

| title | readback slots | renderer share of wall clock | rate |
|---|---|---:|---|
| The Messenger | **1 of 36,897** (100.0% not-wanted) | 6.7% | 59.9 fps, 8,880 frames |
| Astro Bot | 293 of 19,233 (98.5% not-wanted), 8.9 GiB | 9.4% | single-digit |

So readback is **not** the render path's dominant cost, the submit does **not** block because of it,
and § 2's timeline-semaphore idea is neither blocked by it nor rescued by it. What remains true is
narrower and still worth knowing: Astro Bot really does copy 8.9 GiB back under `no-color-target`,
which is a caller-shape question, not a residency one.

**The general lesson is the expensive half.** A harness artifact was ruled out by comparing two
harnesses, and both had the same artifact. *Agreement between instruments is not independence.*
Check the mechanism directly — one `grep` for the log line that says the path activated — rather
than inferring it from two measurements that agree.

**What this does NOT withdraw:** the CPU RTT snapshot-pool fix, which was re-measured on the
windowed path and stands. Astro Bot performs **24,991** pool copies there at a **99.5%** hit rate
with 113 misses; before the capacity-matching change the same workload missed 61.5%. The pool is
genuinely exercised on the shipped path, so that fix is real.

### Which readbacks are avoidable — measured, and it is not the obvious one

The readback finding above raises one question: which of `readback_policy.hpp`'s three reasons
dominates. They have different fixes, and only a per-reason count can choose between them. Measured
2026-09-26 on colour slot 0, ~120 s each:

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

**FALSIFIED: "small render passes are what make heavy titles slow."** This document's own § 1-2
motivated a pass-batching change. The pass-cost census measured it instead: Astro Bot spends
**7.8% of its wall clock inside `render_draw_pass_rgba` at all**, so collapsing passes could not
have moved it whatever the pass count. The Messenger, which runs well, spends 63.8%. Pass length is
real (Astro Bot's passes hold a mean of 1.00 draws, 100% single-draw) and it is not this title's
frontier. Do not restart pass batching without first showing the renderer is a large share of the
target title's wall clock.

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
