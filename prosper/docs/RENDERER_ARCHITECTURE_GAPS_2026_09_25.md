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

## Ruled out

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
