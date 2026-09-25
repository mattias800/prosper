# Renderer architecture gaps (2026-09-25)

This is a structural review of prosper's render submission and resource-residency architecture,
written after a heavy-3D performance question ("why is a 3D title at single-digit fps when the
scene itself is not large?"). It is deliberately *not* another leaf-level profile: #3770 and
`PERFORMANCE_ROADMAP_HANDOFF_2026_09_23.md` already establish that individual leaves are being
shaved for single-digit-percent movements, and two of those attempts measured *backwards*. The
claim here is that the ceiling is set by four architectural properties, not by the leaves.

Every prosper-side number below comes from prosper's own code comments, its own profiler output, or
its own status docs. Nothing here is copied from any other implementation; the designs named are
standard real-time-Vulkan practice (timeline semaphores, write-protect page tracking,
address-keyed resource caches), not anyone's proprietary work.

## 1. The CPU blocks on the GPU at every submission — no timeline semaphore exists

`tests/fixtures/render_runner.h:2609` `submit_and_wait()` is the only submission path:

```
vkCreateFence(...)                     // a fresh fence object per submission
render_locked_queue_submit(...)        // vkQueueSubmit
vkWaitForFences(..., 5s)               // CPU blocks until the GPU drains
  -> on timeout, render_locked_queue_wait_idle(queue)
```

prosper's own comment at `render_runner.h:1305` records the frequency:

> A Blue Prince gameplay submit performs 15.68 queue submits and 15.68 fence waits — sixteen
> CPU<->GPU round trips per rendered frame

That is a full pipeline drain sixteen times a frame. Between each, the GPU is idle while the CPU
records, and the CPU is idle while the GPU executes; the two never overlap. It also allocates and
destroys a `VkFence` per submission.

**`grep -rn 'timelineSemaphore\|VkSemaphoreTypeCreateInfo' src/ tests/fixtures/ frontends/`
returns nothing.** prosper has no timeline semaphore anywhere. Since #3418 adopted Vulkan 1.4,
`VkSemaphoreType​Timeline` has been core-available (1.2+) and free to use.

The shape that removes the stall: one device-wide timeline semaphore; `Submit()` takes the next
monotonic tick, signals it, passes a **null fence**, and returns the tick without waiting.
Completion becomes a non-blocking `known_gpu_tick >= tick` query. Resource lifetime stops being
"the fence we blocked on" and becomes "retire everything whose tick has passed", drained
opportunistically. A real CPU wait then happens only where a consumer genuinely needs bytes back
(readback, storage writeback, a guest `WAIT_REG_MEM`) — which `flush_now` already classifies at
`render_runner.h:12048`, so the three reasons are already attributed and countable.

CONFIDENCE: HIGH — the submission path and the absence of timeline semaphores are both direct
reads of the tree, and the 15.68 figure is prosper's own recorded measurement.

## 2. "Has this buffer changed?" is answered by reading every byte

`ResidentRenderBufferCache::find` (`render_runner.h:4398`) decides residency reuse with:

```
if (watched || std::memcmp(entry->owner->snapshot.get(), source, key.bytes) == 0)
```

So the **hit** path reads `2 x bytes` of memory bandwidth (the CPU-side snapshot plus the guest
source) to conclude nothing changed. The miss path then does two further full copies — into the
snapshot and into the mapped arena (`:4414`–`:4415`). prosper also keeps a second full CPU copy of
every resident buffer purely to compare against.

A write-watch exists (`src/host/memory/guest_write_watch.hpp`) and *is* wired into this cache, but
it is gated into near-irrelevance (`render_runner.h:4333`–`4365`):

- it requires **two full equality validations** before it is even armed;
- **two** dirty queries permanently disable it for that entry (`watch_disabled = true`);
- a `Unknown` query result disables it permanently as well.

So exactly the buffers that change occasionally — the common case — fall back to unconditional
`memcmp` forever.

The 2026-09-23 GTA profile is consistent with this being the dominant cost, and it is prosper's own
profile:

| Symbol / population | Measured |
|---|---|
| `__memmove_avx512_unaligned_erms` | 11.63% self |
| `__memcmp_evex_movbe` | 4.51% self |
| renderer exact source validation | 91,021 calls, **1.63 GB** in 30 s |
| renderer upload batches | 40,009 calls, **2.84 GB** in 30 s |

The industry answer is to never read clean bytes at all: write-protect the guest pages backing a
cached buffer, take the fault, mark the page dirty, and upload only the dirty sub-ranges as a list
of copy regions. A clean buffer then costs **zero** bytes of bandwidth to validate, instead of
`2 x bytes`. prosper already owns the hard half of this — `guest_write_watch.cpp` plus the fault
handler in `exec_image_linux.cpp` — so the work is changing the *policy* (page-granular, permanent,
upload-the-dirty-range) rather than building new machinery.

CONFIDENCE: HIGH for the mechanism and the gating; MED for attributing the profile shares
specifically to it, since those shares are sampled CPU on one TID and `memmove` has other callers.

## 3. No GPU-resident buffer keyed by guest address, so residency is opt-in and narrow

Residency today is keyed by `{identity, bytes}` (`render_runner.h:4131`) and is reachable only
through a chain of preconditions: `readonly_buffer_pass` requires that *every draw in the pass* has
only read-only buffers (`:9229`–`:9236`), `bytes >= 4096`, a planned range group, and a configured owner
limit. Anything that fails a clause falls back to copying into a per-pass arena
(`parallel_render_memcpy_batch`, `:11959`), which is a parallelised memcpy rather than an avoided
one.

The structural alternative is a single buffer cache keyed by **guest address range** with a
page-table lookup, holding persistent device-local buffers, an LRU and a memory budget, with
GPU-modified ranges tracked so a GPU-written buffer is read back only when the CPU actually reads
it. That makes residency the default rather than an opt-in fast path, and it is what makes point 2
cheap to apply uniformly.

CONFIDENCE: MED — the preconditions are read directly from the source; the projected benefit is an
inference from the same profile as point 2.

## 4. Submission granularity is per-pass, not per-N-draws

`flush_now` (`:12048`) submits at pass boundaries and whenever a readback or storage writeback is
requested. With the blocking submit of point 1, a pass boundary is a drain. Once submission is
non-blocking, the natural policy is the opposite: keep one command buffer open across passes and
flush on a **draw count** (plus the genuine sync points), so long draw chains never sit recorded but
unsubmitted while the GPU idles, and short passes never each pay a submit.

CONFIDENCE: MED — the mechanism is a direct read; the tuning constant needs measuring here.

## 5. Smaller: per-draw resource resolution is interpreted

The same profile shows `resolve_dynamic_fetch` 4.69%, `find_spirv_descriptor_binding` 1.91%, and
2,485,440 shader-resource-key equality calls. These are per-draw re-derivations of something that
is fixed per *shader*. Compiling each shader's resource-resolution plan once into a flat, ordered
op list and replaying it per draw is the standard fix. This is listed last deliberately: it is
worth real single-digit percentages, and points 1–3 are worth multiples.

CONFIDENCE: LOW on the projected size — the population is measured, the redundancy is inferred.

## What this predicts, and how to falsify it

If this analysis is right, the leaf-shaving results in #3770 are expected rather than surprising:
descriptor-set reuse measured *backwards* because the frame is not descriptor-bound, it is bound by
sixteen pipeline drains and by reading guest memory twice to decide it had not changed.

The cheapest discriminator is point 1 in isolation, because it needs no policy change anywhere
else: keep every existing copy and compare, replace only `submit_and_wait`'s fence-and-block with a
timeline tick plus deferred retirement, and measure the same GTA Performance Story route. If the
frame time does not move, this analysis is wrong at the top and points 2–3 should be re-argued
before anyone builds a page tracker.

A positive control is mandatory and must not be drawn from the same route (see the charter's
same-source control rule): a title already known to be GPU-bound rather than submit-bound should
*not* improve, and if everything improves uniformly the instrument is measuring itself.

## Ruled out

- **"The recompiler or shader quality is the frontier."** Not for this class of cost: the profile's
  top self shares are `memmove`, `malloc` and `memcmp`, none of which are shader work, and the
  per-draw GPU device time is already a small fraction of the submit/fence wall-clock interval
  (`gpu_device_ms` is documented at `render_runner.h:1325` as a subset of that interval).
- **"Descriptor-set reuse is the win."** Falsified by measurement in #3770: 22.886% exact
  pass-local repeats, and both the targeted leaf and the enclosing renderer moved in the wrong
  direction. Do not revive it from repeat counts alone.
