# Evergate renderer-performance handoff (2026-07-19)

This document is the native-Linux continuation point for Evergate (`PPSA01885`). It records the
state after the Windows performance pass through PR #1052, including what is landed, what was
measured, which experiments were rejected, and which jobs remain intentionally unfinished.

Use this document with:

- [`RENDERER_PERFORMANCE_2026_07.md`](RENDERER_PERFORMANCE_2026_07.md) for the full renderer history;
- [`FRONTEND_APP.md`](FRONTEND_APP.md) for app and environment-variable contracts;
- [`../scripts/evergate/README.md`](../scripts/evergate/README.md) for the reviewed progression and
  snapshot routes;
- [`VERIFICATION.md`](VERIFICATION.md) and `tools/snapshot/AGENTS.md` for merge verification.

## Executive status

Evergate renders correctly at native resolution on both Windows and Linux. The title/menu is much
faster than the original bring-up and can approach roughly 20 FPS on the measured Windows host.
The draw-heavy new-game/opening transition is still only about 6 FPS. The original performance
objective—representative play at roughly 20-30 FPS—is **not complete**.

The final Windows tranche, PR #1052 (`d5a5cddb08d0e02f17ee369a88c2c09d23ab684d`), stores a shader
content hash once in the immutable analysis object instead of re-hashing every shader dword during
every warm compiled-cache lookup. Its measured dense-transition result was:

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Executor submit | 130.35 ms | 115.69 ms | -11.2% |
| Draw-realization wall | 39.43 ms | 36.11 ms | -8.4% |
| Shader lookup, worker-summed | 90.02 ms | 71.56 ms | -20.5% |
| Coarse app rate in the dense region | about 5.7-6.0 FPS | about 6.3-6.6 FPS | bounded gain |

This is a real improvement, not a claim that the overall objective has been reached. Resume from
remote `master`; do not revive the old performance branches.

## Current jobs and ownership state

All code tranches from the Windows pass are either merged or explicitly rejected. No implementation
branch is intentionally left active. The remaining work is captured here so native-Linux development
can claim one focused issue/PR at a time.

| Priority | Job | State | Acceptance evidence |
|---:|---|---|---|
| 0 | Establish a native-Linux Evergate baseline | Not started | Fresh-save realtime route reaches the same dense opening; matched 25-submit windows and app FPS recorded |
| 1 | Reuse one exact shader analysis throughout a draw | Proposed | Fewer exact analysis validations and lower shader-lookup/realization time with mutation tests and all snapshots green |
| 2 | Cache immutable AGC resource-layout plans | Proposed | Metadata walk time falls while current SGPR/EUD descriptor values and metadata invalidation remain exact |
| 3 | Replace repeated sparse user-SGPR lookups | Proposed | Lower metadata/fold time with unchanged resource tables and shader outputs |
| 4 | Find exact Linux texture-write evidence | Research | Cross-submit validation bytes fall without hash/TTL/fault-delivery shortcuts |
| 5 | Reduce backend submission/fence/readback boundaries | Architectural | Fewer waits/readbacks with dependency-graph evidence and byte-identical render tests |
| Separate | Intermittent loading stall (#712) | Open risk | Diagnose from submit/suspend logs; do not blame renderer caches without matching evidence |

Do not combine these jobs into one broad renderer PR. Each changes a different correctness contract
and needs its own before/after evidence.

## Known-good checkpoint and performance route

The representative performance checkpoint is the draw-heavy transition after selecting **New Game**
from the main menu. Title-idle or logo samples are not representative. A correct run progresses
through the title/save flow into the opening scene and produces roughly 470-490 realized draws per
submit in the dense window.

Start with the committed realtime route:

```text
scripts/evergate/reach-first-gameplay-realtime.pad
```

It is anchored to successful pad-read counts, so slow synchronous rendering cannot cause a complete
press/release pulse to pass between polls. Snapshot acceleration uses a different wall-time route;
do not mix the two.

On a faster machine, the guest can consume the original route before the wall-time-driven logo/title
transitions finish. If that happens, create a local-only extended route by retaining all committed
edges and appending these Cross pulses:

```text
p190-195:cross
p205-210:cross
p220-225:cross
p235-240:cross
p250-255:cross
p265-270:cross
p280-285:cross
p295-300:cross
p310-315:cross
p325-330:cross
p340-345:cross
p355-360:cross
p370-375:cross
p385-390:cross
p400-405:cross
p415-420:cross
p430-435:cross
p445-450:cross
p460-465:cross
p475-480:cross
p490-495:cross
```

The Windows profiling route used the following full sequence. This is included so the checkpoint is
reproducible even if the shorter committed route changes:

```text
p16-21:cross
p31-40:cross
p51-56:cross
p64-69:cross
p77-82:cross
p89-96:cross
p103-108:cross
p117-125:cross
p134-140:cross
p144-148:cross
p156-162:cross
p173-178:cross
p190-195:cross
p205-210:cross
p220-225:cross
p235-240:cross
p250-255:cross
p265-270:cross
p280-285:cross
p295-300:cross
p310-315:cross
p325-330:cross
p340-345:cross
p355-360:cross
p370-375:cross
p385-390:cross
p400-405:cross
p415-420:cross
p430-435:cross
p445-450:cross
p460-465:cross
p475-480:cross
p490-495:cross
```

## Native-Linux bring-up

Use a fresh clone or private worktree and keep game dumps, save data, frame dumps, and profiling logs
outside Git. Configure the normal RelWithDebInfo desktop app using the repository's documented Linux
dependencies:

```sh
cmake -S prosper -B prosper/build-linux \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DPROSPER_APP=ON \
  -DPROSPER_AUDIO_SDL3=ON \
  -DPROSPER_PAD_SDL3=ON
cmake --build prosper/build-linux -j8
ctest --test-dir prosper/build-linux --output-on-failure
```

Run Evergate at native scale and full cadence from `prosper/`:

```sh
PROSPER_GUEST_FS=1 \
PROSPER_GUEST_ARGS=-force-gfx-direct \
PROSPER_RENDER_EVERY=1 \
PROSPER_RENDER_SCALE=1 \
PROSPER_SAVEDATA_DIR=/path/to/fresh-evergate-save \
PROSPER_PAD_SCRIPT=@scripts/evergate/reach-first-gameplay-realtime.pad \
  ./build-linux/prosper-app --dump /path/to/PPSA01885-app0
```

For profiling, add:

```sh
PROSPER_RENDER_TIMING=1
PROSPER_DRAW_REALIZE_THREADS=8
```

Use immediate present if the Linux frontend and compositor support it. Start with eight realization
workers only to compare against the Windows data; after recording a native-Linux baseline, a local
thread-count sweep may find a different optimum.

Before every performance run:

1. Use a fresh save directory.
2. Ensure `PROSPER_RENDER_EVERY_FOR_MS` is unset.
3. Keep `PROSPER_RENDER_SCALE=1` and `PROSPER_RENDER_EVERY=1`.
4. Disable heavy per-target/pixel diagnostics unless they are the subject of the experiment.
5. Confirm the route reaches the opening transition, not merely the title screen.
6. Compare rolling `[render-window]` lines at matched draw counts.
7. Repeat a candidate if backend time moves independently of the targeted phase.

Never carry snapshot-only `PROSPER_RENDER_SCALE=4` or `--render-every 500` settings into an
interactive/performance run. Those deliberately lower resolution or skip guest submits.

## Authoritative Windows baseline

The final A/B host was an Intel Core i7-13700K (16 cores / 24 logical processors) with an NVIDIA RTX
4090. Each run used one RelWithDebInfo app build, a fresh save, native 1920x1080 output, every graphics
submit, immediate present, eight realization workers, and the same read-anchored route.

Three matched 25-submit dense windows around PR #1052 were:

| Draws/submit | Baseline total | Final total | Baseline realization | Final realization | Baseline shader lookup | Final shader lookup |
|---:|---:|---:|---:|---:|---:|---:|
| 488.2 / 488.8 | 133.07 ms | 117.98 ms | 40.15 ms | 36.46 ms | 92.50 worker-ms | 72.15 worker-ms |
| 477.0 / 477.8 | 132.68 ms | 114.89 ms | 40.50 ms | 35.83 ms | 91.33 worker-ms | 71.23 worker-ms |
| 469.3 / 469.2 | 125.31 ms | 114.20 ms | 37.65 ms | 36.05 ms | 86.23 worker-ms | 71.31 worker-ms |

At roughly 478-489 draws/submit after #1052:

| Stage | Approximate cost |
|---|---:|
| Executor total | 115-118 ms/submit |
| Draw-realization wall | 36 ms |
| Executor backend | 79-81 ms |
| Table construction, worker-summed | 133-135 ms |
| Shader lookup, worker-summed | 71-72 ms |
| Table metadata, worker-summed | 64 ms |
| Dynamic fold, worker-summed | 58 ms |
| Table resource finalization, worker-summed | 2 ms |
| Frontend total | 68-70 ms/submit |
| Frontend resource construction | 43 ms/submit |
| Texture work inside construction | 29 ms/submit |
| Buffer/view work inside construction | 6 ms/submit |

Frontend and executor counters describe different pipeline portions. Do not add them naively into a
single frame time. Use app FPS as the outcome, submit-aligned totals for throughput, and the isolated
phase that a candidate is supposed to change for attribution.

## Landed performance work relevant to the frontier

The detailed historical measurements live in `RENDERER_PERFORMANCE_2026_07.md`. The most important
current mechanisms are:

- Exact persistent texture decode caching, including BC/narrow formats and a 1 GiB bounded working set.
- Immutable shader analysis and interpolation-layout reuse after exact byte validation.
- Parallel draw realization, with eight workers the best tested Windows balance.
- Persistent GPU color targets and deferred intermediate scanout readback.
- Persistent host-buffer pools, packed upload arenas, image/view/sampler bindings, pipeline layouts,
  and graphics pipelines under exact bounded keys.
- PR #1036 direct frontend buffer views: about 11.4 ms less resource construction and 8.8% faster
  complete route, while preserving the owned zero-fill fallback.
- PR #1052 cached immutable shader content hashes: 20.5% less worker-summed shader lookup and 11.2%
  less executor submit time in the matched dense sample.

Each of these retains a disable switch or exact test contract documented in the performance overview.

## Rejected experiments: do not repeat blindly

### Cross-submit Windows texture write watch (#1040)

Dense submits still perform roughly 42-45 exact source comparisons over 127-129 MiB, costing around
23-29 ms in texture construction. The observed Windows source mappings are writable physical aliases,
not private/flexible allocations. `MEM_WRITE_WATCH` is unavailable for them. Passive watch found no
eligible mappings; page-protection faults are unsafe because Windows exception delivery can overwrite
the guest SysV red zone; parallel exact compare was neutral or slower.

Hash, sampling, generation, and TTL shortcuts are not correctness-preserving substitutes for write
evidence. Issue #1040 remains open only for an exact mechanism.

### Exact draw/submit realization cache (#1045)

A cache keyed first by `GpuState*`, then by exact program, 96 stage-user registers, vertex count, and
interpolation state, produced about one hit for roughly 940 misses per dense submit. Draw inputs
genuinely vary. The experiment was reverted and the issue closed.

### Shader decode-cache lock scope (#1047)

Moving readability and byte comparison outside the decode-cache mutex preserved tests but made the
Evergate candidate slower (about 113 ms at roughly 488 draws versus about 100 ms in the preceding
control). It was reverted and closed.

### Read-only compute-buffer writeback (#1049)

SPIR-V reflection was extended experimentally to detect buffer writes and skip live-compute
writeback/notifications for proven read-only resources. Tests passed after excluding LDS atomics, but
Evergate showed no overlapping compute writes, no texture-validation reuse gain, and neutral timing.
The experiment was fully reverted and closed.

### Draw-realization thread-count sweep

The measured host has 24 logical CPUs, but 16 workers increased dense realization to about 47 ms and
total executor time to about 148 ms. Four workers reached about 50 ms realization and 151 ms total.
Eight workers remained best. The hot stage has cache/memory contention as well as parallel work.

### Exact resource/materialization caches with weak reuse

Several exact keying attempts—including per-draw state, broad descriptor bundles, index buffers, and
materialized guest buffers—spent more on hashing/ownership than they avoided. Require measured reuse
before retaining another broad exact-key cache.

## Ranked continuation details

### 1. One exact shader analysis per program per draw

The pixel shader is currently analyzed/validated while building its stage table, again for fragment
interpolation layout, and again while constructing the compiled-shader cache key. The vertex shader is
validated during compiled lookup. These operations share the same immutable code version inside one
draw realization.

A promising design is to acquire one immutable analysis handle at the draw boundary and pass it to:

- pixel PC-relative dispatch selection during stage-table construction;
- fragment interpolation-layout lookup;
- vertex/fragment compile-key construction.

Correctness requirements:

- Perform exact same-address mutation validation at the draw boundary.
- Never key solely by guest address or analysis identity across independent validation boundaries.
- Preserve byte-identical cross-address compiled-cache hits.
- Retain embedded PC-relative table tails in the analyzed span.
- Keep full shader-byte equality as the hash-collision authority.
- Prove the regression would fail if a stale analysis were reused.

Measure exact analysis-cache validation counts and shader-lookup time before changing public executor
interfaces. PR #1030 also touched the stage-table/executor area and is already in the base.

### 2. Immutable AGC resource-layout plan

`build_shader_resources` repeatedly validates and walks static AGC header metadata for every stage/draw,
while current SGPR/EUD descriptor values remain dynamic. The dense submit executes roughly 950 stage
table builds. Cache only an immutable layout/plan—classes, slots, offsets, and validated structural
bounds—then apply current register/descriptor values every draw.

Before implementation, establish whether registered AGC metadata can mutate in place. If it can, the
plan needs an exact version/byte identity and invalidation contract. Do not assume code address implies
immutable metadata.

### 3. Faster user-register snapshots

`read_user_sgprs` performs 32 sparse-map probes per block, with multiple blocks per stage. A dense,
immutable per-draw register snapshot or a faster register-file representation could reduce metadata
and fold time. Preserve register write ordering and avoid copying a larger state object per draw unless
the profile proves the trade is favorable.

### 4. Exact texture mutation detection on Linux

Native Linux may expose page/mapping primitives that differ from Windows. First log allocation type,
aliasing, protection, and actual mutation frequency for the 42-45 cross-submit texture candidates.
Then evaluate an exact dirty-page mechanism. The mechanism must account for every guest write path,
GPU/DMA writes, alias, remap, and address reuse before exact comparisons can be skipped.

Do not import the retired Windows protection-fault scheme: its failure was ABI correctness, not merely
platform API availability.

### 5. Backend synchronization and ownership

Dense Windows submits contained about 14 backend callbacks and six queue submissions/fence waits.
Typical isolated costs included 5-7 ms of GPU wait and 2.5-3 ms of readback, while the complete backend
bucket remained about 79-81 ms after #1052. A previous deferred-submit prototype removed waits but
retained complete independent resource graphs and regressed overall time.

Any retry needs one coordinated ownership model for command buffers, descriptor/upload arenas, color
targets, compute dependencies, and final scanout. Use corrected mixed-operation captures/dependency
graphs; do not reorder graphics and compute by type.

## Validation contract for every continuation PR

For executor/renderer changes:

1. Add a focused regression that fails without the change and exercises the claimed invariant.
2. Run `git diff --check`.
3. Build and run the complete Linux ctest suite.
4. Run every snapshot guard:

   ```sh
   cd prosper
   PROSPER_BOOT_TRACE=build-linux/boot_trace \
   PROSPER_SCREENSHOT=build-linux/screenshot \
     python3 tools/snapshot/snapshot.py check
   ```

5. Re-run Evergate from a fresh save at scale 1/full cadence.
6. Compare at least three matched dense 25-submit windows and repeat noisy candidates.
7. Report app FPS honestly; a phase win is not completion of the 20-30 FPS objective.
8. Preserve Messenger, Dead Cells, Blasphemous II, and Evergate snapshot behavior.
9. Use independent review for cache identity, mutation, synchronization, or resource-lifetime changes.

PR #1052's final exact-head verification passed Linux 116/116, Windows 95/95, and all four snapshot
guards. Its focused relocated-code and same-address-mutation tests establish the content-hash contract.

## Completion criteria for the performance objective

Do not close the Evergate performance objective because one title/menu window reaches 20 FPS. Completion
requires a representative native-resolution, full-cadence run that:

- reliably progresses from a fresh save through the title into the opening/gameplay route;
- does not stall at the initial loading checkpoint;
- sustains roughly 20-30 FPS in the draw-heavy opening, not merely on static menus;
- preserves correct rendering and snapshots;
- passes the complete Linux suite and all renderer guards;
- records hardware, present mode, route, save isolation, draw counts, app FPS, and stage timings.

Until that evidence exists, this document and its draft PR are the explicit continuation record.
