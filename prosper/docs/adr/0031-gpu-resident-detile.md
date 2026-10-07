---
kind: adr
status: proposed
date: 2026-10-07
---

# ADR 0031: Keep guest textures GPU-resident: detile, retile and format-convert on the GPU

## Context

prosper reads a guest texture by copying its tiled bytes out of guest memory, rearranging them into a
linear layout on a CPU thread, and uploading the result. A compute shader's storage-image result takes
the reverse trip: linear on the GPU, retiled for the guest. Both trips exist because the guest's tiled
memory is the source of truth. The problem is that they run on the host, and they run again whenever
the cache in front of them cannot prove that nothing changed.

**Measured, 2026-10-07.** *Assassin's Creed Black Flag Resynced* (`PPSA28183`), Windows, RTX 4070
SUPER, default launch. The always-on alarm reported:

```
[perf-alarm] rule=host-copy-per-flip value=23.73 MiB/flip ...
  sites=detile:15.8MiB/flip@3.16MiB/call(130 calls), storage-materialize:7.9MiB/flip@7.91MiB/call(26 calls)
[transfer-pressure] HIGH host-copy 584 MiB/s ... detile=1471.1MiB
```

Two thirds of the per-flip host copying is detile, and the remaining third is storage
materialization. This is not specific to one title: the 2026-09-26 gap analysis puts `detile` at
26.2 GiB on another route and names the shared shape, a host copy with no dirty tracking
(`docs/gpu/RENDERER_ARCHITECTURE_GAPS_2026_09_25.md:684-689`).

**Where the bytes are spent.**

- Every host detile goes through `detile_surface` (`src/gpu/texture/tile.cpp:1464`). It charges
  `Transfer::Detile` (`:1467`) and feeds the `[tile-census]` diagnostic (`:1466`, keyed by op, size,
  bytes per element and tile mode, `:1439-1461`). The alarm's `detile` site is exactly that call.
- Storage images bound to a compute dispatch are re-staged from guest memory for each dispatch and
  charged to `Transfer::StorageMaterialize`
  (`frontends/shared/live/live_compute.cpp:10080-10083`). The comment at `:10084-10088` records why
  this is mostly waste: nothing there has a write-watch, so a repeat copy of an unchanged source
  cannot be told apart from a needed one. When the binding is detiled directly into staging
  (`PROSPER_NO_DIRECT_STORAGE_DETILE` off, `:9892-9996`), the bytes are charged as detile instead.
- `[gpu-seed-refused]` (`live_compute.cpp:10060`) names why renderer-owned bindings took this CPU
  round trip.

**What already runs on the GPU, and why these surfaces do not use it.**

- *Sampled detile.* `build_compute_detile_float16` (`src/gpu/recompiler/spirv_builder.cpp:281`) is
  shipped and tested (`tests/gpu/execute/test_gpu_detile.cpp`, `test_gpu_detile_storage.cpp`), and it
  matches the CPU address walk including padding and pipe XOR. It is admitted only by
  `gpu_detile_shape` (`frontends/shared/live/submit_renderer/image_resources.cpp:3118-3136`): format
  `Float16` with 2 or 4 components, `tile_mode == 27`, one mip, one sample, no compression, not in a
  mip tail, not a depth-compare view, and for 2D only when the texture is not eligible for the
  persistent decode cache (`:3116-3117`). Every other format and tile mode falls to the host.
- *Storage retile.* The GPU retile of storage writebacks is on by default
  (`live_compute.cpp:11066-11086`, `tests/gpu/execute/test_gpu_retile.cpp` with about thirty ctest
  arms). Its admission declines for twelve named reasons, recorded by `GpuRetileDecline`
  (`frontends/shared/compute/gpu_retile_census.hpp:27-43`) at the point of decision
  (`live_compute.cpp:11092-11109`): aliased, imported, partial write, inexact bytes, mip tail or
  offset, no resource, no staging, unsupported shape, packed extension off or unsupported, layout
  mismatch, prepare failed. A declined image takes the CPU `tile_surface` path.

So the GPU kernels exist, are bit-checked against the host, and cover one format family for reads
and most shapes for writes. The default for everything else is the host. The rule this violates is
`PERF-P5`: memory that tracking can prove unchanged is neither compared nor copied in full. The cache
half of the problem belongs to `PERF-P9` (ADR 0010) and to the page-tracking ADR drafted in parallel
as 0032; this ADR is the transform half.

## Decision

Spec rule `PERF-P12` (`docs/spec/performance.md`), `Status: proposed (adr:0031)`:

1. **The GPU is the default transform.** Every tile mode and format that prosper detiles, retiles or
   format-converts on the host today gets a GPU path, driven by the same tiling tables the host
   path uses (`src/gpu/texture/tile.*`), so there is one description of each layout and two
   executors of it. A new tile mode or format lands with both.
2. **The host path stays, as the reference.** It is not deleted. Each GPU kernel is tested
   bit-for-bit against the host function on the same input, with padding, pipe XOR, mip tails and
   partial tiles exercised, in the style of `test_gpu_detile.cpp` and `test_gpu_retile.cpp`. The host
   path remains the runtime fallback when a GPU prepare fails, and that fallback is counted, never
   silent.
3. **Storage images stay resident across dispatches.** A storage image's GPU copy is the current
   version until the guest CPU writes the pages beneath it. Re-staging per dispatch happens only on
   evidence of such a write, as reported by page tracking (ADR 0032, drafted in parallel). Until that
   tracking exists, the per-dispatch re-stage is a counted fallback, as `PERF-P9` already says of
   byte comparison.
4. **Declines are named.** Each remaining host-path admission decline is counted by reason, as
   `GpuRetileDecline` does for writebacks today, and the sampled-detile admission at
   `image_resources.cpp:3118-3136` gets the same census. A host transform is never the unexplained
   default.
5. **Measured, not assumed.** Each migration step is a same-binary A/B on the reference workloads,
   following `.claude/skills/perf-change/`, with the change switched off as the control arm. The
   instrument is the `host-copy-per-flip` alarm and the `[transfer-pressure]` per-category totals,
   with the frontend, present mode and route named in the claim. A step that moves the bytes and
   not the frame time is still recorded, since the bytes are what this rule constrains.

## Consequences

- **Easier:** host CPU time and memory bandwidth spent on layout transforms fall by up to the
  measured share (on Black Flag, 15.8 of 23.7 MiB/flip is detile). The transform overlaps other GPU
  work instead of blocking a submit thread on a CPU pass, which helps `PERF-P6`.
- **Harder:** each tile mode now has two implementations that must agree. The bit-for-bit tests
  carry that burden. The GPU path also has failure modes the host path does not (allocation loss,
  prepare failure), and the retile suite already covers them (`gpu_retile_allocation_fallback`,
  `gpu_retile_mapping_loss` and the others in `prosper/CMakeLists.txt:6211-6258`).
- **Ordering:** the CPU-visible copy stops existing for GPU-transformed surfaces. Any diagnostic
  that reads the decoded host buffer (`PROSPER_DUMP_RAWTILE`, `PROSPER_SLICEMAP`; both force the host
  path today at `image_resources.cpp:3132-3133`) keeps forcing it. That makes a diagnostic run
  differ from a default run, which the instrument-trap rules already require a run to state.
- **Enforcement:** `runtime:host-copy-per-flip` and `runtime:host-copy-pressure` flag the regression;
  the decline censuses name the cause; the bit-for-bit tests block a divergent kernel. No ratchet
  rule fits, because a host transform is a legitimate fallback and cannot be counted statically.

## Alternatives considered

- **Faster host detile (SIMD, more threads).** This makes each call cheaper and leaves the bytes
  where they are. The upload after the detile still crosses PCIe at full size, the CPU still has to
  touch every byte, and the threads compete with the guest's own threads on the same cores. It
  shrinks the constant and keeps the shape that `PERF-P5` forbids. It remains worthwhile for the
  reference path and the fallback, but not as the default.
- **Cache detiled copies with better dirty tracking, and keep the host transform.** This is
  necessary, and it is ADR 0010 and the parallel ADR 0032. It is not sufficient: a surface the guest
  really does rewrite every frame (video planes, streamed atlases, compute outputs that a later
  dispatch reads) still pays a full host transform for every change. Page tracking decides *whether*
  to transform, and this ADR decides *where*. The two compose, and neither replaces the other.
- **Sample the tiled layout directly from the shader (recompiler emits address swizzles).** This
  avoids the transform but puts per-tile-mode address arithmetic into every sampling shader and
  loses hardware filtering on tiled data. It is a much larger change to the recompiler with no
  measurement behind it. It is not rejected outright; it is out of scope here.

## Migration order

By measured bytes, largest first, as the census reports them:

1. Run `[tile-census]` and `[transfer-pressure]` on each reference workload and on Black Flag.
   Rank the (tile mode, bytes per element, format) keys by bytes per flip. That ranking is the queue.
   This ADR makes no assumption about which key leads, since none has been ranked yet.
2. Widen sampled GPU detile from `Float16`/`tile_mode 27` to the leading keys, one key per PR, each
   with its bit-for-bit arm and its A/B.
3. Lift the 2D persistent-cache exclusion (`image_resources.cpp:3116-3119`): retain the GPU-detiled
   image in the cache instead of declining the GPU path because the cache is CPU-decode-shaped.
4. Work down the `GpuRetileDecline` reasons in order of declined bytes.
5. Make storage images resident across dispatches once ADR 0032's page tracking can report a guest
   CPU write. This removes most of the `storage-materialize` site.

## Open questions

- Does Black Flag's 3.16 MiB/call detile come from a few re-decoded surfaces (a cache problem) or
  many distinct ones (a transform problem)? `PROSPER_DETILE_STATS` (`image_resources.cpp:2909-3104`)
  answers this and should be run before step 2. CONFIDENCE: LOW on the split.
- Format conversion that is not pure layout (scalar narrowing, sRGB, BCn re-encode) may not match
  bit-for-bit across GPU vendors. Is "bit-for-bit against the host" the right contract for those, or
  is a stated tolerance needed?
- How does a GPU-resident image interact with the guest reading its memory back (the
  GPU→CPU→GPU latch in the gap analysis, `RENDERER_ARCHITECTURE_GAPS_2026_09_25.md:691`)?

## Approval

The project owner accepts or rejects this ADR. Acceptance moves `PERF-P12` to `accepted` and
unblocks migration step 2 onward as default-on changes. Step 1, the census, needs no approval.
