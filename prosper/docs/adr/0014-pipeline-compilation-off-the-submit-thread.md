---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0014: Compile pipelines off the submit thread, from a persistent cache

## Context

`PERF-P3` forbids shader or pipeline compilation on the submit thread after warm-up, and the
`shader-compile` alarm measures compile time as a share of the frame budget. What exists: one
device-lifetime `VkPipelineCache` shared by graphics pipeline creation, with opt-in disk persistence
in a versioned, checksummed envelope (`docs/gpu/GRAPHICS_PIPELINE_CACHE.md`, #3378). Loading stays
opt-in because repeated *Grand Theft Auto V* runs crashed in NVIDIA's cache-load path and the cause
is unresolved. Pipelines are still created on demand on the path that needs them.

vkd3d-proton and DXVK avoid in-frame compilation with background compile threads, persistent state
caches, and `VK_EXT_graphics_pipeline_library` to link pre-compiled stages quickly when a full
pipeline is missing. Structural references only.

## Decision

1. A pipeline the submit needs and the cache lacks is compiled on a worker. Whether the submit
   waits for it or proceeds is decided per case by what the guest observes: a draw whose output the
   guest reads must not be dropped (`FAIL-1`), so the default is to wait, counted by the alarm.
2. Where `VK_EXT_graphics_pipeline_library` is available, stages are compiled once into libraries
   and linked per state combination, which turns most misses into fast links.
3. Disk persistence becomes the default once the NVIDIA cache-load crash is understood. Until then
   it stays opt-in, and this ADR records that dependency rather than overriding it.
4. A recorded set of pipeline keys per title can pre-warm the cache at startup, keys only, never
   shader bytes from a dump.

Adds spec rule `PERF-P7`.

## Consequences

Second and later launches stop compiling in-frame, and first-launch compiles move off the critical
path where the guest does not observe the result. The NVIDIA crash is a blocker for point 3, not
for points 1, 2 and 4. `CONFIDENCE: MED` on how often a missing pipeline can proceed without waiting
-- most guest draws feed something the guest eventually presents.

## Alternatives considered

- Make disk loading the default now: rejected while a driver crash on load is unexplained.
- Skip draws whose pipeline is not ready (as some emulators do): rejected, it drops guest content,
  which `FAIL-1` forbids.

## Approval

Requires the project owner's acceptance.
