---
kind: spec
status: accepted
owner: area:infra
last-verified: 2026-10-05 61557f29
---

# Steady-state frame invariants

The invariants a frame satisfies after warm-up. They are a direction and a review rule as much as a
description: prosper violates `PERF-P1` and `PERF-P5` today, and each rule says where. A PR that
moves away from one says so in its description. The charter (`CLAUDE.md`, *Architecture and
performance ratchets*) names the six and points here; this page is authoritative.

Static rules in the ratchet are proxies: a call-site count cannot tell per-draw from one-time. The
`runtime:` signals are the always-on perf alarms (`src/diagnostics/perf/`, read with
`src/diagnostics/AGENTS.md`); they measure the frame itself but run only where a title runs, which
CI cannot do (no dumps, no GPU). Performance claims follow the same-binary A/B procedure in
`.claude/skills/perf-change/`.

### PERF-P1 -- no in-frame GPU wait or readback the guest does not observe

After warm-up, no CPU wait on the GPU and no GPU-to-CPU readback occurs inside a frame unless the
guest observes the result. A new blocking site names, beside it and in its PR, the guest-visible
result it delivers.
Violated today: the main render submit waits on its fence (`submit_and_wait()` in the Vulkan
backend, which waits in `vkWaitForFences` through `wait_and_finish()`; staged fix #3948).
Status: accepted
Enforcement: ratchet:blocking-sync, runtime:gpu-sync-wait, runtime:surface-readback

### PERF-P2 -- no Vulkan object creation per draw or dispatch

Pools, memory, fences and pipelines are created once and reused; a draw or dispatch looks them up.
Status: accepted
Enforcement: ratchet:vk-object, runtime:pipeline-cache-thrash

### PERF-P3 -- no shader or pipeline compile on the submit thread

A compile that a submit needs is either already cached or performed off the submit thread.
Status: accepted
Enforcement: runtime:shader-compile

### PERF-P4 -- no process-global lock on a hot path

A per-draw or per-submit path takes no lock that every guest thread contends.
Status: accepted
Enforcement: runtime:hle-blocking-wait, review: (lock scope and hotness are not visible to a text scan; docs/performance/THREAD_WAIT_PROFILING.md measures it per run)

### PERF-P5 -- per-draw cost does not grow with guest resource size

Memory that tracking can prove unchanged is neither compared nor copied in full.
Violated today: a resident-buffer hit is re-validated by a full `memcmp` once write-watch disables
itself after repeated dirty queries (#3155).
Status: accepted
Enforcement: runtime:host-copy-pressure, runtime:host-copy-per-flip

### PERF-P6 -- bounded frames in flight, recording overlaps execution

Frame N+1 records while frame N executes on the GPU, with a fixed bound on frames in flight.
Status: accepted
Enforcement: runtime:gpu-sync-wait, review: (overlap is a property of the submit design; it is visible only on a GPU timeline, tools/gpu_timeline)

### PERF-P7 -- a pipeline the cache lacks is compiled off the submit thread

A missing pipeline is compiled on a worker, from libraries where the device supports them, and a
persistent cache makes later launches compile nothing already seen. Whether the submit waits is
decided by what the guest observes; a draw is never dropped for want of a pipeline (`FAIL-1`).
Status: proposed (adr:0014)
Enforcement: runtime:shader-compile, adr:0014

### PERF-P8 -- guest-visible GPU effects retire in stream order

A completion label, EOP event, flip, guest-memory writeback or write-watch invalidation is applied
only after all GPU work and writebacks that precede it in the command stream, and in stream order.
Every other wait the executor performs may be deferred.
Status: proposed (adr:0009)
Enforcement: runtime:gpu-sync-wait, adr:0009

### PERF-P9 -- caches validate against page tracking, not by comparing bytes

A cache asks the guest memory tracker whether any page under its source changed. A full comparison
is a counted fallback where tracking is unavailable, never the normal path.
Status: proposed (adr:0010)
Enforcement: runtime:host-copy-pressure, adr:0010

### PERF-P10 -- a first-use graphics pipeline costs a link, not a compile

On a device with pipeline libraries, shader-stage libraries are built when a guest program is
recompiled, off the submit thread; a draw that meets new fixed-function state fast-links them and is
never dropped, and an optimized pipeline replaces the link in the background.
Status: proposed (adr:0029)
Enforcement: runtime:shader-compile, adr:0029

### PERF-P12 -- guest textures are detiled, retiled and converted on the GPU

Each tile mode and format prosper transforms has a GPU path driven by the shared tiling tables, and
that path is the default. The host transform is the bit-for-bit test reference and a counted
fallback, and a storage image stays resident across dispatches until the guest CPU writes it.
Every host transform is charged to `Transfer::Detile` or a named sibling; until it is, the runtime
instruments below see only the `detile_surface` shapes and cannot flag a regression elsewhere.
Status: proposed (adr:0031)
Enforcement: runtime:host-copy-per-flip, runtime:host-copy-pressure, adr:0031

### PERF-P13 -- write protection is armed over guest memory only where its fault is red-zone safe

The guest memory tracker is portable and each host supplies only the primitive. A host whose write
fault is delivered below the guest thread's RSP (Windows) tracks guest-writable memory with a
fault-free primitive or falls back to a counted compare, never with a page-protection guard.
Status: proposed (adr:0032)
Enforcement: runtime:host-copy-pressure, adr:0032

### PERF-P14 -- the GPU reads guest buffers in place unless a copy is measured cheaper

A buffer a draw or dispatch reads is imported from guest memory where the device accepts the
pointer; a device-local copy is kept only where measured reads per change make it cheaper, and is
invalidated by page tracking. Every remaining CPU copy of a guest buffer is counted.
Status: proposed (adr:0033)
Enforcement: runtime:host-copy-pressure, runtime:host-copy-per-flip, adr:0033

### PERF-G1 -- releases are compared on the reference workloads

Each release candidate is measured against the previous release on the reference workloads, and a
regression beyond run-to-run spread, or a new steady-state alarm, is a release finding.
Status: proposed (adr:0016)
Enforcement: adr:0016

## Deferred until the instruments exist

Profile-guided and AI-assisted optimisation are deliberately not proposed yet. Both need ground
truth to be checked against: replayable workloads (ADR 0005), a representation to transform (ADRs
0011 and 0012), and a regression gate to accept or reject a transform (ADR 0016). Proposed before
those exist, an optimiser would have nothing to prove it correct or faster.

## Ruled out

- **Enforcing the frame invariants in CI.** CI has neither the game dumps nor a GPU (the
  GPU-execution jobs run on lavapipe), so a frame budget cannot be measured there. The static
  ratchet rules are the CI half; the runtime alarms are checked at release and in A/B runs.
