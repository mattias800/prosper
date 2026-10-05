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
performance ratchets*) states the same six; this page is the form with IDs and enforcement, and
the charter will point here once that change lands on its own (charter edits are reviewed alone).

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
backend; staged fix #3948).
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

## Ruled out

- **Enforcing the frame invariants in CI.** CI has neither the game dumps nor a GPU (the
  GPU-execution jobs run on lavapipe), so a frame budget cannot be measured there. The static
  ratchet rules are the CI half; the runtime alarms are checked at release and in A/B runs.
