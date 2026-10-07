---
kind: adr
status: proposed
date: 2026-10-07
---

# ADR 0032: Page-granular write tracking on Windows hosts

## Context

`PERF-P9` (proposed, ADR 0010) asks every cache to validate against page tracking and to treat a
full byte comparison as a counted fallback. On Linux that tracker exists; on Windows it does not,
so every Windows cache validation is the fallback.

**Linux today.** `src/host/memory/guest_write_watch.cpp` arms a watch by write-protecting the
guest pages (`watch_mprotect`, `guest_write_watch.cpp:612`) and services the first store from the
SIGSEGV handler (`exec_image_linux.cpp:1328` calls `guest_write_watch_handle_fault_ex`). It arms
only when that handler runs on the sigaltstack, because a signal frame written on the faulting
stack would land in the guest's 128-byte SysV red zone (`guest_write_watch.cpp:1468`,
`guest_write_watch_set_fault_onstack`, `guest_write_watch.hpp:379`). Host writers bracket
themselves with `guest_write_watch_notify_host_write` / `_done`, GPU writes with
`guest_write_watch_notify_gpu_write` (`guest_write_watch.hpp:301-315`).

**Windows today.** The same file carries a second, reduced copy of the tracker under `#ifdef
_WIN32` (`guest_write_watch.cpp:73` onwards) whose `GuestWriteWatch::create` always returns empty
(`guest_write_watch.cpp:350-366`). The recorded reason is the same red-zone hazard with no escape:
Windows builds its exception-dispatch frame below the interrupted RSP *before* a vectored handler
runs, and there is no alternate exception stack. The project already designs around this in three
other places -- direct memory is a sparse-file section so a first touch never faults
(`hle_kernel_mem.cpp:4861-4866`), SSE4a `EXTRQ` is detoured rather than trapped
(`exec_image_win.cpp:238-241`, `:609-611`). The VEH does call `guest_write_watch_handle_fault`
(`exec_image_win.cpp:1034-1036`), which returns false. The reduced copy is exactly what the
host-platform seam forbids: a platform arm that re-implements logic instead of supplying a
primitive (`docs/architecture/HOST_PLATFORM_SEAM.md`, ADR 0002).

**What falls back.** Callers treat an empty watch as "keep the exact compare": the resident-buffer
cache in `tests/fixtures/render_runner.h:4936-4999` (which also disables a watch after two dirty
queries, the `PERF-P5` violation #3155 records), and the compute buffer cache and renderer
validations counted by `WriteWatchCensus` (#4681). Measured by #4681 on *Black Flag*, Windows/NVIDIA,
one ~110 s run: 1,180 renderer compares (5.21 GB) and 5,955 compute-cache compares (66.07 GB); of the
renderer compares 5 found a change. `AC_BLACK_FLAG_STATUS.md` § Ruled out puts the constant-ring
compare at ~11 ms/frame under load for 0.25-0.3 MiB actually rewritten per frame.

**What #4681 ruled out.** `GetWriteWatch` works on private memory (including a placeholder replaced
with `MEM_WRITE_WATCH`) and fails with `ERROR_INVALID_PARAMETER` on every section view;
`MapViewOfFile3` refuses `MEM_WRITE_WATCH`. `PROSPER_VALIDATION_MAPPING_CENSUS` classified the
compared bytes with `VirtualQuery` (`src/host/platform/mapping_class_win.cpp:15-16`): 100% section
views, 0% private, in every run. So `GetWriteWatch` alone tracks none of today's traffic
(`RENDERER_PERFORMANCE_2026_07.md` § Ruled out).

## Decision

1. **One tracker, primitives per host.** The page state machine, registrations, generations,
   host/GPU write notifications and the compare-to-arm ordering live once in `src/host/memory/`.
   Each host supplies only a primitive in `src/host/platform/page_tracking*` (one header, per-OS
   sources selected by the build, as `mapping_class_*` and `committed_section_*` are):
   - `protect_run(addr, len)` / `unprotect_run(addr, len)` -- make a run read-only / writable;
   - `fault_is_red_zone_safe()` -- whether a write fault on a guest thread is delivered without
     writing below that thread's RSP;
   - `harvest(addr, len, out_pages)` -- optional fault-free "which pages were written since the last
     harvest", or `unsupported` for a range it cannot see.
   The `_WIN32` copy at `guest_write_watch.cpp:73-557` is deleted; Windows runs the portable tracker
   over a primitive that currently answers `unsupported` / not red-zone safe. Behaviour does not
   change in that step.
2. **Write protection is armed over guest-writable memory only where `fault_is_red_zone_safe()`
   holds.** On Windows it does not, so a `VirtualProtect(PAGE_READONLY)` + vectored-handler guard is
   **not** adopted for guest-written pages until something makes the fault red-zone safe. A silent
   corruption of a leaf function's locals is worse than any compare cost. (Spec rule `PERF-P13`.)
   One such thing exists in practice and is evaluated in Migration step 5: statically rewriting the
   guest instructions that can fault while red-zone bytes are live, so the fault is taken with RSP
   already below the red zone (the shadPS4 design, below).
3. **Windows' fault-free tracking comes from `harvest` over private memory.** The tracker asks the
   primitive first; a range the primitive cannot see stays on the counted full-compare fallback, and
   the fallback names the reason (`section-view`, `unsafe-fault`, `disabled`) in the census.
4. **Direct memory moves into `harvest`'s reach only where aliasing allows**, measured first: a
   direct-memory range mapped at exactly one VA can be backed by private `MEM_WRITE_WATCH` memory;
   one that is aliased stays section-backed. Whether to back single-alias ranges privately, and how
   a later alias migrates a range back to a section without racing guest threads, is decided by the
   census in Migration step 2, not by this ADR.
5. **Performance is decided by a same-binary A/B** (`.claude/skills/perf-change/`) on the three
   reference workloads plus *Black Flag* (where #4681 measured the traffic), with the new path behind
   one `PROSPER_*` guest-behaviour-neutral switch whose control arm is the full compare. A tracker
   that answers "unchanged" wrongly is a correctness defect, so the A/B also runs
   `PROSPER_WATCH_QUERY_AUDIT=1`-style re-derivation: every "clean" answer re-compared in a
   diagnostic arm, zero disagreements required.

## Consequences

- The Windows arm shrinks to primitives; Linux behaviour is unchanged by step 1.
- The compare cost on Windows falls only for memory `harvest` can see. If the step-2 census finds
  the compared bytes are mostly aliased direct memory, this ADR delivers the seam and the reasons
  in the census, and little speed-up; that is an acceptable, recorded outcome.
- Costs of `harvest`: `GetWriteWatch` is a kernel call per range per validation, and
  `WRITE_WATCH_FLAG_RESET` makes harvest-and-rearm one call; it is O(pages) in the queried range, not
  O(bytes). A write by the GPU (imported host memory) or by a DMA-shaped HLE producer is invisible
  to it, exactly as on Linux, and keeps going through `notify_gpu_write` / `notify_host_write`.
- Ordering stays as on Linux: re-arm (reset) *before* reading the bytes a cache retains, so no write
  between compare and arm can be blessed (`render_runner.h:4990-4993`).
- Enforcement: the census (fallback bytes by reason) and `runtime:host-copy-pressure`; review for
  rule 2, because no static check can tell guest-writable memory from host-only memory.

## Alternatives considered

- **`VirtualProtect` guard + vectored handler** (mark dirty, restore `PAGE_READWRITE`, re-arm on
  validation). Cheap in principle: one exception per first write per page per epoch, and a page
  written twice faults once; concurrent writers both fault and the second finds the page already
  writable, which the handler must accept. Rejected for guest-written memory by the red-zone hazard
  above; the handler cannot repair bytes the kernel overwrote before it ran. Safe for host C++
  writers (the Microsoft x64 ABI has no red zone), but nothing proves a page has no guest writer, so
  it is not a usable discriminator. Large pages could not be protected below their own granularity
  either (from knowledge, unverified).
- **`GetWriteWatch` on today's mappings.** Ruled out by #4681: section views are refused.
- **Copy-on-write views (`PAGE_WRITECOPY`)**: fault-free from the guest's side, but a written page
  becomes private and stops being coherent with its aliases and with GPU-imported memory, which is
  the property the sparse-file section exists to provide.
- **Hash or compare only changed ranges.** Still needs a change signal; hashing the whole range is
  O(size) like the compare (ADR 0010, Alternatives).
- **Hypervisor dirty logging (Windows Hypervisor Platform) or a kernel driver.** Would move the guest
  into a VM or ship a driver; out of proportion to a cache-validation cost.
- **How other emulators do it.**
  - *Native x86-64 guests on Windows (the same constraint as prosper)*. Read from their sources;
    these are design references only, and nothing was copied:
    - shadPS4 uses `VirtualProtect` plus a vectored handler, over 4 KiB pages with per-page watcher
      counts (`src/video_core/page_manager.cpp`, `src/core/address_space.cpp:268`). It makes that
      safe with **static red-zone protection** (`src/core/cpu_patches.cpp:1128-1952`, enabled in
      `src/emulator.cpp:554`). It computes a per-instruction 128-bit red-zone liveness mask over the
      guest code, and rewrites each memory access that can fault while red-zone bytes are live into
      a trampoline that runs it with `rsp` lowered by 128. The dispatch frame then lands below the
      live bytes.
    - KytyPS5 uses `VirtualProtect` plus a vectored handler (`common/hostException.cpp`,
      `graphics/host_gpu/pageManager.cpp`). Whether it guards the red zone was not checked.
  - *Fault-free, Linux:* AnyPS5 uses asynchronous userfaultfd write-protect plus the `PAGEMAP_SCAN`
    ioctl (`GuestWriteWatch.cpp`), which is a natural `harvest` for the Linux arm. shadPS4 offers
    synchronous uffd-wp as an opt-in.
  - *Fault-free, Windows:* portps5 uses `MEM_WRITE_WATCH` arenas with a bounded `GetWriteWatch`
    reset pass (`WriteTracker.hpp`). This is the shape of Decision 3.
  - From knowledge, unverified: Ryujinx and yuzu protect host pages or instrument stores in their JIT.
    Their guest is JIT-compiled ARM code that never runs on the host stack under the SysV ABI, so the
    red-zone constraint does not arise for them.

## Migration order

1. Extract the primitive seam and delete the Windows copy of the tracker; no behaviour change, Linux
   tests unchanged, the Windows arm reports `unsupported`. Its own PR (a relocation).
2. Extend `PROSPER_VALIDATION_MAPPING_CENSUS` with the alias count of each compared direct-memory
   range (`guest_write_watch_va_to_phys` already returns one on Linux), on *Black Flag* and the
   reference workloads. This decides whether step 4 is worth doing.
3. Windows `harvest` over private `MEM_WRITE_WATCH` allocations, with a test that writes a page from
   guest-ABI code and from host code and requires both to be reported, and a mutation arm that
   expects `unsupported` for a section view.
4. Only if step 2 shows single-alias direct memory carries the compared bytes: private backing for
   single-alias ranges, with the migration-to-section path specified and tested under concurrent
   guest writers. Then the A/B in Decision 5.
5. Evaluate static red-zone protection as the way to make the protect/fault primitive red-zone safe
   on Windows, for the memory `harvest` cannot see:
   - Build a red-zone liveness analysis over the guest's executable segments.
   - Rewrite the at-risk memory accesses through trampolines.
   - Add a test that faults from a leaf with a live red zone and checks the bytes, with a mutation
     arm that skips the rewrite and must corrupt them.

   This step is gated on the step-2 census showing that section-backed memory carries the cost, and
   on measuring the rewrite's own overhead and coverage (indirect branches and self-modifying code
   must be refused, never guessed). Only after this does `fault_is_red_zone_safe()` answer yes on
   Windows.

## Open questions

- Is the Windows dispatch frame's overlap with `[RSP-128, RSP)` measured anywhere in this project,
  or only reasoned? The code comments assert it; a test that faults from a leaf with a live red zone
  and checks the bytes would settle it on each Windows version. `CONFIDENCE: MED`.
- What share of compared bytes is single-alias? Unknown until step 2.
- How much guest code static red-zone protection must rewrite on a real title, and its runtime cost.
  Its correctness depends on decoding every reachable instruction, which ties it to prosper's
  existing guest-code analysis tools.
- Does private backing change anything the guest can observe (`sceKernelVirtualQuery` results,
  commit accounting)? It must not; step 4 needs the HLE conformance tests (ADR 0020) to say so.

## Approval

Requires the project owner's acceptance. Acceptance unblocks step 1 (a seam refactor ADR 0002
already permits) and the `PERF-P13` rule; steps 3-4 additionally wait on the step-2 census.
