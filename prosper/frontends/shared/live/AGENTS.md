# `frontends/shared/live/` — the two live Vulkan backends

What turns a decoded guest submit into real GPU work at runtime, shared by every frontend
(`prosper-app`, `boot_trace`, `tools/screenshot`) so they all drive the *same* backend rather than
three that drift. Built as the `prosper_live_renderer` static library, which exists only where CMake
found Vulkan.

- `live_renderer.cpp` — the graphics half: DrawItem → Vulkan compositor, render-target/RTT authority,
  present. It is a *driver* for the offscreen backend rather than the backend itself; the pipeline,
  pass and readback code it calls lives in `tests/fixtures/render_runner.h` (whose directory name is
  a trap — see that folder's `AGENTS.md`).
- `live_compute.cpp` — the compute half, and a **separate Vulkan backend**, not a caller of the
  render one. It builds its own device (or adopts the renderer's when one is published), its own
  pipeline cache, descriptor pools, memory pool and command buffers, and does not include
  `render_runner.h` at all. Reflected storage buffers and storage images are materialized from guest
  memory, dispatched, and written back into guest memory synchronously.
- `gpu_retile.hpp` — exact 2D storage-image writeback layout conversion in the guest compute
  submission. Keeps linear comparison baselines separate from the tiled host-read buffer.
- `packed_rtt_conversion.hpp` — device-owned RGBA8→packed-10-bit sampled conversion.
  Records transfers and conversion into the guest compute submission; setup failures retain their
  `VkResult` so optional fallback cannot hide device loss.
- `live_target_format.hpp` — the guest↔Vulkan pixel-format mapping. Compiled with `-Werror=switch`
  on purpose: a silent RGBA8 fallback has cost two titles a whole render layer.
- `decode_scratch.hpp` — the pooled full-surface intermediates that both the texture decode branches
  in `live_renderer.cpp` and the per-dispatch seed/pack buffers in `live_compute.cpp` stage through
  (`ScratchBuffer` is the drop-in shape for the latter's old `std::unique_ptr<uint8_t[]>`).
  Header-only and Vulkan-free. It is here rather than in
  `shared/texture/` because that folder holds *decision* logic that owns no memory, and this owns
  the memory. The one rule it exists to make explicit: a lease arrives holding the PREVIOUS
  surface, so a caller that fills it partially must say so (`zero_tail`) — a fresh
  `std::vector<uint8_t>(n, 0)` used to supply that tail for free, and nothing else would notice it
  had stopped. `PROSPER_DECODE_SCRATCH_POISON=1` is the arm that makes such a caller fail loudly
  instead: it fills every lease before handing it out, so nothing can quietly inherit the zeros a
  fresh mmap used to supply.

## The boundary that is easy to get wrong

The compute backend is where *guest memory* and *device memory* meet twice per dispatch — an upload
before, a writeback after — and both directions have a synchronization contract that no output check
can verify:

- **A host READ of anything the device wrote needs `record_host_read_barrier()`**
  (`src/gpu/execute/host_read_barrier.hpp`). Waiting the dispatch fence orders execution; it does not
  perform the availability operation into the host domain. Every writable storage buffer, every
  storage image's staging buffer, the comparator's flag word and every retained result baseline is
  device-written host-visible memory, so every one of them carries the dependency (#3249).
- **The scope is the ALLOCATION, not the binding.** `allocate_memory`/`release_memory` run a pool
  that recycles host-visible allocations across bindings and submits, and a later binding maps a
  recycled allocation and *reads its retained contents* to decide whether an upload is needed. So a
  buffer this dispatch never maps still needs the barrier if the device wrote it.
- **Neither can be checked by asserting the result.** Every allocation here is `HOST_COHERENT` (both
  tiers of `host_memory_type` require it), and on the drivers this project runs on the unsynchronized
  code returns byte-identical results. The guard is `live_compute_host_read_barrier`, which asserts
  the barrier was *recorded*; synchronization validation cannot see this class either (#3248).

`execute_live_compute_items()` is the one entry point exposed for tests, which is how
`game_compute_exec`, `live_compute_descriptor_array` and `live_compute_host_read_barrier` drive the
production backend without a frontend or a game dump.

## Measuring buffer setup and writeback

The persistent buffer budget charges primary allocations and optional exact-result baselines.
Primary admission first proves that enough unpinned storage can be reclaimed, then drops unpinned
baselines before evicting primary entries. Baseline admission uses spare capacity only. One owner
pin protects both handles through completion; a failed submission without completion proof keeps
that pin. Reclaiming a baseline preserves the primary's guest-content validation and write watches.
Unchanged writeback preserves watch-promotion progress already earned by exact source validation;
it neither resets that progress nor counts it a second time. Changed content resets progress.

`PROSPER_COMPUTE_BUFFER_TIMING=1` emits `[compute-buffer-timing]` records after cleanup. Use
`PROSPER_COMPUTE_TIMING_CODE` / `PROSPER_COMPUTE_TIMING_HASH` to select a program and
`PROSPER_COMPUTE_TIMING_CAPTURE_ONLY=1` to bound records to F8; the existing `TRACE_ONLY` selector
also applies. No guest-content hashes or extra comparisons are added, and the added timers read no
clocks when disabled. Keep `PROSPER_COMPUTELOG` off when measuring: its additional scans and output
are included in enclosing timers.

Each resolved allocation owner gets one row, including read-only owners. `owner-index` identifies
the flattened descriptor-table entry; `bindings` includes the owner and exact aliases, and `aliases`
counts only the additional entries. Writability is combined across aliases. On setup failure, later
descriptors may have `owner-resolved=0`; exclude those placeholders from allocation counts. Join
records using submit/dispatch/order/program identity. CPU fast paths, proven no-ops and declines
before materialization produce no buffer rows; these records are not the whole F8 population.

`cache`, `validation`, `baseline`, `gpu-compare` and `writeback` report decisions already made by the
backend. `unobserved` means the stage was not reached. GPU comparison progresses through eligible,
prepared, recorded and a completed changed/unchanged result; `no-result` means completion occurred
but the flag was not read, and `setup-failed` retains the CPU fallback. Check `ok` before treating a
row as a completed dispatch. `host-key` is the existing cache-key pointer, whereas `host-backed`
describes effective source selection (a short host mirror falls back to guest memory).

`cache-bytes` and `cache-limit` report end-of-dispatch requested residency. The owner's
`primary-allocation-bytes` and `result-allocation-bytes` report the surviving cache entry's separate
charges (zero if absent). These use Vulkan resource memory requirements, as the existing budget
does; the underlying allocation pool can supply a larger allocation. They are not a physical
device-memory usage measurement or cap.

Byte counters are lengths passed to comparisons and copies, not actual CPU bytes read by an
early-exiting comparison. `compared-bytes` includes the initial exact comparison against pooled
allocation contents. `upload-skipped` is the existing cached-source authority bit: zero upload bytes
on a cold, already-equal pooled allocation does not set it. Atomic image writeback uses
`guest_layout_ms`; `guest-copied-bytes` counts only the ordinary contiguous buffer copy.

Timers are nested, not additive: `setup_ms` covers owner materialization after resource/alias checks;
`validation_ms` includes upload compare/copy/map/watch on cache hits. `writeback_ms` encloses result
comparison, guest copy/layout, map, host-write-watch notification, baseline creation, architectural
notification, source validation/watch rearming and provenance. Those last two have separate
`source_validation_ms` and `provenance_ms` fields. Use the existing dispatch phase totals for the
remaining setup checks and loop overhead; do not claim the sum of owner timers covers all setup.
