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
- `live_target_format.hpp` — the guest↔Vulkan pixel-format mapping. Compiled with `-Werror=switch`
  on purpose: a silent RGBA8 fallback has cost two titles a whole render layer.
- `decode_scratch.hpp` — the pooled full-surface intermediates the texture decode branches in
  `live_renderer.cpp` stage through. Header-only and Vulkan-free. It is here rather than in
  `shared/texture/` because that folder holds *decision* logic that owns no memory, and this owns
  the memory. The one rule it exists to make explicit: a lease arrives holding the PREVIOUS
  surface, so a caller that fills it partially must say so (`zero_tail`) — a fresh
  `std::vector<uint8_t>(n, 0)` used to supply that tail for free, and nothing else would notice it
  had stopped.

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
