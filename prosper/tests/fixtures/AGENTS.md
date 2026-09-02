# `prosper/tests/fixtures/` — shared harnesses and hand-built inputs

Header-only material that more than one test needs: the Vulkan/compute execution harnesses, and
hand-constructed binary inputs (a synthetic PRX, an ATRAC9 vector, captured GTA V shader words) that
would otherwise be pasted into each test that wants them. Nothing here is a test itself — every file
is included by tests in the sibling directories, which is why it is all headers with `inline`
definitions and no translation unit of its own.

Two kinds live here and they are worth telling apart:

- **Harnesses** — `render_runner.h`, `compute_runner.h`, `image_compute_runner.h`, `test_scratch.h`.
  These *do* things: bring up a device, record commands, read pixels back.
- **Fixtures proper** — `at9_testvec.h`, `handmade_prx.h`, `synth_prx.h`, `spirv_*.h`, `test_data.h`,
  `gta5_*_fixture.hpp`. These are data, frozen at a known-good state so an assertion has something
  stable to compare against.

## `render_runner.h` is NOT test-only, and the directory name is the trap

It is the project's **offscreen Vulkan backend**, and the shipping frontend compiles it:
`frontends/shared/live/live_renderer.cpp` and `present/present_blit.cpp` include it. (This line used
to name `live_compute.cpp` as a third includer. It is not one — checked with the preprocessor, not
with grep: `-M` over that translation unit's own compile command lists no `render_runner.h`. The
compute backend is a separate Vulkan backend that shares the device, not this header.) `render_draw_pass_rgba` is therefore a live render path — a wrong failure path in it
silently drops real rendered content in a real game, and "it is under `tests/`" has misled reviewers
into treating changes here as test-only (#3210 had to say so explicitly in its own body).

Consequences for anything you change in it:

- **Dropping a pass or skipping a draw is a VISIBLE regression** when the condition can fire on a
  healthy device. Guard on the API's own `VkResult`, not on a heuristic, and say in the PR what a
  false trigger would cost.
- **It cannot depend on the app** — no capture singleton, no frontend state — because it is also
  compiled into Vulkan tests that link none of that.
- **It cannot depend on the HLE's locks either, and that one is easy to miss** (#2953). The live
  route reaches this backend only under `g_agc_state_mu` (`src/hle/graphics/hle_agc.cpp`, #278), so
  reading the live path alone makes the persistent caches look safely single-threaded. `gpu_replay`,
  `boot_trace` and every Vulkan test compile this header and link no HLE, so on them that invariant
  simply does not exist. Process-lifetime state declared here carries its own synchronisation:
  `render_draw_pass_rgba` holds `BackendPersistentResourceGuard` for its whole body, which covers
  every `static` that function owns. It does **not** cover the colour-target and depth/stencil
  caches' other entry points (`invalidate_persistent_color_target*`,
  `readback_persistent_color_target`, `snapshot_persistent_ds_images`, and the frontend's direct
  iteration of both), so do not add a new one without reading #3240.
- **A readback has TWO halves and they are independent** (#2944). Any device write the CPU then maps
  and reads needs `record_host_read_barrier()` — the availability operation into the host domain,
  which a fence wait does not perform — and, when the memory may be non-coherent,
  `invalidate_mapped_readback()`. Coherent memory still needs the first. Neither half can be checked
  by reading the pixels back: on every driver here the unsynchronized code returns correct bytes, so
  the guard is `host_read_barrier`, which asserts the barrier was RECORDED.
  Since #3249 the availability half lives in `src/gpu/execute/host_read_barrier.hpp` and is only
  **re-exported** into `prosper::test` here, because the live compute backend needs the identical
  rule for its dispatch results and two spellings of it is how the next site gets missed. The
  invalidate half stays in this header, with the allocator that can actually return non-coherent
  memory. Its counter is process-wide and shared with the compute guard,
  `live_compute_host_read_barrier`.
- **Two writes to the same image are not ordered by a barrier that names some other image** (#3248).
  Any `vkCmdPipelineBarrier` between them supplies the execution dependency, and an execution
  dependency is not enough for write-after-write: the first write also has to be made available.
  `record_transfer_write_after_write_barrier()` is that dependency. The mip-assembly path had a
  full-image clear followed by per-level copies with barriers only on the copy sources, which is
  the shape to look for whenever a path writes one destination from several commands.
  Unfalsifiable from output, and worse than the readback case: assembly clears to black on purpose,
  so a clear that lands after a copy produces exactly what a correct run produces for a level the
  guest never rendered. The guard is `mip_assembly_barrier`.
- **A diagnostic in here is a Vulkan API user like any other, and until #3248 no test ran one.**
  `PROSPER_GEOM_PROBE` and `PROSPER_DRAW_ISO` record real commands; because no ctest case armed
  them, `tools/vkval` could not see their misuse however it was configured, and both were misusing
  Vulkan *and* misreporting because of it. `render_diagnostic_paths` runs both once so the layer
  does see them. Arm a new env-gated render path there in the same change that adds it — and note
  the arming has to come from the ctest ENVIRONMENT, not `setenv()` inside the test, because these
  names are cached one-shot reads.

## Adding to it

Prefer extending an existing harness over a second one that does nearly the same thing; the tests
that share `render_runner.h` share its device, caches and statistics, and a parallel copy would
diverge. Fixtures that only one test uses belong beside that test, not here.
