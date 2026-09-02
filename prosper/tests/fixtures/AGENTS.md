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
`frontends/shared/live/live_renderer.cpp`, `live_compute.cpp` and `present/present_blit.cpp` all
include it. `render_draw_pass_rgba` is therefore a live render path — a wrong failure path in it
silently drops real rendered content in a real game, and "it is under `tests/`" has misled reviewers
into treating changes here as test-only (#3210 had to say so explicitly in its own body).

Two consequences for anything you change in it:

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

## Adding to it

Prefer extending an existing harness over a second one that does nearly the same thing; the tests
that share `render_runner.h` share its device, caches and statistics, and a parallel copy would
diverge. Fixtures that only one test uses belong beside that test, not here.
