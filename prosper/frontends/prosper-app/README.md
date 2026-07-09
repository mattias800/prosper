# prosper-app

The OS-integration frontend: an SDL3 window + Vulkan swapchain that presents prosper's rendered
frames. See `prosper/docs/FRONTEND_APP.md` for the design and issue #164 for the plan.

**Status:** the present pipeline + the shared guest boot are done. `--test-pattern` shows an animated
frame (verifies window + swapchain). `--dump <app0>` boots the actual game via the shared
`boot_program()` path. **Caveat:** the game won't appear on screen yet — nothing reaches the window
until the composite renderer is registered (the render-frontier-owned `set_submit_renderer` lambda),
which is the coordinated next step. See `prosper/docs/FRONTEND_APP.md`.

## Build

Default OFF. Needs SDL3 (fetched automatically if not installed) and Vulkan:

```bash
cmake -S prosper -B build-app -DPROSPER_APP=ON
cmake --build build-app -j8 --target prosper-app
```

On Windows this is built/run under WSL2; the window appears on the Windows desktop via WSLg. A real
GPU needs your vendor's WSL Vulkan driver (else it falls back to the `llvmpipe` software rasterizer —
fine for verifying the plumbing, slow for real frames). Confirm your device with `vulkaninfo`.

## Run

```bash
./build-app/prosper-app --test-pattern          # animated gradient in a window; Esc / close to quit
./build-app/prosper-app --test-pattern --frames 30   # present 30 frames then exit 0 (smoke check)
```

Without `--test-pattern` it waits for frames from the core's present layer (P0b: the guest boot).

## Options

- `--test-pattern` — feed a synthetic animated frame through the real present path (no guest).
- `--frames N` — present N frames then exit 0 (non-interactive smoke; exit 1 if it couldn't).
