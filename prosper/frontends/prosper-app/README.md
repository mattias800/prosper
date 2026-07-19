# prosper-app

The OS-integration frontend: an SDL3 window + Vulkan swapchain that runs a PS5 title on the desktop —
rendering, audio out, and controller in. See `prosper/docs/FRONTEND_APP.md` for the design and issue
#164 for the plan.

**Status:** boots and displays the game natively on Linux, Windows, and macOS, with audio and
controllers. `--dump <app0>` boots a title
via the shared `boot_program()` path, composites its GPU submits with the shared live renderer, and
presents them to the window; `sceAudioOut` is routed to the host and a host gamepad drives
`libScePad`. `--test-pattern` verifies the window + swapchain without a dump.

## Build

Default OFF. Enable the app and, for the full experience, the SDL3 audio + gamepad frontends. SDL3 is
fetched + built automatically if not installed; Vulkan is required.

```bash
cmake -S prosper -B build-app -DPROSPER_APP=ON -DPROSPER_AUDIO_SDL3=ON -DPROSPER_PAD_SDL3=ON
cmake --build build-app -j8 --target prosper-app
```

(`-DPROSPER_APP=ON` alone builds a video-only app — no audio/controllers.)

### Native Windows

From the repository root, the launcher configures a MinGW/Ninja build with video, WASAPI audio, and
SDL3 controller support, then starts the supplied title:

```powershell
.\prosper\scripts\run-windows.ps1 .\PPSA24651-app0
```

It reuses `prosper/build-mingw-app` on later runs. Use `-NoBuild` to skip the configure/build check,
`-TestPattern -Frames 120` to smoke-test the real window and Vulkan swapchain without a game, and
`-GuestArgs ''` when a title must not receive Unity's `-force-gfx-direct` argument. A Vulkan SDK,
CMake, Ninja, and MinGW-w64 UCRT are required; the launcher discovers the standard winget WinLibs
installation automatically.

WSLg remains an alternate way to run the Linux build, but it is no longer the primary Windows path.
Prebuilt Windows archives include `start-prosper.ps1`; see `prosper/docs/WINDOWS_RELEASE.md` for the
no-UI launch command, save-data selection, keyboard map, and runtime requirements.

## Run

```bash
# The game (reaching the frame loop needs these two env switches, same as boot_trace):
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
  ./build-app/prosper-app --dump /path/to/PPSA24651-app0

# Window/swapchain smoke without a dump:
./build-app/prosper-app --test-pattern
```

F11 or Alt+Enter toggles borderless desktop fullscreen. Esc or closing the window quits.

## Options

- `--dump <app0>` — boot and display the PS5 title at this app0 directory (positional path also works).
- `--test-pattern` — feed a synthetic animated frame through the real present path (no guest).
- `--frames N` — present N frames then exit 0 (non-interactive smoke; exit 1 if it couldn't).
- `--record PATH` — record the final controller stream to a replayable `PROSPER_PAD_SCRIPT` route.
  Missing parent directories are created; release the last held button before stopping so its interval
  is closed and flushed.

## Notes

Set `PROSPER_APP_DUMP_FRAMES=1` to retain the renderer's normal composited BMP checkpoints while the
interactive harness runs. `PROSPER_FRAME_DIR` selects their directory (the current directory by default).
This is useful when a title only reproduces in the realtime app; use the screenshot frontend for routed
PNG sequences and manifests when both frontends reproduce the same behavior.

For frontend A/B diagnostics, `PROSPER_APP_DISABLE_AUDIO=1`, `PROSPER_APP_DISABLE_PAD=1`, and
`PROSPER_APP_DISABLE_DIALOG=1` individually disable the corresponding SDL backend. The app then uses
the core's realtime silent audio sink, keyboard/scripted pad input, or headless dialog auto-dismiss.
These switches isolate frontend-specific behavior without requiring a separate build.

- Two Vulkan contexts by design (`docs/FRONTEND_APP.md`): the core renders headless; the app owns a
  separate presentation device; frames cross as CPU pixels via `present_readback`.
- On window-close the guest thread is detached and reclaimed by process exit (a cooperative
  flip-boundary stop is a documented follow-up).
- `PROSPER_APP_STALL_DUMP_MS=<milliseconds>` prints the recent Windows guest-exception ring when no new
  presented frame arrives within that interval. It is an unattended-run diagnostic, disabled by default.
- `PROSPER_APP_GUEST_DUMP_MS=<milliseconds>` prints that ring plus the live Windows guest-thread
  instruction/stack pointers once after the app loop has run for the requested time, even while frames
  continue. Use it for guest progression stalls that are not render stalls; it is disabled by default.
  `PROSPER_APP_GUEST_DUMP_INTERVAL_MS` repeats the sample at that interval, and
  `PROSPER_APP_GUEST_DUMP_PATH` writes thread samples to a dedicated append-only file so other trace
  streams cannot interleave with them. `PROSPER_APP_GUEST_DUMP_PTHREAD=<id>` limits each sample to
  one guest pthread (for example `0x2` for a title's main thread) to reduce observer overhead.
  On Windows, each enabled sample briefly uses `SuspendThread`/`GetThreadContext`/`ResumeThread` for
  every selected live guest thread, captures all of its active registered waits in the same suspension
  window (nested waits are listed rather than reduced to an arbitrary slot), and
  reads at most 16 KiB of its registered stack. This perturbs scheduling and is a diagnostic checkpoint,
  not a profiler. The registry supports 1,024 live guest
  threads and uses unique generation-token publication, so a sampler never consumes a partially published
  or reused slot. Each registration pins its actual Windows thread handle; the sampler duplicates that
  handle and revalidates the same generation after suspension, so a recycled numeric thread ID cannot
  redirect a checkpoint to an unrelated thread. Lifecycle registration runs only at thread start/exit.
  Wait kind/source metadata is published on the existing Windows interruptible-wait path even when
  timed sampling is disabled; the sampler only reads it when one of these checkpoints is requested.
