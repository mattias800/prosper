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

Esc or closing the window quits.

## Options

- `--dump <app0>` — boot and display the PS5 title at this app0 directory (positional path also works).
- `--test-pattern` — feed a synthetic animated frame through the real present path (no guest).
- `--frames N` — present N frames then exit 0 (non-interactive smoke; exit 1 if it couldn't).
- `--record PATH` — record the final controller stream to a replayable `PROSPER_PAD_SCRIPT` route.
  Missing parent directories are created; release the last held button before stopping so its interval
  is closed and flushed.

## Notes

- Two Vulkan contexts by design (`docs/FRONTEND_APP.md`): the core renders headless; the app owns a
  separate presentation device; frames cross as CPU pixels via `present_readback`.
- On window-close the guest thread is detached and reclaimed by process exit (a cooperative
  flip-boundary stop is a documented follow-up).
