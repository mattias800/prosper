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

Pass `-PresentMode mailbox` (low-latency vsync) or `-PresentMode immediate` (tearing permitted) to
the Windows launcher when the driver supports it. The default is FIFO.

WSLg remains an alternate way to run the Linux build, but it is no longer the primary Windows path.
Prebuilt Windows archives include `start-prosper.ps1`; see `prosper/docs/WINDOWS_RELEASE.md` for the
no-UI launch command, save-data selection, keyboard map, and runtime requirements.

## Run

```bash
# The game (reaching the frame loop needs this env switch, same as boot_trace):
PROSPER_GUEST_ARGS=-force-gfx-direct \
  ./build-app/prosper-app --dump /path/to/PPSA24651-app0

# Window/swapchain smoke without a dump:
./build-app/prosper-app --test-pattern

# No game on the command line: the window opens empty and asks for one (see below).
./build-app/prosper-app
```

Pause or F10 pauses/resumes at a guest flip boundary and freezes the audio stream. F11 or Alt+Enter
toggles borderless desktop fullscreen. Esc or closing the window quits.

## Opening a game from the app

The command line stays the primary way in — every agent route, snapshot run, and CI check drives this
binary by argv, and that path is unchanged. Someone who runs a released build without one would
otherwise face an empty window, so a dump path can also arrive interactively (#1469):

- **Drop a game folder on the window.** The app0 root, a file inside it (`eboot.bin`), `sce_sys`, or a
  file one level deeper all resolve to the title. One game is taken per drop, so dropping several
  folders at once does not start one and immediately switch away from it. Not accepted under
  `--test-pattern`, which already has something feeding the present layer.
- **Ctrl+O** opens the host's native folder picker. Available only while no game is running — once a
  guest boots it owns the keyboard (O is its R1) — and not under `--test-pattern`, which already has
  something feeding the present layer.
- **A launch with no arguments at all** opens that picker straight away, so a double-clicked app is not
  a dead end. `--pick` forces it for any launch; `--no-pick` disables it entirely.

A PS5 title is a *directory*, so this is a folder picker, not a file picker. A folder that is not a
title is reported and changes nothing — including the folder that merely *contains* your games, which
a library view would scan but this does not.

**One boot per launch.** `run_entry` does not observe `prosper_request_stop()` yet (#352), so a booted
guest cannot be torn down. Opening a title after this process has already tried to boot one therefore
starts a second process with the new game and shuts this one down. That path appends
`--dump <new title>` to the current run's own arguments, which works because `--dump` is last-wins.

It is the boot *attempt* that is spent, not the success: a title that fails to load also uses up the
launch, because the boot appends into shared state and re-runs one-shot global setup. Picking another
game after a failure is still fine — it just goes to a fresh process, same as switching titles.

This is the *user* opening a game they chose. It does not change what a **guest** can ask for: the SDL
dialog backend below still never lets a title open a host file picker or see a host path.

## The game library

Point the app at a directory holding your titles and it can enumerate them — content id, display name,
and cover icon — instead of asking for a folder every time.

```bash
./build-app/prosper-app --list-games --games-dir /path/to/games
#   PPSA13579	Blasphemous 2	/path/to/games/PPSA13579-app0
#   PPSA24651	The Messenger	/path/to/games/PPSA24651-app0

./build-app/prosper-app --set-games-dir /path/to/games   # remember it for next time
```

`--list-games` writes one tab-separated record per line — content id, display name, app0 path — to
**stdout**, with everything explanatory on stderr, so a script or an agent can consume it directly. It
exits 0 when it found titles, 1 when the directory held none, and 2 when the directory is unset or is
not a directory. It never opens a window, initializes Vulkan, or boots a guest.

Split records on tabs rather than whitespace: the first field is **empty** for a title with no
readable `param.json`, and display names contain spaces.

Where the directory comes from, highest priority first:

1. `--games-dir <path>`
2. `PROSPER_GAMES_DIR` in the environment
3. the persisted setting, written by `--set-games-dir`

Persistence exists only so someone who chose a folder in the GUI is not asked again next launch; it
never overrides what a command line or a script asked for. The settings file lives at
`$XDG_CONFIG_HOME/prosper/prosper-app.conf` (falling back to `~/.config/prosper/`, and `%APPDATA%\prosper\`
on Windows), overridable with `PROSPER_APP_CONFIG`. It is a plain `key = value` file you can edit by
hand. Settings a build does not recognize are carried through unchanged when it rewrites the file, so an
older build will not delete a newer build's settings — but comments are not preserved, since the app
regenerates its own header.

The scan looks **one level deep** and accepts a child directory as a title when
`resolve_app0_root()` does — the same test the drop and picker paths use. A title's own asset
subdirectories are therefore never mistaken for separate games, and the games directory itself is not
considered even when it happens to be a title root (use `--dump` or the picker for one specific game).
Names come from `sce_sys/param.json`, preferring the entry for the dump's own `defaultLanguage`; a title
with no readable metadata still appears, named after its directory, since the name is presentation and
`boot_program` only needs `eboot.bin`.

The listing is therefore "what the drop and picker paths would accept", not a guarantee that every entry
boots: that gate accepts `sce_sys/param.json` on its own, so a metadata-only folder with no `eboot.bin`
is listed and will fail when opened. Keeping one definition of "is this a title" across all three entry
points is worth more than pre-filtering the list.

### The library view

With a games directory set, launching with no game shows a grid of cover art instead of an empty
window. Arrow keys move the selection, Enter/Space opens the highlighted title, clicking a cover opens it
directly, and **Change folder...** picks a different games directory and remembers it. Esc quits. With no
directory set yet, the window explains that and offers the same folder picker on Enter or a click.

Keyboard and mouse only for now — **controller navigation is not implemented** (tracked separately).
Nothing initializes SDL's gamepad subsystem while the library is up: the pad backend does that inside
the guest boot, by which point the library is gone.

Cover art is each dump's `sce_sys/icon0.png`. A title whose icon is missing or undecodable still appears
as a launchable button labelled with its content id — what matters is that it is bootable, not that it
has a picture.

The view is drawn with Dear ImGui on the app's existing Vulkan device and swapchain
(`third_party/imgui`), and disappears the moment a guest boots: prosper runs one game per launch, so the
library never draws over a running title. If it cannot be brought up — no ImGui in this build, or a
device that refuses the render pass — the window falls back to the flat idle colour and every
command-line path keeps working.

Selection movement lives in `library_nav.hpp`, which is pure and unit-tested, so the grid's behaviour is
covered in ordinary CI rather than only by someone pressing arrow keys.

## Options

- `--dump <app0>` — boot and display the PS5 title at this app0 directory (positional path also works).
  Last-wins: a later `--dump` overrides an earlier one, and a positional path applies only while no
  dump has been given.
- `--test-pattern` — feed a synthetic animated frame through the real present path (no guest).
- `--pick` — open the host folder picker at startup even though arguments were given. Ignored when the
  run already has a game (`--dump`/positional) or `--test-pattern`.
- `--no-pick` — never open the picker at startup, including on a no-argument launch. Wins over `--pick`.
- `--games-dir <path>` — where your PS5 titles live, for this run only. See "The game library" above.
  A missing or empty path is an error (exit 2).
- `--set-games-dir <path>` — record that directory for future launches, then exit. An empty path
  (`--set-games-dir ""`) clears the stored setting; a missing one is an error (exit 2).
- `--list-games` — print the library as plain text and exit, with no window, no Vulkan and no guest.
- `--frames N` — present N frames then exit 0 (non-interactive smoke; exit 1 if it couldn't).
- `--present-mode fifo|mailbox|immediate` — choose swapchain latency behavior. FIFO is the default;
  mailbox is low-latency vsync, and immediate may tear. Unsupported optional modes fall back to FIFO.
- `--record PATH` — record the final controller stream to a replayable `PROSPER_PAD_SCRIPT` route.
  Missing parent directories are created; release the last held button before stopping so its interval
  is closed and flushed.
- `--record-axis flip|pad-read` — timestamp recorded intervals by display flips (the backward-compatible
  default) or by successful guest input-state reads. Pad-read routes remain stable when presentation
  pauses while the title continues polling input.

## Notes

Set `PROSPER_APP_DUMP_FRAMES=1` to retain the renderer's normal composited BMP checkpoints while the
interactive harness runs. `PROSPER_FRAME_DIR` selects their directory (the current directory by default).
This is useful when a title only reproduces in the realtime app; use the screenshot frontend for routed
PNG sequences and manifests when both frontends reproduce the same behavior.

For frontend A/B diagnostics, `PROSPER_APP_DISABLE_AUDIO=1`, `PROSPER_APP_DISABLE_PAD=1`, and
`PROSPER_APP_DISABLE_DIALOG=1` individually disable the corresponding SDL backend. The app then uses
the core's realtime silent audio sink, keyboard/scripted pad input, or headless dialog auto-dismiss.
These switches isolate frontend-specific behavior without requiring a separate build.

The SDL dialog backend presents message/error/save confirmations, IME text entry, SaveData virtual-slot
lists, and SaveData percentage progress. LIST uses only guest-provided virtual directory identifiers and
an optional new-save item; prosper never exposes a host path or opens an arbitrary host file picker.
Progress uses a non-focusable utility window updated from the app's main thread, so the title keeps
presenting and retains keyboard/controller focus.

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
