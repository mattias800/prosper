# Using the Linux release

The Linux archives contain the native `prosper-app` frontend: a Vulkan window with audio and
controller/keyboard input. They do not contain PS5 software, firmware, keys, or a game picker.
Supply your own legally obtained dump whose module segments are already unencrypted.

## Which download

| File | Use it when |
|---|---|
| `prosper-linux-x86_64.AppImage` | Normal desktop use. One file, no installation. |
| `prosper-linux-x86_64.tar.gz` | The AppImage will not start, or you want the tree on disk. |

Both carry the same binary and the same bundled libraries, and each ships a `.sha256` beside it:

```bash
sha256sum -c prosper-linux-x86_64.AppImage.sha256
```

The AppImage needs to mount itself, which requires FUSE. On a system without it, either extract the
same tree with `./prosper-linux-x86_64.AppImage --appimage-extract`, run it once with
`APPIMAGE_EXTRACT_AND_RUN=1`, or take the tarball.

## Requirements

- 64-bit Linux on an x86-64 CPU.
- **glibc 2.39 or newer** (Ubuntu 24.04, Debian 13, Fedora 40, and anything more recent). This is
  the one thing the archives cannot carry for you: no AppImage bundles libc, so the release is built
  on the oldest runner that is still supported and that build's glibc is the floor. Older
  distributions must build from source. `BUILD.txt` inside the archive records the floor actually
  measured for that build, which is often a little lower than the 2.39 stated here.
- A Vulkan 1.1-capable driver and a working loader (`libvulkan.so.1` plus your GPU's ICD — the
  `mesa-vulkan-drivers`, `vulkan-radeon`, `vulkan-intel`, or proprietary NVIDIA package for your
  distribution). The loader is deliberately **not** bundled, because it has to match the driver
  installed on your machine.
- The usual desktop libraries: X11 or Wayland, `libfontconfig`, `libfreetype`, `glib`, `libdrm`.
  These are excluded from the bundle on purpose — they must match your display stack.
- An unpacked title directory conventionally named `<TITLE_ID>-app0`, containing `eboot.bin` and the
  title's other files.

FFmpeg and libva **are** bundled, in `usr/lib` inside the archive. Their sonames differ on every
distribution (Ubuntu 24.04 ships `libavcodec.so.60`, Fedora 43 ships `libavcodec.so.62`), so a build
that relied on yours would not start anywhere else. SDL3 is statically linked into the binary.

## Start a title

With the AppImage:

```bash
chmod +x prosper-linux-x86_64.AppImage
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
  ./prosper-linux-x86_64.AppImage --dump ~/ps5/PPSA24651-app0
```

The tarball ships `start-prosper.sh`, which sets those switches for you and creates a `savedata`
directory beside itself:

```bash
tar -xzf prosper-linux-x86_64.tar.gz
cd prosper-linux-x86_64
./start-prosper.sh ~/ps5/PPSA24651-app0
```

Use a separate save location when testing a fresh save or keeping titles isolated:

```bash
./start-prosper.sh ~/ps5/PPSA24651-app0 --savedata-dir ~/prosper-saves/PPSA24651
```

`-force-gfx-direct` is the current Unity default. For a title that must not receive it, launch with
`--guest-args ''`.

Presentation defaults to FIFO vsync. Pass `--present-mode mailbox` for low-latency vsync or
`--present-mode immediate` to permit tearing; either optional mode falls back to FIFO when
unsupported.

## Point it at a library of titles

`prosper-app` can scan a directory of `<TITLE_ID>-app0` folders. This form prints what it finds and
exits without opening a window, which is also the quickest way to confirm the download runs at all:

```bash
./prosper-linux-x86_64.AppImage --list-games --games-dir ~/ps5
```

`--set-games-dir <path>` records the location for future launches.

## Input and exit

SDL3 automatically uses a connected controller as pad 0. Keyboard input is composed over it:

| Host input | Guest control |
|---|---|
| WASD or arrows | D-pad and left stick |
| J or Space | Cross |
| K | Square |
| L | Circle |
| I | Triangle |
| U / O | L1 / R1 |
| Y / H | L2 / R2 |
| Enter | Options |
| Pause or F10 | Pause/resume the guest and audio at a flip boundary |
| F11 or Alt+Enter | Toggle host-window fullscreen |
| F9 | Capture the current frame for offline replay |
| Escape | Exit |

Pause/F10 is host-owned and is not forwarded to guest keyboard input. Alt+Enter is consumed by the
host window and is not forwarded as the guest Options button. Closing the window also exits.
Shutdown currently terminates the process after flushing logs because cooperative guest-thread
teardown is not implemented yet.

## Smoke test and recordings

Test the window, Vulkan swapchain, and presentation path without a game:

```bash
./start-prosper.sh --test-pattern --frames 120
```

Record controller/keyboard input as a replay route while playing:

```bash
./start-prosper.sh ~/ps5/PPSA24651-app0 --record ./routes/session.pad
```

Add `--record-axis pad-read` when successful input-state reads are a more stable route clock than
display presentation; flip recording remains the default. Release the final held button before
exiting so its interval can be written. Runtime logs are printed to the terminal; include those
logs, the title ID, GPU/driver, distribution, and the release version from `BUILD.txt` when
reporting a problem.

## Compatibility

prosper is an experimental compatibility layer, not a general-purpose PS5 runner. Titles reach
different milestones and regressions are possible. See the repository's
[`COMPATIBILITY.md`](https://github.com/mattias800/prosper/blob/master/COMPATIBILITY.md) for the
tested state of each title. No compatibility is implied merely because a dump boots.
