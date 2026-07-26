# Using the Windows release

The Windows archive contains the native `prosper-app` frontend: a Vulkan window with audio and
controller/keyboard input. It does not contain PS5 software, firmware, keys, or a game picker. Supply
your own legally obtained dump whose module segments are already unencrypted.

## Requirements

- 64-bit Windows 10 or Windows 11 on an x86-64 CPU.
- A Vulkan 1.1-capable graphics driver. Update the AMD, Intel, or NVIDIA driver if Windows cannot
  load `vulkan-1.dll` or the app reports that no Vulkan device is available.
- An unpacked title directory conventionally named `<TITLE_ID>-app0`, containing `eboot.bin` and the
  title's other files.

The archive is portable: MinGW, C++, and SDL3 runtimes are statically linked. Extract the complete
archive to a writable directory. Keep `start-prosper.ps1` beside `prosper-app.exe`.

## Start a title

Open PowerShell in the extracted directory and pass the title's `app0` directory:

```powershell
./start-prosper.ps1 'D:/PS5/PPSA24651-app0'
```

The launcher creates a `savedata` directory beside the executable. Use a separate location when
testing a fresh save or keeping titles isolated:

```powershell
./start-prosper.ps1 'D:/PS5/PPSA24651-app0' -SavedataDir 'D:/prosper-saves/PPSA24651'
```

If PowerShell blocks the unsigned local script, use this process-local policy; it does not change the
machine policy:

```powershell
powershell -ExecutionPolicy Bypass -File ./start-prosper.ps1 'D:/PS5/PPSA24651-app0'
```

The direct executable form is also supported. The environment is currently required because there
is no settings UI:

```powershell
$env:PROSPER_GUEST_FS = '1'
$env:PROSPER_GUEST_ARGS = '-force-gfx-direct'
$env:PROSPER_SAVEDATA_DIR = "$PWD/savedata"
./prosper-app.exe --dump 'D:/PS5/PPSA24651-app0'
```

`-force-gfx-direct` is the current Unity default. For a title that must not receive it, launch with
`-GuestArgs ''`.

Presentation defaults to FIFO vsync. Pass `-PresentMode mailbox` for low-latency vsync or
`-PresentMode immediate` to permit tearing; either optional mode falls back to FIFO when unsupported.

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
| Escape | Exit |

Pause/F10 is host-owned and is not forwarded to guest keyboard input. Alt+Enter is consumed by the
host window and is not forwarded as the guest Options button. Closing
the window also exits. Shutdown currently terminates the process after flushing logs because
cooperative guest-thread teardown is not implemented yet.

## Smoke test and recordings

Test the window, Vulkan swapchain, and presentation path without a game:

```powershell
./start-prosper.ps1 -TestPattern -Frames 120
```

Record controller/keyboard input as a replay route while playing:

```powershell
./start-prosper.ps1 'D:/PS5/PPSA24651-app0' -Record "$PWD/routes/session.pad"
```

Add `-RecordAxis pad-read` when successful input-state reads are a more stable route clock than
display presentation; flip recording remains the default. Release the final held button before
exiting so its interval can be written. Runtime logs are printed
to the PowerShell console; include those logs, the title ID, GPU/driver, and the release version when
reporting a problem.

## Compatibility

prosper is an experimental compatibility layer, not a general-purpose PS5 runner. Titles reach
different milestones and regressions are possible. See the repository's
[`COMPATIBILITY.md`](https://github.com/mattias800/ps5ys/blob/master/COMPATIBILITY.md) for the tested
state of each title. No compatibility is implied merely because a dump boots.
