# Building and running prosper

Everything here is the mechanical part: dependencies, configure/build/test commands per platform,
and how to launch a dump. For what prosper is, see the [README](README.md); for the rules a change
is held to, [`CLAUDE.md`](CLAUDE.md) and [`docs/VERIFICATION.md`](prosper/docs/VERIFICATION.md).

CI builds and tests Linux, Windows/MinGW and macOS on every push and pull request, and runs host
ASan/UBSan and TSan jobs. Those are the supported configurations.

The live graphics backend, the standalone compute backend and the app require **Vulkan 1.4** headers
and a 1.4 device; older loaders and devices are rejected with a diagnostic. Ubuntu 24.04's system
headers and loader are 1.3, so Linux CI installs pinned Khronos 1.4.341 headers and loader through
`.github/actions/setup-vulkan`. Capability policy and the measurements behind the floor are in
[`docs/VULKAN_RUNTIME.md`](prosper/docs/VULKAN_RUNTIME.md).

## Linux (primary)

Requires a C++20 compiler, CMake and Ninja. A Vulkan loader is needed for the graphics tests (the
headless path uses the `llvmpipe` software ICD). `spirv-tools` provides `spirv-val`, which the
`spv_validate` test requires. The AvPlayer backend uses FFmpeg for demux and VA-API for hardware
video decode.

```sh
sudo apt install ninja-build pkg-config libvulkan-dev mesa-vulkan-drivers \
  vulkan-validationlayers spirv-tools libavformat-dev libavcodec-dev libavutil-dev \
  libswresample-dev libswscale-dev libva-dev
```

That gives the distribution's 1.3 headers, which build and run the test suite. For the live
backends and the app, supply 1.4 headers and loader as the CI action above does.

```sh
cmake -S prosper -B prosper/build-linux -G Ninja
cmake --build prosper/build-linux
ctest --test-dir prosper/build-linux --no-tests=error   # unit + boot + Vulkan-execution tests
```

Add `-DPROSPER_APP=ON -DPROSPER_AUDIO_SDL3=ON -DPROSPER_PAD_SDL3=ON` for the windowed frontend with
audio and controllers; CMake fetches SDL3 when it is not installed. Video source-open requires a
working VA-API render device by default; select a non-default node with
`PROSPER_AVP_VAAPI_DEVICE=/dev/dri/renderD129`. `PROSPER_AVP_ALLOW_SOFTWARE=1` is an explicit
diagnostic fallback, not normal playback.

## Windows

The supported toolchain is 64-bit MinGW-w64 UCRT. In an MSYS2 UCRT64 shell, the headless core —
no SDL, no Vulkan, and the configuration CI runs on every push:

```sh
pacman -S --needed git mingw-w64-ucrt-x86_64-{gcc,cmake,ninja,spirv-tools}
cmake -S prosper -B prosper/build-windows-core -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_DISABLE_FIND_PACKAGE_Vulkan=TRUE
cmake --build prosper/build-windows-core
ctest --test-dir prosper/build-windows-core --output-on-failure --no-tests=error
```

For the windowed app, add the Vulkan packages and enable the SDL3 backends:

```sh
pacman -S --needed mingw-w64-ucrt-x86_64-{vulkan-headers,vulkan-loader}
cmake -S prosper -B prosper/build-windows-app -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPROSPER_APP=ON \
  -DPROSPER_AUDIO_SDL3=ON -DPROSPER_PAD_SDL3=ON
cmake --build prosper/build-windows-app --target prosper-app
./prosper/build-windows-app/prosper-app.exe --test-pattern --frames 120
```

From native PowerShell, the repository launcher does configure, build and launch in one command
against WinLibs or MSYS2 MinGW plus an installed Vulkan SDK:

```powershell
.\prosper\scripts\run-windows.ps1 .\PPSA24651-app0
```

The full native build, screenshot and diagnostic recipe is in
[`WINDOWS_PORT_HANDOFF.md`](prosper/docs/WINDOWS_PORT_HANDOFF.md).

On a Windows host, drive **git** from PowerShell and use WSL only for cmake and running: a worktree
created from Windows stores a Windows-path gitdir link that WSL's git cannot resolve.

## macOS

Built as x86_64 and tested under Rosetta 2 — the guest's code is x86-64, so there is no ARM path.
This exercises the Darwin guest-execution substrate, not just a compile.

```sh
brew install ninja spirv-tools
softwareupdate --install-rosetta --agree-to-license
cmake -S prosper -B prosper/build-macos -G Ninja \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_DISABLE_FIND_PACKAGE_Vulkan=TRUE
cmake --build prosper/build-macos
ctest --test-dir prosper/build-macos --output-on-failure --no-tests=error
```

The SDL3 + MoltenVK app needs a universal driver (Homebrew's is arm64-only);
`prosper/scripts/fetch-macos-vulkan.sh` fetches one, and `-DPROSPER_MACOS_MOLTENVK=<path>` points the
build at it. The app's Vulkan 1.4 floor applies here too: a MoltenVK device exposing less than 1.4 is
rejected. The CI job disables Vulkan and runs the headless core and substrate tests only. Details and
current limits: [`docs/PORTING.md`](prosper/docs/PORTING.md).

## Testing notes

- **`--no-tests=error` is not optional when you quote a result.** Plain `ctest` exits 0 on a build
  directory with nothing registered, so "no tests ran" and "everything passed" are otherwise the
  same exit code. Quote the discovered test *count* alongside the exit code — a count is
  falsifiable, an exit code alone is not.
- **`spv_validate` fails rather than skips when `spirv-val` is missing.** It is the strict SPIR-V
  gate, and silently skipping is how a real defect once shipped.
- **A dump other than `PPSA24651` skips three cases rather than failing them.** Configuring
  `-DGAME_DUMP=<DUMP_ROOT>/<TITLE_ID>-app0` at another title skips the cases that assert against
  *The Messenger's* own bytes, so read the skip count and not only the exit code.
- **Snapshot guards are a release-time regression inventory, not a per-commit gate.** See
  [`tools/snapshot/AGENTS.md`](prosper/tools/snapshot/AGENTS.md).

A single tool can be built without configuring the project:

```sh
g++ -O2 -std=c++20 prosper/tools/self_dump/self_dump.cpp -o self_dump
./self_dump <DUMP_ROOT>/PPSA24651-app0/eboot.bin [--symbols]
```

## Editor / LSP

CMake exports `compile_commands.json` into the build directory; a symlink at the repository root
points clangd at it. Each worktree is its own source tree and needs its own configure, or its files
resolve no includes.

```sh
cmake -S prosper -B prosper/build-linux
ln -sf prosper/build-linux/compile_commands.json compile_commands.json
clangd --check=<file> --compile-commands-dir=.     # exit 0 = parsed
```

## Running a dump

You supply your own legally-obtained dump; nothing in this repository or its releases contains
games, firmware or keys.

A `v*` tag publishes desktop archives on the
[releases page](https://github.com/mattias800/prosper/releases) —
`prosper-linux-x86_64.AppImage`, `prosper-linux-x86_64.tar.gz` and `prosper-windows-x64.zip`, each
with a `.sha256`. Take the AppImage for normal Linux desktop use and the tarball when the AppImage
runtime cannot start (it needs FUSE). Each archive ships a launcher that supplies the guest
environment, creates local save data and enables the window, audio and controller:

```bash
tar -xzf prosper-linux-x86_64.tar.gz && cd prosper-linux-x86_64
./start-prosper.sh ~/ps5/PPSA24651-app0
```

```powershell
./start-prosper.ps1 'D:/PS5/PPSA24651-app0'
```

The Linux archives carry FFmpeg, libva and the Vulkan loader because their sonames differ per
distribution, and link SDL3 statically. What you must supply is **glibc 2.39 or newer** and your
GPU's Vulkan driver. `--list-games --games-dir ~/ps5` prints the titles it can see and exits without
opening a window, which is the quickest way to confirm a download runs.

There is no game picker or settings UI. Requirements, keyboard mapping, save-data selection,
recordings and troubleshooting: [`LINUX_RELEASE.md`](prosper/docs/LINUX_RELEASE.md),
[`WINDOWS_RELEASE.md`](prosper/docs/WINDOWS_RELEASE.md), and
[`packaging/linux/`](prosper/packaging/linux/README.md) for how the archives are built.

Expect the milestone recorded in [`COMPATIBILITY.md`](COMPATIBILITY.md) for a given title and
nothing beyond it.

## Building from a source checkout, in a worktree

Several agents and the maintainer run this repository concurrently, so the main checkout and its
build directory are contended. Work in your own worktree with its own build directory, and remove
it when your branch merges — `prosper/tools/worktree_reclaim.py` reports what is reclaimable.
See `CLAUDE.md` for the full protocol, including why run artifacts must not go in `/tmp`.
