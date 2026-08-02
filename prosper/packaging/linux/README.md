# Linux desktop release packaging

Builds the two artifacts that tagged releases publish alongside the Windows zip:

| Artifact | What it is |
|---|---|
| `prosper-linux-x86_64.AppImage` | Single self-contained file, the desktop default |
| `prosper-linux-x86_64.tar.gz` | The same AppDir as a plain tree |

The user-facing instructions are `prosper/docs/LINUX_RELEASE.md`, which is copied into both archives
as `README.md`.

## Running it

```bash
cmake -S prosper -B build-release-app -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPROSPER_APP=ON -DPROSPER_AUDIO_SDL3=ON -DPROSPER_PAD_SDL3=ON
cmake --build build-release-app --target prosper-app

prosper/packaging/linux/package-linux-app.sh --build-dir build-release-app --out-dir ~/prosper-rel
prosper/packaging/linux/verify-linux-app.sh  --out-dir ~/prosper-rel
```

`verify-linux-app.sh` is the gate, and CI runs it on **every pull request** — the packaging step is
deliberately not tag-gated, because a release path that only ever executes on a tag is one that is
discovered to be broken during a release. Only the artifact *upload* is gated (see below).

## Why the closure is bundled

`prosper-app` links FFmpeg and libva through pkg-config, and their sonames differ on every
distribution:

| | libavcodec | libavutil | libavformat | libswresample | libswscale |
|---|---|---|---|---|---|
| Ubuntu 24.04 | `.so.60` | `.so.58` | `.so.60` | `.so.4` | `.so.7` |
| Fedora 43 | `.so.62` | `.so.60` | `.so.62` | `.so.6` | `.so.9` |

A binary built against one set does not start against the other, so `linuxdeploy` copies the closure
into `usr/lib` and rewrites RUNPATH to `$ORIGIN/../lib`. SDL3 is **not** part of this: it is already
statically linked (`SDL_STATIC` in `prosper/CMakeLists.txt`), and `verify-linux-app.sh` asserts that
no `libSDL3` shared object appears, the same way the Windows job asserts no MinGW/SDL DLL.

## Why the runner is pinned

The CI job pins `ubuntu-24.04` rather than using `ubuntu-latest`. **No AppImage bundles glibc**, so
the runner's glibc is the floor for every user, and `ubuntu-latest` follows the newest image — it
would raise that floor with no diff and no failing check. `ubuntu-22.04` would give a lower floor
(2.35) but the Ubuntu 22 images begin brownout-based deprecation on 2026-09-17.

`glibc-floor.sh` measures the floor from the artifacts themselves and `verify-linux-app.sh` fails
when it exceeds `--max-glibc`, so the floor is a checked contract rather than a comment. The measured
value is recorded per build in `BUILD.txt`.

## Ruled out

- **A plain tarball of the built binary.** Not portable at all: the FFmpeg sonames above differ on
  every distribution, so an artifact built on the CI runner fails to start on Fedora, Arch, or any
  SteamOS/Bazzite derivative. Measured directly — `objdump -p prosper-app` on an Ubuntu 24.04 build
  names `libavcodec.so.60`, which does not exist on Fedora 43. #1782.
- **AppImage as a fix for glibc.** It is not one, and this was measured rather than assumed: an
  AppImage built on Fedora 43 fails on Ubuntu 24.04 with
  `libc.so.6: version 'GLIBC_2.43' not found`. AppImage solves *soname* portability, not libc
  versioning; only the build host's age does that. #1782.
- **Bundling the closure of a fully-featured distribution FFmpeg, on any host.** Bundling Fedora 43's
  closure (231 libraries — Samba, Kerberos, ICU, glib, OpenCL, rsvg) produces a binary that
  segfaults in `call_init` before `main`, inside a bundled system-integration library's ELF
  initializer. Three separate libraries were confirmed to do this, each found by gdb after excluding
  the previous one: `libcrypt.so.2`, then `libsystemd.so.0`, then `libtalloc.so.2` (Samba). So
  excluding them one at a time is unbounded — and the excluded ones cannot be assumed present on
  every user's machine either, which is what makes the exclusion route a dead end rather than a
  longer list. What
  makes the shipped configuration work is the *runner*, not an exclusion list: Ubuntu 24.04's FFmpeg
  pulls 116 libraries and none of that family. Packaging on a different host may well hit this
  again; `verify-linux-app.sh` executes the packaged binary precisely so it fails loudly rather than
  shipping. #1782.

## Upload gating

`#1462` removed the Windows package-and-upload from pull requests because `actions/upload-artifact`
hung and blocked a merge. That defect was in the upload, not in the packaging, so here only the
upload is conditional:

- tag pushes (`refs/tags/v*`) — always, this is what the `release` job publishes;
- pull requests — only when the PR carries the **`ci:artifacts`** label, so a branch build can be
  downloaded and opened by hand without giving every PR a release-sized artifact.

Adding the label does not re-run the workflow on its own (`labeled` is not among the default
`pull_request` event types), and re-running the job does not help either — a re-run replays the
**original** event payload, which still does not carry the label. Push a commit after applying it.

## Pinned tooling

`linuxdeploy` and `appimagetool` are pinned **by tag and by sha256**, and the script refuses to run
a tool whose content does not match. `continuous` is deliberately not used: it is a moving target
that would silently change what a release was built with.

`NO_STRIP=1` is set. linuxdeploy strips bundled libraries with the binutils `strip` inside its own
AppImage, which is older than some hosts' toolchains — on Fedora 43 every library fails with
`unknown type [0x13] section .relr.dyn` and the run exits non-zero. Distribution libraries arrive
stripped anyway. `prosper-app` itself is left unstripped so a crash report keeps its symbols.
