# screenshot

Run a game and capture a PNG every N rendered frames until M screenshots, then exit. Handy for
snapshot-testing a run and for eyeballing progression across a boot.

Reuses the shared boot path (`boot_program`) and the shared live renderer (`frontends/shared`), so
the game boots and composites exactly as `boot_trace` / `prosper-app` do — this tool just samples the
present layer periodically and writes PNGs.

Readback prefers the latest composited renderer frame. Before one exists, a flipped guest display
buffer is captured through the present layer's raw-scanout fallback, so the tool records real black or
partial early output instead of waiting indefinitely for a Vulkan draw.

## Build

Built automatically with the live Vulkan renderer. Unix uses zlib; native Windows uses the built-in
Windows Imaging Component PNG encoder and needs no zlib package:

```bash
cmake -S prosper -B build -DPROSPER_APP=ON      # or any config that finds Vulkan
cmake --build build --target screenshot
```

The executable is `screenshot.exe` in a native Windows build. Commands and environment variables are
otherwise identical in PowerShell; use `$env:NAME = 'value'` to set each variable.

## Usage

```
screenshot <app0-dir> [--every N] [--count M] [--out DIR] [--timeout SECS]
           [--warmup-seconds S] [--warmup-submits N]
           [--render-every N] [--render-every-for-seconds S]
           [--manifest PATH | --no-manifest]
           [--min-distinct-frames N] [--max-stale-seconds S]
           [--min-pixel-distinct-frames N] [--max-pixel-stale-seconds S]
           [--require-composited-frame] [--min-present-count N]
           [--min-frame-seq N] [--require-crc32 N] [--allow-guest-fault]
           [--no-stop-after-guest-fault] [--guest-fault-settle-seconds S]
```

| Option | Default | Meaning |
|--------|---------|---------|
| `<app0-dir>` | **required** | Game dump root, e.g. `.../PPSA24651-app0` |
| `--every N` | 60 | Rendered frames between screenshots |
| `--seconds S` | — | Wall-clock seconds between screenshots (overrides `--every`; e.g. `--seconds 1` = one per second) |
| `--count M` | 30 | Number of screenshots, then exit |
| `--out DIR` | `.` | Output directory; missing directories and parents are created |
| `--timeout S` | 900 | Give up after S seconds if the game isn't rendering enough (0 = no limit) |
| `--warmup-seconds S` | 0 | Advance the guest for S seconds without synchronous Vulkan rendering |
| `--warmup-submits N` | 0 | Advance without rendering until GPU submit N |
| `--render-every N` | 1 | Render every Nth draw-carrying submit during the sampling phase |
| `--render-every-for-seconds S` | unset | After S seconds, stop sampling and render every submit to rebuild temporal history |
| `--manifest PATH` | `<out>/<run>.jsonl` | Write the machine-readable capture manifest here |
| `--no-manifest` | off | Disable the default JSONL sidecar |
| `--min-distinct-frames N` | 0 | Fail unless at least N distinct source publications were captured |
| `--max-stale-seconds S` | unset | Fail if one source publication is reused longer than S seconds |
| `--min-pixel-distinct-frames N` | 0 | Fail unless at least N samples differ from the preceding PNG |
| `--max-pixel-stale-seconds S` | unset | Fail if identical pixels persist longer than S seconds, even across new publications |
| `--require-composited-frame` | off | Fail unless prosper composited at least one PNG — a raw guest-scanout fallback or a republished guest scanout does not count |
| `--min-present-count N` | 0 | Fail unless a captured sample reaches guest flip N |
| `--min-frame-seq N` | 0 | Fail unless a captured sample reaches rendered-frame N |
| `--require-crc32 N` | unset | Fail unless a sample has this RGBA CRC32 (decimal or `0xHEX`) |
| `--allow-guest-fault` | off | Do not fail the run when the guest's primary thread dies (deliberate fault-reproduction routes) |
| `--no-stop-after-guest-fault` | off | Keep sampling the full request after the guest's primary thread dies |
| `--guest-fault-settle-seconds S` | 1 | Quiescence required before that early stop: the guest must be dead **and** the present layer silent this long |

Only the game is required; everything else has a sane default.
Directory-creation and PNG write failures include the failing path and operating-system error.

Every normal run writes a JSONL manifest beside its PNGs. Each sample records the atomic present-source
identity, guest flip count, rendered-frame sequence, dimensions, CRC32, distinct RGB color count,
non-black pixel count, a 16x9 luminance signature, and standard 64-bit average/difference hashes,
capture source, elapsed time, input route, and whether the source advanced or was stale. `source` is
one of `composited` (prosper rendered it), `guest_scanout` (prosper rendered nothing for that flip
and republished the guest's own display buffer — see `docs/GRAPHICS.md`), or `raw_scanout` (read
straight out of guest memory before any renderer frame existed). The color
count and signatures are computed from the visible presented RGBA result (including alpha) and support scene-collapse,
coverage, and SSIM likeness guards without decoding the PNG again. The hashes and CRC32 remain useful
diagnostics, but are not sufficient likeness oracles by themselves. A final summary records distinct-frame and
maximum-stale metrics plus the exit status. Manifests flush after every sample so a killed run retains
usable evidence. `--no-manifest` preserves the old PNG-only behavior.

Source progression and pixel progression are reported separately. A game can publish a new renderer
frame every second while drawing the same image, so `source-distinct` alone is not evidence of visible
progress. Use the pixel assertions for loading screens, frozen cinematics, and other visual checkpoints.

Assertions preserve every PNG and the manifest, print the concrete failed condition, and exit nonzero.
This lets an automated progression run distinguish "120 files written" from "120 advancing frames" or
"the requested checkpoint was reached." A timeout or incomplete screenshot count is also a failure.

## The guest dying is a failure of the run

If the guest's **primary thread** dies during capture, the run reports `status=GUEST-FAULT` and exits
nonzero even when every capture assertion passed. Every PNG the tool managed to sample is still
written — frames from a crashing boot are real evidence — but the verdict says so, because after the
fault the sampler is re-photographing one stale frame and nothing in an exit status used to reveal
that (#2007: a title that crashed 0.4 s into boot produced 25 identical PNGs and `status=ok`).

- The summary line carries `guest=running|returned|faulted`, and the manifest summary carries
  `guest_state`, `guest_kind`, `guest_detail`, `guest_fault_rip`, `guest_fault_addr` and `status`,
  so a batch consumer can filter without re-reading the run log.
- `--allow-guest-fault` is for routes that *intend* to sample a crashing title. It restores exit 0
  and reports `status=GUEST-FAULT-ALLOWED` — never `ok` — so the fault stays visible in the record,
  and it does not disarm the `--min-*` / `--require-*` / `--max-*` assertions. (It is the flag that
  makes the early stop below observable at all: without it a fault is exit 1 either way.)
- A guest whose entry **returns** (a title exiting normally) is reported as `guest=returned` and is
  not a failure on its own; a run cut short by it still trips the saved/requested assertion.

Scope: this observes the thread `run_entry` entered. A *worker* thread's fatal fault already
`_exit(90)`s the whole process on Linux, which every caller that checks an exit status can see; the
primary thread was the one that could die silently.

### Sampling stops once nothing new can arrive

The verdict was only half of it: the loop used to keep running to the full `--seconds`/`--count`/
`--timeout` after the guest had died, writing one identical PNG per interval. On the `PPSA26414`
reproduction that was 24 byte-identical PNGs over 24 s (#2584).

Sampling now stops when **both** hold: the guest's primary thread is dead, *and* the present layer
has published nothing for `--guest-fault-settle-seconds` (1 s by default). The second condition is
not redundant — `GuestOutcome` covers only the thread `run_entry` entered, so other guest threads and
a renderer backlog can still publish frames after it dies. A run that keeps producing frames keeps
sampling them, so the stop cannot silently truncate evidence; it can only remove a tail that is
already known-identical.

- **Every sample already taken is kept**, and the run finishes normally: same verdict, same
  manifest, same exit status. The assertions are *not* untouched, and three bullets below state
  exactly which: one assertion is excused, two disarm the stop outright, and the rest cannot be
  affected in either direction.
- **The short artifact set explains itself.** The summary line reads
  `done: 3/25 screenshot(s) … stop=guest-fault … status=GUEST-FAULT`, and the manifest summary
  carries `stop_reason` (`request-satisfied` | `timeout` | `guest-fault`) beside the `saved` and
  `requested` it already had. Three PNGs where 25 were asked for must never read as a crashed
  harness.
- **The saved/requested assertion is excused only where the skipped samples would otherwise have
  been written** — wall-clock (`--seconds`) mode with at least one sample taken, where the sampler
  is due again regardless of the guest and the remainder are guaranteed duplicates. In frame
  (`--every`) mode a dead, quiet guest produces no new frame for `due` to fire on, so nothing more
  would have been written and the shortfall still fails.
- **`--max-stale-seconds` and `--max-pixel-stale-seconds` disarm the stop entirely.** Both are
  *maxima over the samples that were taken* — `CaptureTracker::observe` accumulates them, and only at
  a sample — so cutting the tail deletes precisely the quiet interval they exist to catch. A run with
  `--allow-guest-fault --max-pixel-stale-seconds 5` that fails over the full window (25 samples,
  ~24 s stale, exit 1 `FAILED`) would, if the stop were left armed, take one sample, report 0 s and
  exit 0. So arming either bound keeps the full-length loop, and the manifest run header records
  `"stop_after_guest_fault": false` — the policy the run actually used, not the one requested — with
  the bounds themselves in the same `assertions` object next to it. The run log says so too.
- **Nothing else the tool asserts can be affected.** Every remaining flag assertion is a floor —
  `--min-distinct-frames`, `--min-pixel-distinct-frames`, `--min-present-count`, `--min-frame-seq`,
  `--require-composited-frame`, `--require-crc32` — and fewer samples can only push a floor further
  from being satisfied. The one assertion that is not a flag, *manifest could not be written
  completely*, is an I/O check on the tool itself: a shortened run makes fewer sample writes, but the
  summary write and the `fclose` still happen, so an unwritable manifest is still caught. The stop
  can shorten a run; it cannot turn a failing run into a passing one.
- **A guest that *returns* does not trigger this.** A title exiting on its own is not a dead guest,
  and a run cut short by it must keep tripping the saved/requested assertion, as above.
- `--no-stop-after-guest-fault` restores the old full-length sampling, and both the flag and the
  settle window are recorded in the manifest's run header, so an archived artifact set says whether
  its tail was dropped or never produced.

Warmup is useful when llvmpipe makes a frame-counted startup take minutes. The guest and GPU command
decoder continue at native speed while Vulkan work is skipped; normal screenshots begin once warmup
ends. During warmup, both rendered frames and the raw-scanout fallback are suppressed from capture, so
diagnostic target/resource overrides can preserve producers without saving early frames (#588). The two
warmup gates are additive when both are supplied. `--timeout` covers
the entire run, including warmup.

The renderer prefers discrete, integrated, and virtual GPUs (in that order) over CPU Vulkan devices.
llvmpipe remains the automatic fallback when no usable hardware Vulkan device is available. The selected
graphics and compute devices are printed in the run log, so snapshot performance does not depend on the
loader's unspecified physical-device enumeration order.

Renderer cadence is useful when the sequence itself must include boot, so a complete warmup is not
appropriate. Sparse rendering can omit temporal producers and is therefore not a gameplay oracle by
itself. Pair `--render-every N` with `--render-every-for-seconds S` to accelerate a long intro, then
render every submit long enough to rebuild temporal render targets before the checkpoint frames.

**"Frames" = rendered frames** (composited images handed to the present layer), *not* guest flips —
the guest flips far faster than llvmpipe renders, so counting flips would bunch every shot into the
first second. Rendered frames spread the shots evenly across the run.

## Filenames

`<titleCode>_<runTimestamp>_<index>.png`, e.g. `PPSA24651_20260709-195505_00.png`.

- **titleCode** — the dump's basename with a trailing `-app0` removed.
- **runTimestamp** — `YYYYMMDD-HHMMSS` captured once at start, so every shot in a run shares it and a
  folder of many runs groups and sorts cleanly.
- **index** — zero-padded, `00`, `01`, …

## Guest environment

Reaching a rendering frame loop needs the render-frontier guest switches. This tool defaults
`PROSPER_GUEST_ARGS=-force-gfx-direct` (Unity/Messenger recipe) if unset. For other titles set the
appropriate env first, e.g. a UE4 title:

```bash
PROSPER_GUEST_ARGS= PROSPER_NULL_PAGE=1 screenshot /path/PPSA01885-app0
```

`PROSPER_GUEST_FS` is **no longer set on Linux or Windows** (#2098). Guest `%fs` TLS is **on by
default** there — Linux since #825, Windows since #624 — and the variable that exists is the
opt-OUT, `PROSPER_NO_GUEST_FS=1`. The name is read at exactly one place in the tree,
`guest_tls.cpp:46`, inside `#ifdef __APPLE__`, where it opts in to Rosetta trap-mode `%fs`
emulation; the tool still sets it there, because on macOS dropping it would turn trap mode off.

Do not carry it into a new recipe (#2095). Setting it on Linux or Windows is harmless at runtime and
misleading in the way that matters: it turns a default-on path into one people believe they are
enabling, so nobody checks it when guest TLS is the actual cause.

## Example

```bash
# 30 shots, one every 60 rendered frames, into ./shots
screenshot /mnt/c/.../PPSA24651-app0 --out shots

# Skip Dead Cells' submit-heavy startup, then capture ten normal frames one second apart.
screenshot /mnt/c/.../PPSA15552-app0 --warmup-seconds 3 --seconds 1 --count 10 --out shots
```
