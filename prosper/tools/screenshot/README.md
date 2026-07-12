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

Built automatically wherever Vulkan + zlib are available (same block as `boot_trace`):

```bash
cmake -S prosper -B build -DPROSPER_APP=ON      # or any config that finds Vulkan
cmake --build build --target screenshot
```

## Usage

```
screenshot <app0-dir> [--every N] [--count M] [--out DIR] [--timeout SECS]
           [--warmup-seconds S] [--warmup-submits N]
           [--manifest PATH | --no-manifest]
           [--min-distinct-frames N] [--max-stale-seconds S]
           [--require-composited-frame] [--min-present-count N]
           [--min-frame-seq N] [--require-crc32 N]
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
| `--manifest PATH` | `<out>/<run>.jsonl` | Write the machine-readable capture manifest here |
| `--no-manifest` | off | Disable the default JSONL sidecar |
| `--min-distinct-frames N` | 0 | Fail unless at least N distinct source publications were captured |
| `--max-stale-seconds S` | unset | Fail if one source publication is reused longer than S seconds |
| `--require-composited-frame` | off | Fail if every PNG came from raw guest scanout fallback |
| `--min-present-count N` | 0 | Fail unless a captured sample reaches guest flip N |
| `--min-frame-seq N` | 0 | Fail unless a captured sample reaches rendered-frame N |
| `--require-crc32 N` | unset | Fail unless a sample has this RGBA CRC32 (decimal or `0xHEX`) |

Only the game is required; everything else has a sane default.
Directory-creation and PNG write failures include the failing path and operating-system error.

Every normal run writes a JSONL manifest beside its PNGs. Each sample records the atomic present-source
identity, guest flip count, rendered-frame sequence, dimensions, CRC32, capture source, elapsed time,
input route, and whether the source advanced or was stale. A final summary records distinct-frame and
maximum-stale metrics plus the exit status. Manifests flush after every sample so a killed run retains
usable evidence. `--no-manifest` preserves the old PNG-only behavior.

Assertions preserve every PNG and the manifest, print the concrete failed condition, and exit nonzero.
This lets an automated progression run distinguish "120 files written" from "120 advancing frames" or
"the requested checkpoint was reached." A timeout or incomplete screenshot count is also a failure.

Warmup is useful when llvmpipe makes a frame-counted startup take minutes. The guest and GPU command
decoder continue at native speed while Vulkan work is skipped; normal screenshots begin once warmup
ends. During warmup, both rendered frames and the raw-scanout fallback are suppressed from capture, so
diagnostic target/resource overrides can preserve producers without saving early frames (#588). The two
warmup gates are additive when both are supplied. `--timeout` covers
the entire run, including warmup.

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
`PROSPER_GUEST_FS=1` and `PROSPER_GUEST_ARGS=-force-gfx-direct` (Unity/Messenger recipe) if unset. For
other titles set the appropriate env first, e.g. a UE4 title:

```bash
PROSPER_GUEST_ARGS= PROSPER_NULL_PAGE=1 screenshot /path/PPSA01885-app0
```

## Example

```bash
# 30 shots, one every 60 rendered frames, into ./shots
screenshot /mnt/c/.../PPSA24651-app0 --out shots

# Skip Dead Cells' submit-heavy startup, then capture ten normal frames one second apart.
screenshot /mnt/c/.../PPSA15552-app0 --warmup-seconds 3 --seconds 1 --count 10 --out shots
```
