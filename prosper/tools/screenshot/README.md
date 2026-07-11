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

Only the game is required; everything else has a sane default.
Directory-creation and PNG write failures include the failing path and operating-system error.

Warmup is useful when llvmpipe makes a frame-counted startup take minutes. The guest and GPU command
decoder continue at native speed while Vulkan work is skipped; normal screenshots begin once rendering
does. During warmup, the raw-scanout fallback is suppressed so the output folder does not fill with
misleading loading frames. The two warmup gates are additive when both are supplied. `--timeout` covers
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
