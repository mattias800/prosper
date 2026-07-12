# GPU timeline

`PROSPER_GPU_TIMELINE=<path>.prgtl` records the guest's folded GPU-submit and VideoOut-present
boundaries without registering or invoking the Vulkan renderer. Use the cheap semantic index to find
the exact submit of interest, then select that stable submit number in a second run to create one
immutable replay capsule. Vulkan warmup and renderer sampling do not control the selection.

```bash
PROSPER_GUEST_FS=1 \
PROSPER_GUEST_ARGS=-force-gfx-direct \
PROSPER_GPU_TIMELINE=/tmp/dead-cells.prgtl \
PROSPER_CAPTURE_TITLE=PPSA15552 \
PROSPER_PAD_SCRIPT=@scripts/dead-cells/reach-first-gameplay.pad \
  ./build-linux/boot_trace /path/to/PPSA15552-app0

./build-linux/gpu_timeline /tmp/dead-cells.prgtl
./build-linux/gpu_timeline /tmp/dead-cells.prgtl --records
```

Capture one submit found in that index:

```bash
PROSPER_GPU_TIMELINE=/tmp/dead-cells-detail.prgtl \
PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT=18420 \
PROSPER_GPU_TIMELINE_CAPTURE=/tmp/dead-cells-submit-18420.prgcap \
  ./build-linux/boot_trace /path/to/PPSA15552-app0

./build-linux/gpu_timeline /tmp/dead-cells-detail.prgtl --records
./build-linux/gpu_replay --inspect-only /tmp/dead-cells-submit-18420.prgcap
./build-linux/gpu_replay /tmp/dead-cells-submit-18420.prgcap /tmp/replay.bmp
```

The summary reports duration, submit/present/draw/dispatch counts, rates, target extents, and whether
an incomplete final record was discarded. `--records` prints the globally ordered submit/present
index. Version-2 detail records link the selected submit to its `.prgcap` and report semantic versus
realized draw/dispatch counts, missing operations, unique shader/resource versions, and resource bytes.
Run metadata includes the revision, title, and input route when the corresponding capture environment
variables are set.

## Format contract

- Explicit little-endian versioned header; C++ struct layout is never serialized.
- Every record has a global sequence, monotonic timestamp, bounded payload, and FNV-1a checksum.
- The writer buffers output and flushes every 256 records. Normal process teardown closes it; a killed
  process can lose only its buffered tail.
- The reader recovers all complete records if the final record is truncated. A checksum error or bad
  framing in a complete record is corruption and fails inspection.
- Captures are local, may contain title-derived addresses/state, use the gitignored `.prgtl` suffix,
  and must never be committed.

## Current boundary

Version 2 retains backward support for version-1 semantic indexes and adds exactly one bounded detailed
submit per run. The linked version-5 `.prgcap` contains content-hashed/deduplicated shaders and resource
bytes, graphics and compute items, and the original mixed PM4 operation order. An operation whose shader
cannot be realized stays in the manifest with `realized=no`; inspection never silently treats a partial
submit as complete. A timeline-selected capsule has no live-output hash oracle unless the renderer also
produced one; replay reports `oracle=no` and renders without pretending an expected pixel hash exists.

The capture is selected by exact submit number, not yet by route checkpoint or present. It contains the
selected submit and temporal RTT surfaces currently resident in the renderer, but does not automatically
pull earlier producer submits. Automatic present-to-producer dependency closure is tracked by #595.
