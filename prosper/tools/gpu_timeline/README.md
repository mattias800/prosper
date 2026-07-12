# GPU timeline

`PROSPER_GPU_TIMELINE=<path>.prgtl` records the guest's folded GPU-submit and VideoOut-present
boundaries without registering or invoking the Vulkan renderer. The initial semantic index is meant
for native-speed progression analysis and as the stable identity layer for later resource-version and
dependency-complete replay work (#594/#595).

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

The summary reports duration, submit/present/draw/dispatch counts, rates, target extents, and whether
an incomplete final record was discarded. `--records` prints the globally ordered submit/present
index. Run metadata includes the revision, title, and input route when the corresponding capture
environment variables are set.

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

Version 1 is a semantic index: submit numbers, PM4 order ranges, draw/dispatch counts, final color
target identity/extent, and flip boundaries. It does **not** yet contain shader/resource bytes or
enough state to render offline. The next #594 slice adds deduplicated immutable resource versions;
#595 uses those versions to build the selected present's producer dependency closure.
