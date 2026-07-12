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

Capture a selected submit and, optionally, its immediate predecessor from the same run:

```bash
PROSPER_GPU_TIMELINE=/tmp/dead-cells-detail.prgtl \
PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT=18420 \
PROSPER_GPU_TIMELINE_CAPTURE=/tmp/dead-cells-submit-18420.prgcap \
PROSPER_GPU_TIMELINE_CAPTURE_PREDECESSOR=/tmp/dead-cells-submit-18419.prgcap \
  ./build-linux/boot_trace /path/to/PPSA15552-app0

./build-linux/gpu_timeline /tmp/dead-cells-detail.prgtl --records
./build-linux/gpu_replay --inspect-only /tmp/dead-cells-submit-18420.prgcap
./build-linux/gpu_replay --graph /tmp/dead-cells-submit-18420.prgcap
./build-linux/gpu_replay --graph-json /tmp/dead-cells-graph.json /tmp/dead-cells-submit-18420.prgcap
./build-linux/gpu_replay /tmp/dead-cells-submit-18420.prgcap /tmp/replay.bmp
./build-linux/gpu_replay --prepend /tmp/dead-cells-submit-18419.prgcap \
  /tmp/dead-cells-submit-18420.prgcap /tmp/replay-with-producer.bmp
```

Capture a bounded ordered window directly into one deduplicated bundle:

```bash
PROSPER_GPU_TIMELINE=/tmp/dead-cells-bundle.prgtl \
PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT=18420 \
PROSPER_GPU_TIMELINE_CAPTURE=/tmp/dead-cells-submit-18420.prgcap \
PROSPER_GPU_TIMELINE_CAPTURE_BUNDLE=/tmp/dead-cells-depth16.prgbundle \
PROSPER_GPU_TIMELINE_CAPTURE_DEPTH=16 \
PROSPER_GPU_TIMELINE_CAPTURE_MAX_UNIQUE_MB=1024 \
PROSPER_GPU_TIMELINE_CAPTURE_WHEN_TARGET_DIM=642x362 \
PROSPER_GPU_TIMELINE_CAPTURE_MIN_DRAWS=80 \
PROSPER_GPU_TIMELINE_EXIT_AFTER_CAPTURE=1 \
  ./build-linux/boot_trace /path/to/PPSA15552-app0

./build-linux/gpu_replay --bundle /tmp/dead-cells-depth16.prgbundle /tmp/closure.bmp
```

See [`../gpu_replay/README.md`](../gpu_replay/README.md) for graph interpretation, pixel-oracle
semantics, draw isolation, resource/shader extraction, and the distinction between draw and operation indices.

The summary reports duration, submit/present/draw/dispatch counts, rates, target extents, and whether
an incomplete final record was discarded. `--records` prints the globally ordered submit/present
index. Version-2+ detail records link the selected submit to its `.prgcap` and report semantic versus
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

Version 4 remains backward-compatible with version-1 through version-3 indexes. For every selected temporal
image leaf it adds an online full-run target lifetime: earliest/latest graphics writer, matching writes and
submits, retained-window start, independent lifetime/window truncation flags, and the earliest writer's raw
clear words, color-control mode, target mask, and format. Lifetime aggregation is keyed by target range, so
memory scales with distinct surfaces rather than draw count.

Version 3 added bounded latest-writer history. When detailed capture is
enabled, it retains lightweight target-writer summaries for the previous 64 submits and records the
latest same-run writer identity for every temporal image leaf in the selected capsule. Set
`PROSPER_GPU_TIMELINE_HISTORY=N` to change that bounded window (maximum 65536). A producer record includes
the consumer submit/operation/range, the in-submit future writer, and either the matched prior graphics
submit/draw/PM4 order/target extent or an explicit unresolved result.

Version 2 added exactly one bounded detailed submit per run. The linked version-5 `.prgcap` contains
content-hashed/deduplicated shaders and resource
bytes, graphics and compute items, and the original mixed PM4 operation order. An operation whose shader
cannot be realized stays in the manifest with `realized=no`; inspection never silently treats a partial
submit as complete. A timeline-selected capsule has no live-output hash oracle unless the renderer also
produced one; replay reports `oracle=no` and renders without pretending an expected pixel hash exists.

The capture is selected by exact submit number, not yet by route checkpoint or present. It contains the
selected submit and temporal RTT surfaces currently resident in the renderer, but does not automatically
pull earlier producer submits. Automatic present-to-producer dependency closure is tracked by #595.
`gpu_replay --graph` resolves dependencies inside the selected submit and reports the remaining external
versions; a `future-writer` on an external leaf is the characteristic temporal read-before-write case that
needs the earlier version of the same logical surface.

Producer identity records do not contain producer-time resource bytes. Retaining an old `GpuState` and
materializing it after the selected submit would read mutable guest memory too late and create false
evidence. `PROSPER_GPU_TIMELINE_CAPTURE_PREDECESSOR=<path>` instead captures exactly submit `N-1` at
producer time in the same run and links both capsules with ordinary detail records. It is deliberately a
one-level closure probe; if the predecessor graph has temporal leaves, capture must recurse further.

For a recursive probe, `PROSPER_GPU_TIMELINE_CAPTURE_BUNDLE=<path>` captures the selected submit and
`PROSPER_GPU_TIMELINE_CAPTURE_DEPTH=N` contiguous submits ending there (2..4096). Bundle v2 captures at
producer time and stores small submit manifests plus exact resource versions in a global content-defined,
checksummed chunk dictionary; no retained `GpuState` or standalone predecessor files are used. Equal content
is reused only after byte comparison, never from guest address identity. Version-1 bundles remain readable.
`PROSPER_GPU_TIMELINE_CAPTURE_MAX_UNIQUE_MB=N` bounds unique chunk
data to 64..4096 MiB (default 1024) and aborts bundle installation if exhausted. The selected standalone
`.prgcap` remains the graph/oracle reference and is currently required with the bundle.

`PROSPER_GPU_TIMELINE_CAPTURE_TARGET_DIM=WxH` restricts predecessor bundle submits to graphics draws for
matching target extents while leaving the selected consumer complete. This is a capture-volume diagnostic,
not renderer substitution. It may still be expensive when relevant passes reference large shared assets;
unique-byte budget enforcement remains authoritative.

`PROSPER_GPU_TIMELINE_EXIT_AFTER_CAPTURE=1` terminates the process only after the selected standalone
capsule and any requested bundle have been installed successfully. Use it for long captures instead of a
wall-clock timeout; capture failure or budget exhaustion leaves the process running for diagnostics.

`PROSPER_GPU_TIMELINE_CAPTURE_WHEN_TARGET_DIM=WxH` changes the configured submit into a lower endpoint
bound: the first later submit with a draw targeting that extent is selected. Optionally require a semantic
draw count with `PROSPER_GPU_TIMELINE_CAPTURE_MIN_DRAWS=N`. This keeps a long synchronous capture from
selecting the wrong scene when it perturbs wall-clock pacing. A requested bundle still begins at
`CAPTURE_SUBMIT - CAPTURE_DEPTH + 1`; if the endpoint moves beyond that window, predecessor manifests roll
forward so the final bundle contains exactly the latest requested depth. The dictionary may retain content
seen by evicted manifests during capture, and the unique-byte budget remains authoritative. Finalization
compacts unreachable dictionary entries before installing the bundle.
