# GPU timeline

`PROSPER_GPU_TIMELINE=<path>.prgtl` records the guest's folded GPU-submit and VideoOut-present
boundaries without registering or invoking the Vulkan renderer. Use the cheap semantic index to find
the scene shape of interest, then select it with target, draw-count, and draw-index predicates to create one
immutable replay capsule. Submit numbers are run-local. Vulkan warmup and renderer sampling do not control
the selection.

```bash
PROSPER_GUEST_FS=1 \
PROSPER_GUEST_ARGS=-force-gfx-direct \
PROSPER_GPU_TIMELINE=/tmp/dead-cells.prgtl \
PROSPER_CAPTURE_TITLE=PPSA15552 \
PROSPER_PAD_SCRIPT=@scripts/dead-cells/reach-first-gameplay.pad \
  ./build-linux/boot_trace /path/to/PPSA15552-app0

./build-linux/gpu_timeline /tmp/dead-cells.prgtl
./build-linux/gpu_timeline /tmp/dead-cells.prgtl --records
./build-linux/gpu_timeline /tmp/dead-cells.prgtl --depth-summary 642x362
./build-linux/gpu_timeline /tmp/dead-cells.prgtl --signatures 91:94 8
./build-linux/gpu_timeline /tmp/dead-cells.prgtl --select 636x420 77:85 91:94 8
```

Capture a selected submit and, optionally, its immediate predecessor from the same run:

For unusually large submits, add `PROSPER_GPU_CAPTURE_METADATA_ONLY=1` to retain the selected
shader/operation/descriptor metadata without copying resource bytes. This avoids a suspicious resource
footprint obscuring the checkpoint; use the resulting capsule for inspect, validation, or graphing,
not pixel replay. Full captures preflight a 512 MiB resource limit, configurable with
`PROSPER_GPU_CAPTURE_MAX_MB=1..3072`.

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
PROSPER_GPU_TIMELINE=/tmp/dead-cells-gameplay.prgtl \
PROSPER_PAD_SCRIPT=@scripts/dead-cells/reach-first-gameplay-capture.pad \
PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT=1 \
PROSPER_GPU_TIMELINE_CAPTURE=/tmp/dead-cells-gameplay.prgcap \
PROSPER_GPU_TIMELINE_CAPTURE_BUNDLE=/tmp/dead-cells-gameplay.prgbundle \
PROSPER_GPU_TIMELINE_CAPTURE_DEPTH=1000 \
PROSPER_GPU_TIMELINE_CAPTURE_MAX_UNIQUE_MB=1024 \
PROSPER_GPU_TIMELINE_CAPTURE_START_TARGET_DIM=642x362 \
PROSPER_GPU_TIMELINE_CAPTURE_START_TARGET_DRAW_INDEX=0:8 \
PROSPER_GPU_TIMELINE_CAPTURE_WHEN_TARGET_DIM=636x420 \
PROSPER_GPU_TIMELINE_CAPTURE_TARGET_DRAW_INDEX=77:85 \
PROSPER_GPU_TIMELINE_CAPTURE_MIN_DRAWS=91 \
PROSPER_GPU_TIMELINE_CAPTURE_MAX_DRAWS=94 \
PROSPER_GPU_TIMELINE_CAPTURE_MIN_DISPATCHES=8 \
PROSPER_GPU_TIMELINE_CAPTURE_MAX_DISPATCHES=8 \
PROSPER_GPU_TIMELINE_EXIT_AFTER_CAPTURE=1 \
  ./build-linux/boot_trace /path/to/PPSA15552-app0

./build-linux/gpu_replay --bundle /tmp/dead-cells-gameplay.prgbundle \
  --bundle-intermediate-through-target 642x362 /tmp/closure.bmp
```

See [`../gpu_replay/README.md`](../gpu_replay/README.md) for graph interpretation, pixel-oracle
semantics, draw isolation, resource/shader extraction, and the distinction between draw and operation indices.

The summary reports duration, submit/present/draw/dispatch counts, rates, target extents, and whether
an incomplete final record was discarded. `--records` prints the globally ordered submit/present
index, v6 target spans, and exact v8 ordered DMA records. `--signatures DRAWS DISPATCHES` groups target-span sequences for candidate
submit-count ranges; each range accepts `N` or `MIN:MAX`. `--select WxH DRAW_INDEX DRAWS DISPATCHES` applies
the same semantic predicate as live capture, reports its first/last match and total, and exits nonzero when no
submit matches. It reports an inconclusive error instead of treating a truncated target signature as a negative.
Version-2+ detail records link the selected submit to its `.prgcap` and report semantic versus
realized draw/dispatch counts, missing operations, unique shader/resource versions, and resource bytes.
Run metadata includes the revision, title, and input route when the corresponding capture environment
variables are set.

When MRT0-only target spans are insufficient, set `PROSPER_GPU_TIMELINE_MRT_SUBMIT=N` on a second
native-speed run. For timing-sensitive routes, use the semantic
`PROSPER_GPU_TIMELINE_MRT_MIN_DRAWS`, `MAX_DRAWS`, `MIN_DISPATCHES`, and `MAX_DISPATCHES` predicates
instead; only their first matching submit is logged. For every semantic draw in the selected submit,
the recorder prints all eight raw `CB_COLOR` addresses, formats, extents, target write masks, and
shader export masks. It also reports raw `CB_COLOR_CONTROL` and its mode, so a zero effective write
mask can be separated from color writes disabled by the global color mode. Use this to prove whether
a GPU-only input was produced through MRT1..7 before
proposing multi-attachment renderer work. The diagnostic does not realize shaders, copy resource
bytes, or invoke Vulkan. Exact submit numbers are run-local, so prefer predicates derived from prior
positive and nearby negative `.prgtl` samples when timing can move the endpoint.

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

Version 8 reads version-1 through version-7 indexes. It appends exact ordered `DMA_DATA` records to each
submit: source and destination guest identities, byte count, selectors, PM4 command order, and packet address.
Older indexes retain their historical count/incomplete signal and do not invent individual DMA operations.
Detailed current-version capture v14 also closes both DMA endpoint ranges into content-addressed blobs so
standalone replay can execute draw-to-DMA-to-draw or DMA-to-compute sequences at their original order.

Version 6 reads version-1 through version-5 indexes. It adds run-length encoded color-target extent spans over
the semantic draw sequence, excluding run-local addresses, so scene predicates can be discovered and tested
offline before a detailed rerun. Span storage is bounded and an incomplete signature is marked truncated.

Version 5 added a compact per-submit manifest of distinct depth/stencil surfaces without realizing shaders or
copying general resources: plane and HTILE bases, raw view/format/size/override programming, target extent,
draw/test/write/clear counts, and compare-op coverage. `gpu_timeline FILE --depth-summary [WxH]` groups
complete lifetimes and reports raw programming transitions. This is the fast first check for a stale persistent
DS cache identity.

Set `PROSPER_GPU_TIMELINE_DEPTH_HASH_DIM=WxH` only for a focused native run. Matching submits hash the readable
guest depth, stencil, and padded HTILE spans and record the latest overlapping graphics/compute/DMA/WRITE_DATA
writer. The summary reports distinct backing versions and transitions; unlike a full `.prgcap`, it does not copy
every shader/resource payload. This mode enabled writer provenance automatically and identified Dead Cells'
exact HTILE fill without the multi-minute full-bundle capture attempted first.

Version 4 added full-run temporal-image lifetime metadata. For every selected temporal
image leaf it adds an online full-run target lifetime: earliest/latest graphics writer, matching writes and
submits, retained-window start, independent lifetime/window truncation flags, and the earliest writer's raw
clear words, color-control mode, target mask, and format. Lifetime aggregation is keyed by target range, so
memory scales with distinct surfaces rather than draw count.

Version 3 added bounded latest-writer history. When detailed capture is
enabled, it retains lightweight target-writer summaries for the previous 64 submits and records the
latest same-run writer identity for every external texture or storage-image leaf in the selected capsule,
including inputs that are not overwritten later in the selected submit. Set
`PROSPER_GPU_TIMELINE_HISTORY=N` to change that bounded window (maximum 65536). A producer record includes
the consumer submit/operation/range, the in-submit future writer, and either the matched prior graphics
submit/draw/PM4 order/target extent or an explicit unresolved result.

Version 2 added exactly one bounded detailed submit per run. The linked current-version `.prgcap` contains
content-hashed/deduplicated shaders and resource
bytes, graphics and compute items, complete raw depth-surface programming, and the original mixed PM4 operation
order. An operation whose shader
cannot be realized stays in the manifest with `realized=no`; capture v7 also retains its bounded raw shader,
stage coverage, first rejected opcode/PC, decoded state, and failure reason. Inspection never silently treats a partial
submit as complete. A timeline-selected capsule has no live-output hash oracle unless the renderer also
produced one; replay reports `oracle=no` and renders without pretending an expected pixel hash exists.

Without a semantic endpoint predicate, capture uses the exact configured submit. With target/draw/dispatch
conditions, that submit becomes a lower bound and the first matching route checkpoint is selected. Present
selection is not implemented. A capsule contains the selected submit and temporal RTT surfaces currently
resident in the renderer, but does not automatically
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

`PROSPER_GPU_TIMELINE_CAPTURE_CHECKPOINT_EVERY=N` atomically writes the current rolling bundle after every N
captured predecessor submits (1..4096). Use it when the guest may crash before the semantic endpoint; the
checkpoint remains replayable, while a later successful endpoint still compacts and replaces it normally.

`PROSPER_GPU_TIMELINE_CAPTURE_TARGET_DIM=WxH` restricts predecessor bundle submits to graphics draws for
matching target extents while leaving the selected consumer complete. This is a capture-volume diagnostic,
not renderer substitution. It may still be expensive when relevant passes reference large shared assets;
unique-byte budget enforcement remains authoritative.

`PROSPER_GPU_TIMELINE_CAPTURE_START_TARGET_DIM=WxH` delays predecessor bundle capture until the first submit
that writes a target with that extent. Use it when native-speed lifetime evidence identifies the beginning of
the relevant surface family; earlier submits remain in the timeline but do not perturb progression or consume
bundle budget. The matching start submit is included.

`PROSPER_GPU_TIMELINE_CAPTURE_START_TARGET_DRAW_INDEX=MIN:MAX` additionally requires that the start extent
occur in the zero-based semantic draw-index window. Use it when an extent is shared by unrelated boot and
gameplay passes; it prevents the earlier pass from starting an expensive rolling capture. It requires
`PROSPER_GPU_TIMELINE_CAPTURE_START_TARGET_DIM` and does not change the independent endpoint selector.

`PROSPER_GPU_TIMELINE_EXIT_AFTER_CAPTURE=1` terminates the process after the selected standalone
capsule and any requested bundle have been installed successfully. Use it for long captures instead of a
wall-clock timeout. A selected-capture realization or write failure exits nonzero instead of waiting for
an outer watchdog; the runtime logs realization and total serialization time before exiting.

`PROSPER_GPU_TIMELINE_CAPTURE_WHEN_TARGET_DIM=WxH` changes the configured submit into a lower endpoint
bound: the first later submit with a draw targeting that extent is selected. Optionally require a semantic
draw-count window with `PROSPER_GPU_TIMELINE_CAPTURE_MIN_DRAWS=N` and
`PROSPER_GPU_TIMELINE_CAPTURE_MAX_DRAWS=N`. This keeps a long synchronous capture from
selecting the wrong scene when it perturbs wall-clock pacing. A requested bundle still begins at
`CAPTURE_SUBMIT - CAPTURE_DEPTH + 1`; if the endpoint moves beyond that window, predecessor manifests roll
forward so the final bundle contains exactly the latest requested depth. The dictionary may retain content
seen by evicted manifests during capture, and the unique-byte budget remains authoritative. Finalization
compacts unreachable dictionary entries before installing the bundle.

For a standalone capsule that reads its own prior-frame targets, `gpu_replay --warmup-repeats N CAPTURE OUTPUT`
executes N unmeasured copies into the persistent RTT/depth caches before the final replay. This is a temporal
convergence diagnostic, not evidence that the first native frame contained those prior versions.

`PROSPER_GPU_TIMELINE_CAPTURE_TARGET_DRAW_INDEX=MIN:MAX` restricts the matching target to a zero-based
semantic draw-index window. Establish the range across multiple desired captures and nearby negative samples;
realized replay indices can differ when earlier semantic draws are unrealized.
`PROSPER_GPU_TIMELINE_CAPTURE_MIN_DISPATCHES=N` and
`PROSPER_GPU_TIMELINE_CAPTURE_MAX_DISPATCHES=N` restrict the same checkpoint by semantic dispatch count.
