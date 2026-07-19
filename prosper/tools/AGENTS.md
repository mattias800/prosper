# AGENTS.md — prosper/tools

Developer/agent tooling. These are debugging and verification aids, not part of
the shipped runtime. Build them from `build-linux/` like everything else.

- **`verify-pr.ps1`** - author-owned PR verification orchestrator for the Windows+WSL development
  environment. Run it only from a clean, pushed PR head: `docs` records the diff check, `core` adds
  Linux and Windows build+ctest, and `renderer -Snapshot NAME` also runs the selected real-game guard
  with `boot_trace` and `screenshot` pinned to that run's selected Linux build directory.
  `-Pr N` posts the generated SHA-bound `AUTHOR VERIFICATION` record. Reviewers inspect its coverage
  but do not rerun it; see the mandatory merge policy in the root `CLAUDE.md`. Run
  `powershell -File tools/test-verify-pr.ps1` after changing the orchestrator; it probes exact-base
  pinning, WSL selection, rejected skip attempts, and untracked-source contamination.
- **`snapshot/`** - routed, multi-frame **rendering regression guard**. Run
  `python3 tools/snapshot/snapshot.py check` after any change that can affect
  rendered output (recompiler, AGC decode, render state, detile, present). It
  catches major scene collapse without treating subtle pixel changes as
  regressions. New or changed baselines require two-run image inspection; see
  `snapshot/AGENTS.md`. **Run this after touching the render path.**
- **`boot_trace/`** — boots a SELF/ELF game image through the loader + HLE and
  runs it, with the fault handler, GPU executor, and (under `PROSPER_RENDER`) the
  live Vulkan renderer. The main harness for exercising a real title headlessly.
- **`screenshot/`** — writes normal composited PNG sequences plus a JSONL evidence manifest. Use
  `--seconds 1` for wall-clock sampling, warmup or `--render-every N --render-every-for-seconds S`
  for slow software rendering,
  and the pixel-distinct/pixel-stale assertions when visible progression matters. Source publication
  counts alone do not prove that the image changed; see `screenshot/README.md`.
- **`self_dump/`** — parse a SELF/ELF and print its segment/program-header map, import NIDs, and
  export RVAs. Use `--find-symbol NID` for a focused import/export query.
- **`re/xref.py`** — find relative data pointers, direct references, runtime function-table
  writers, and indirect callers in a flattened guest module. See `re/README.md`.
- **`shader_histo/`** — histogram RDNA2 opcodes across a title's shaders.
- **`shader_inspect/`** — decode one raw `PROSPER_SHADER_DUMP` binary offline. It prints bounded
  instruction PCs, operands, raw words, signed branch immediates, and resolved branch targets so a
  failed shader's CFG can be mapped without hand-counting variable-length instructions.
- **`imgdump/`** — decode/dump a guest texture to an image for inspection.
- **`gpu_replay/`** — replay a local `PROSPER_GPU_CAPTURE` realized-submit capsule through the same
  Vulkan backend without booting the guest. Capsules include game shaders/resources, use `.prgcap`,
  are gitignored, and must never be committed. The tool exits non-zero on output-hash mismatch.
  See `gpu_replay/README.md` for inspection, validation, dependency graphs, oracle semantics, and extraction.
- **`gpu_timeline/`** — inspect a native-speed `.prgtl` submit/present index recorded with
  `PROSPER_GPU_TIMELINE=<path>`. It does not invoke Vulkan; use it to locate progression and producer
  windows before making an expensive realized `.prgcap`. Timeline files are gitignored and local-only.
- **`spv_validate/`** — `spirv-val` wrapper for recompiled SPIR-V.
- **`niddiag/`, `fetch_niddb.sh`** — NID (Sony symbol hash) resolution helpers.

Verification here is agentic-first (see `docs/VERIFICATION.md`): prefer a
programmatic check (ctest exit code, `spirv-val`, a snapshot hash) over eyeballing.

To drive any runner through a longer input route, set
`PROSPER_PAD_SCRIPT=@scripts/<title>/reach-<state>.pad`. Route files use the same
seconds/flip/pad-read syntax as inline scripts (`3:`, `f300:`, or `p1200:`), accept one entry per
line, `#` comments,
and explicit ranges such as `f300-340:cross`. Full-deflection stick actions use names such as
`left-stick-left` and can be combined with buttons using `+`. See `docs/INPUT_REPLAY.md`.
Set `PROSPER_PAD_RECORD=<path>` on any runner, or use `prosper-app --record <path>`, to capture the
final button stream in that format. Completed button intervals are flushed immediately; scripted stick
directions are supported for playback but are not yet emitted by the recorder.
Set `PROSPER_PAD_SCRIPT_LOG=1` to log each scripted state transition observed at a pad poll.
For long exploratory runs, add `PROSPER_PAD_SCRIPT_RELOAD=1` to live-reload an `@file` route while
preserving its original time/flip/read origin; append only future windows and confirm the reload log.
Wall-clock ranges can be skipped entirely when their duration is shorter than the interval between
polls, especially under synchronous software rendering; use poll-safe holds with neutral gaps,
flip-anchored ranges while presentation advances, or `pA-B:` ranges keyed to the pad-read index printed
by `PROSPER_PAD_SCRIPT_LOG=1`. Point entries use `PROSPER_PAD_FRAME_HOLD` or
`PROSPER_PAD_READ_HOLD` (both default 8) on their count axis.
Pad-read indices advance only for successful `scePadRead`/`scePadReadState` calls; controller metadata
queries and rejected reads do not consume pad-read entries. Seconds and flips keep their first-pad-poll
origin for compatibility with existing routes.

Capture one draw-carrying renderer invocation with:

```bash
PROSPER_GPU_CAPTURE=/tmp/messenger-level.prgcap PROSPER_GPU_CAPTURE_AT=0 \
  PROSPER_GPU_CAPTURE_MIN_DRAWS=30 \
  PROSPER_CAPTURE_REVISION=$(git rev-parse HEAD) \
  PROSPER_CAPTURE_TITLE=PPSA24651 <normal boot_trace command>
./build-linux/gpu_replay /tmp/messenger-level.prgcap /tmp/replayed.bmp
```

`PROSPER_GPU_CAPTURE_MIN_DRAWS`/`MAX_DRAWS` filter by realized item count; `PROSPER_GPU_CAPTURE_AT`
counts matching invocations that reach the registered renderer, after the normal `RENDER_EVERY`
sampling. Aim the live run near the target first; the capture itself writes once.
The live hook snapshots realized draws, compute dispatches, and the original mixed PM4 operation order
before executing the selected submit, then attaches the rendered pixel oracle afterward. An inspected
mixed capsule must report the same draw/compute/operation counts as the live timing line; `computes=0`
for a known mixed submit indicates an obsolete capture build, not proof that the barriers are unnecessary.
`PROSPER_GPU_CAPTURE_AFTER=N` ignores the first `N` renderer invocations before applying the draw-count
filters and `AT` counter. Pair it with `PROSPER_SUBMITLOG`/`PROSPER_RENDER_FIRST` when several early
scenes share the same draw count as a late target.
Resource bytes are preflighted before allocation and default to a 512 MiB total limit. Raise it with
`PROSPER_GPU_CAPTURE_MAX_MB=1..3072` only when a replayable capsule genuinely needs the data. For a
suspect descriptor or very large submit, set `PROSPER_GPU_CAPTURE_METADATA_ONLY=1`: the thin capsule
keeps shaders, operations, pipeline state, and resource descriptors for `--inspect-only`, `--validate`,
and `--graph`, but deliberately cannot render.
Set `PROSPER_CAPTURE_REVISION` explicitly in WSL worktrees: WSL Git cannot resolve their Windows-path
gitdir links, so the build-time fallback revision is `unknown` there.
`PROSPER_SUBMITLOG_DIM=WxH` prints the exact renderer invocation for any submit targeting that Gen5
surface extent, even while `PROSPER_RENDER_FIRST` skips Vulkan work. Use it to start a later render
window on a one-time offscreen producer rather than after the producer has already been lost.
`PROSPER_RENDER_TARGET_DIM=WxH` executes matching target submits even before `PROSPER_RENDER_FIRST`.
This preserves a one-time producer in the RTT cache while still skipping an expensive gap before its
late consumer.
`PROSPER_RENDER_RESOURCE_DIM=WxH` likewise executes submits that sample a matching image, allowing a
producer/consumer A/B without rendering every unrelated submit between them.
`PROSPER_RESOURCE_HASH_DIM=WxH` logs each matching sampled resource's raw guest hash, decoded/sample
hash, nonblack RGB/alpha occupancy, RTT-hit state, last compute/DMA writer, draw index, and PM4 order.
Add `PROSPER_DUMP_RESOURCE_VERSION=1` to write each distinct decoded version as a BMP under
`PROSPER_FRAME_DIR`; this is an inspection artifact, not a pixel oracle. `PROSPER_TARGET_STEP_HASH_DIM`
rerenders matching target passes by prefix and logs per-draw hashes plus dark/white/mean metrics;
`PROSPER_TARGET_STEP_HASH_MIN_DRAWS=N` bounds that intentionally expensive bisect.
`PROSPER_RENDER_DELAY_MS=N` skips synchronous Vulkan work for N milliseconds from the first submit while
the guest and command decoder advance. The `screenshot` frontend exposes this as `--warmup-seconds`, plus
`--warmup-submits` for `PROSPER_RENDER_FIRST`; use wall-clock warmup for progression captures and the exact
submit gate for repeatable renderer investigations.
`PROSPER_GPU_TIMELINE=<path>.prgtl` records every folded submit before renderer sampling plus every
VideoOut flip. `gpu_timeline <path> [--records]` inspects the checksummed index offline. Version 6 also records
compact target-extent spans; use `--signatures DRAWS DISPATCHES` to discover scene shapes and
`--select WxH DRAW_INDEX DRAWS DISPATCHES` to validate the live predicate. Recording is
independent of `PROSPER_RENDER_EVERY`. To materialize one exact indexed submit without rendering the
warmup, set `PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT=N` and
`PROSPER_GPU_TIMELINE_CAPTURE=<path>.prgcap` on a second run. Version-2 detail records link the capsule;
version-8 capsules deduplicate content-addressed shader/resource versions, retain complete raw depth-surface
programming, and preserve mixed draw/dispatch order plus explicit unrealized operations. Failed operations
also retain bounded raw stages and exact rejection/state summaries; inspect them with `gpu_replay --inspect-only`
and extract one with `--dump-failed-shader FAILURE:STAGE PATH`. Selection is
intentionally bounded to the consumer and an optional
immediate predecessor. Explicit-depth `.prgbundle` capture provides bounded recursive cross-submit closure;
automatic present-to-producer selection remains #595.
Timeline version 6 retains the version-5 sliding graphics-target window plus full-run aggregate lifetime metadata
when a detailed capture is requested. `PROSPER_GPU_TIMELINE_HISTORY=N` raises the window to at most 65536.
Producer records identify the latest overlapping prior submit/draw/PM4 order, earliest observed graphics
writer, write/submit counts, truncation, and raw first-writer clear/target state. Raw clear registers are
provenance, not proof of an implicit hardware clear. The timeline intentionally retains no delayed pointers
to mutable guest bytes.
Every current submit also records the v5 distinct DS plane/HTILE identities, raw view/format/size programming,
target extents, and test/write/clear counts. `gpu_timeline FILE --depth-summary [WxH]` groups their full lifetimes.
For a focused run, `PROSPER_GPU_TIMELINE_DEPTH_HASH_DIM=WxH` adds guest depth/stencil/HTILE hashes and latest
overlapping writer provenance without realizing general resources; use it before attempting a full bundle.
Set `PROSPER_GPU_TIMELINE_CAPTURE_PREDECESSOR=<path>.prgcap` to snapshot exact submit `N-1` at producer
time alongside selected submit `N`. Replay the pair with `gpu_replay --prepend producer consumer output`.
This is a one-level probe; graph the producer and recurse when it also reads a temporal version.
For bounded recursion, add `PROSPER_GPU_TIMELINE_CAPTURE_BUNDLE=<path>.prgbundle`,
`PROSPER_GPU_TIMELINE_CAPTURE_DEPTH=2..4096`, and optionally
`PROSPER_GPU_TIMELINE_CAPTURE_MAX_UNIQUE_MB=64..4096` (default 1024). `gpu_replay --bundle bundle output`
executes the ordered window and classifies each temporal frontier as included, seeded, or depth-bounded.
Bundles use content-defined chunks so shifted capture metadata does not defeat cross-submit deduplication.
`PROSPER_GPU_TIMELINE_CAPTURE_CHECKPOINT_EVERY=N` atomically installs the rolling bundle every N captured
predecessor submits, preserving the latest complete window if the guest crashes before the selected endpoint.
`PROSPER_GPU_TIMELINE_CAPTURE_TARGET_DIM=WxH` restricts predecessor captures to draws targeting that extent;
it is a size diagnostic, not a dependency proof. `gpu_replay --bundle-zero-boundary` supplies transparent
pixels to the oldest unseeded temporal leaves for an explicit A/B test and labels the synthetic seeds.
`PROSPER_GPU_TIMELINE_CAPTURE_START_TARGET_DIM=WxH` delays bundle capture until the first matching target
writer while timeline/lifetime recording continues, avoiding progression distortion from irrelevant submits.
`gpu_replay --bundle-tail N` skips older manifests without changing dictionary content; use it only when
lifetime evidence proves the suffix contains the target's beginning, then inspect its lower-bound frontier.
`gpu_replay --bundle-intermediate-through-target WxH` omits later passes from non-final submits only when
dependency evidence proves that the named target family is the sole temporal image frontier.
`gpu_replay --bundle-final-capsule PATH` exports the complete live color RTT cache plus exact valid planes from
the persistent Vulkan depth/stencil cache into a capture-v9 capsule, including base-level depth for 3D image
resources. Verify its standalone output hash against
the bundle before using it for rapid final-submit isolation. `--inspect-only` prints each DS seed's full cache
identity, format, independent validity flags, byte counts, and hashes. Captures v1-v8 remain readable without
invented DS state.
`gpu_replay --bundle-extract-submit N PATH` materializes one exact manifest for normal inspect/graph/validate
work without replaying the bundle.
`gpu_replay --warmup-repeats N CAPTURE OUTPUT` executes the same capsule N times into the persistent RTT/DS
caches before the measured replay, providing an explicit temporal-history convergence probe.
`gpu_replay --bundle-find-ds ADDR` scans compact manifests for guest depth/stencil use, writes, clears, compare
ops, and target extents without reconstructing resource payloads or invoking Vulkan.
`gpu_replay --bundle-ds-summary` groups every DS-active draw by complete captured identity/programming and
reports lifetime transitions manifest-only. Use `--legacy-htile-before-stencil` only for the preserved pre-v6
Dead Cells bundle whose allocation relationship is independently proven; current captures store real HTILE.
`gpu_replay --through-operation N` preserves the inclusive mixed graphics/compute prefix and is the preferred
final-composition bisect after a seeded capsule has matched the full bundle hash.
`gpu_replay --bundle-compact PATH` removes dictionary resources/chunks unreachable from retained rolling
manifests and exits without Vulkan when no image output path is supplied.
Set `PROSPER_GPU_TIMELINE_EXIT_AFTER_CAPTURE=1` for long unattended runs; it exits only after the selected
capsule and requested bundle are installed, never on capture failure or budget exhaustion.
For timing-sensitive routes, `PROSPER_GPU_TIMELINE_CAPTURE_WHEN_TARGET_DIM=WxH` selects the first matching
submit at or after `CAPTURE_SUBMIT`; `PROSPER_GPU_TIMELINE_CAPTURE_MIN_DRAWS=N` and
`PROSPER_GPU_TIMELINE_CAPTURE_MAX_DRAWS=N` bound semantic complexity to reject loading or cinematic passes.
`PROSPER_GPU_TIMELINE_CAPTURE_TARGET_DRAW_INDEX=MIN:MAX` narrows where the selected target may occur in the
raw semantic draw sequence. `PROSPER_GPU_TIMELINE_CAPTURE_MIN_DISPATCHES=N` and
`PROSPER_GPU_TIMELINE_CAPTURE_MAX_DISPATCHES=N` add dispatch-count bounds. Derive the full conjunction from
repeated positives and nearby negative samples with the offline v6 selector.
When the endpoint moves, predecessor manifests roll forward so the final bundle retains the latest requested
depth; dictionary bytes observed by evicted manifests still count against the unique-byte budget.
Per-target RTT is the normal renderer path. `PROSPER_RTT_SINGLE_TARGET=1` restores the obsolete flattened
single-framebuffer compositor only for diagnostic comparison; it cannot represent real post chains or
offscreen target dimensions.

Use `gpu_replay --inspect-only` to print fixed-function state, native color-target dimensions, resource
hashes, explicit clear intent, guest depth/stencil surface identities, and raw stencil-op provenance.
Use `gpu_replay --graph <capture.prgcap>` to resolve captured resource reads to the latest overlapping
earlier draw/dispatch writer in mixed PM4 order. External versions are deduplicated by logical range and
list every consumer; `future-writer=N` identifies a temporal read-before-write whose prior-submit version
is required. `--graph-json <path> <capture.prgcap>` emits the same closure frontier as structured JSON.
The graph is selected-submit scope: external leaves still require cross-submit producer capture (#595).
Version-4+ capsules also print and restore temporal RTT seeds: exact host-rendered surfaces sampled by the
captured submit whose producer ran in an earlier submit. This keeps replay from silently substituting stale
guest-memory bytes for renderer-owned history; older capsules remain readable and report zero seeds.
The draw header also reports raster state as `raster=cull/front-face/polygon-mode`, using Vulkan enum
values. `PROSPER_NO_CULL=1` disables culling; `PROSPER_FLIP_FRONT_FACE=1` preserves the cull mode and
toggles only the resolved winding convention. The latter is useful for isolating `PA_SU_SC_MODE_CNTL`
translation without the overdraw introduced by disabling culling entirely.
`PROSPER_EXECLOG=1` includes command order, target extent, depth/stencil identity, compare/write state, and clear
intent on recompile failures. `PROSPER_DS_CLEARLOG=1` logs only nonzero fold-time `DB_RENDER_CONTROL` clear-enable
writes, which is suitable for detecting transient clear pulses without `PROSPER_GFXLOG` packet volume.
Use `gpu_replay --validate` to reflect every draw's statically used VS/PS descriptor interface and validate it
against the captured runtime resource tables without initializing Vulkan. It exits nonzero for malformed SPIR-V,
stage/set mismatches, missing or duplicate bindings, wrong descriptor classes, or statically provable undersized
buffers. The same gate is available live as `PROSPER_DESCRIPTOR_VALIDATE=warn|strict|poison|all`; strict rejects
invalid draws, poison substitutes conspicuous resources, and all also prints valid manifests.
`--draw N:M` replays an
inclusive contiguous draw range, which is useful for rendering one pass without its downstream
composite/scanout draws; a single `--draw N` remains supported.
`--dump-resource DRAW:vs|ps:BINDING PATH` writes one captured resource's exact backing bytes for
external numeric/image inspection without dereferencing the original guest address.
`--dump-shader DRAW:vs|fs PATH` writes the captured SPIR-V module for validation/disassembly.
Capture v19+ also supports `--dump-realized-shader DRAW:vs|fs PATH`, which writes the exact bounded raw
RDNA2 source for a successfully realized graphics stage; use that output with `shader_inspect`.
`--dump-compute N PATH` writes one realized compute SPIR-V module, and
`--dump-compute-resource N:BINDING PATH` writes its exact pre-dispatch storage-buffer bytes.
`--compute-only N` executes one realized dispatch in isolation; combine it with
`--override-compute-spv N PATH` to minimize or hardware-A/B a captured shader without changing its exact
resource descriptors. The override disables the pixel oracle. Validated tiled 1D/2D compute storage executes
by default; `PROSPER_DISABLE_COMPUTE_TILED_2D_STORAGE=1` restores the old skip for a diagnostic A/B.
`PROSPER_COMPUTELOG=1` separates pipeline creation, submission, dispatch-wait, import, and writeback failures.
Renderer-owned RTT imports include their hash and nonzero-byte count, and buffer writeback logs include the
first eight dwords so bounds/offset constant buffers can be identified without another code change.

For compute producer provenance, `PROSPER_COMPUTELOG=1` records each `DispatchDirect` packet's
threadgroup counts, compute-program address/hash, and AGC-resolved resources from the register state at
that exact packet. Add `PROSPER_COMPUTELOG_DIM=WxH` to emit only dispatches referencing an image with
those dimensions (for example `1024x32` for Messenger's grading LUT). `PROSPER_COMPUTELOG=all` also
prints a per-submit no-match line while a dimension filter is active. Supported compute executes through
Vulkan in retained PM4 order. `PROSPER_COMPUTELOG_CODE=0x...` and `PROSPER_COMPUTELOG_SIZE=N` restrict
writeback before/after hashes to a matching program and/or storage-buffer size during long live runs.

`PROSPER_PROVENANCE_DIM=WxH` retains every decoded `CB_COLOR0_BASE` write across submits, then reports
the last matching writer whenever a draw samples an image of that size. Descriptor resolution can be
limited to likely target submits with `PROSPER_PROVENANCE_MIN_DRAWS=N`; color-target history is still
recorded for smaller earlier submits. This distinguishes a sampled GPU image produced by a draw from
one populated by compute/copy/CPU work without requiring the live Vulkan renderer.

For a differential replay, `PROSPER_STENCIL_CLEAR=<0..255>` overrides the initial stencil attachment
value and `PROSPER_STENCIL_REPLACE=<0..255>` overrides the replacement reference of an
ALWAYS+REPLACE stencil-prime draw. These are diagnostic controls only; they do not change guest-state
extraction or the default render path.
