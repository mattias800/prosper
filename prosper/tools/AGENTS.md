# AGENTS.md — prosper/tools

Developer/agent tooling. These are debugging and verification aids, not part of
the shipped runtime. Build them from `build-linux/` like everything else.

- **`snapshot/`** — golden-image **rendering regression guard**. Run
  `python3 tools/snapshot/snapshot.py check` before/after any change that can
  affect rendered output (recompiler, AGC decode, render state, detile, present).
  See `snapshot/AGENTS.md`. **Run this after touching the render path.**
- **`boot_trace/`** — boots a SELF/ELF game image through the loader + HLE and
  runs it, with the fault handler, GPU executor, and (under `PROSPER_RENDER`) the
  live Vulkan renderer. The main harness for exercising a real title headlessly.
- **`self_dump/`** — parse a SELF/ELF and print its segment/program-header map
  (find file offsets for offline disassembly).
- **`shader_histo/`** — histogram RDNA2 opcodes across a title's shaders.
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
seconds/flip syntax as inline scripts, accept one entry per line, `#` comments,
and explicit ranges such as `f300-340:cross`. See `docs/INPUT_REPLAY.md`.
Set `PROSPER_PAD_RECORD=<path>` on any runner, or use `prosper-app --record <path>`, to capture the
final controller stream in that format. Completed button intervals are flushed immediately.

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
`PROSPER_GPU_CAPTURE_AFTER=N` ignores the first `N` renderer invocations before applying the draw-count
filters and `AT` counter. Pair it with `PROSPER_SUBMITLOG`/`PROSPER_RENDER_FIRST` when several early
scenes share the same draw count as a late target.
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
hash, RTT-hit state, last compute/DMA writer, draw index, and PM4 order. `PROSPER_TARGET_STEP_HASH_DIM`
rerenders matching target passes by prefix and logs per-draw hashes plus dark/white/mean metrics;
`PROSPER_TARGET_STEP_HASH_MIN_DRAWS=N` bounds that intentionally expensive bisect.
`PROSPER_RENDER_DELAY_MS=N` skips synchronous Vulkan work for N milliseconds from the first submit while
the guest and command decoder advance. The `screenshot` frontend exposes this as `--warmup-seconds`, plus
`--warmup-submits` for `PROSPER_RENDER_FIRST`; use wall-clock warmup for progression captures and the exact
submit gate for repeatable renderer investigations.
`PROSPER_GPU_TIMELINE=<path>.prgtl` records every folded submit before renderer sampling plus every
VideoOut flip. `gpu_timeline <path> [--records]` inspects the checksummed index offline. Recording is
independent of `PROSPER_RENDER_EVERY`. To materialize one exact indexed submit without rendering the
warmup, set `PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT=N` and
`PROSPER_GPU_TIMELINE_CAPTURE=<path>.prgcap` on a second run. Version-2 detail records link the capsule;
version-5 capsules deduplicate content-addressed shader/resource versions and retain mixed draw/dispatch
order plus explicit unrealized operations. Selection is intentionally bounded to the consumer and an optional
immediate predecessor; automatic recursive cross-submit producer closure remains #595.
Timeline version 3 retains lightweight graphics-target summaries for the previous 64 submits when a
detailed capture is requested. `PROSPER_GPU_TIMELINE_HISTORY=N` raises that bounded window to at most
4096. Producer records resolve temporal image leaves to the latest overlapping prior submit/draw/PM4
order, or state `unresolved`; they intentionally do not retain delayed pointers to mutable guest bytes.
Set `PROSPER_GPU_TIMELINE_CAPTURE_PREDECESSOR=<path>.prgcap` to snapshot exact submit `N-1` at producer
time alongside selected submit `N`. Replay the pair with `gpu_replay --prepend producer consumer output`.
This is a one-level probe; graph the producer and recurse when it also reads a temporal version.
For bounded recursion, add `PROSPER_GPU_TIMELINE_CAPTURE_BUNDLE=<path>.prgbundle`,
`PROSPER_GPU_TIMELINE_CAPTURE_DEPTH=2..16`, and optionally
`PROSPER_GPU_TIMELINE_CAPTURE_MAX_UNIQUE_MB=64..4096` (default 1024). `gpu_replay --bundle bundle output`
executes the ordered window and classifies each temporal frontier as included, seeded, or depth-bounded.
Bundles use content-defined chunks so shifted capture metadata does not defeat cross-submit deduplication.
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

For skipped-compute producer provenance, `PROSPER_COMPUTELOG=1` records each `DispatchDirect` packet's
threadgroup counts, compute-program address/hash, and AGC-resolved resources from the register state at
that exact packet. Add `PROSPER_COMPUTELOG_DIM=WxH` to emit only dispatches referencing an image with
those dimensions (for example `1024x32` for Messenger's grading LUT). `PROSPER_COMPUTELOG=all` also
prints a per-submit no-match line while a dimension filter is active. Compute remains unexecuted; this
trace identifies work and resource contracts that the HLE currently skips.

`PROSPER_PROVENANCE_DIM=WxH` retains every decoded `CB_COLOR0_BASE` write across submits, then reports
the last matching writer whenever a draw samples an image of that size. Descriptor resolution can be
limited to likely target submits with `PROSPER_PROVENANCE_MIN_DRAWS=N`; color-target history is still
recorded for smaller earlier submits. This distinguishes a sampled GPU image produced by a draw from
one populated by compute/copy/CPU work without requiring the live Vulkan renderer.

For a differential replay, `PROSPER_STENCIL_CLEAR=<0..255>` overrides the initial stencil attachment
value and `PROSPER_STENCIL_REPLACE=<0..255>` overrides the replacement reference of an
ALWAYS+REPLACE stencil-prime draw. These are diagnostic controls only; they do not change guest-state
extraction or the default render path.
