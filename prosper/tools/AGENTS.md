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
- **`spv_validate/`** — `spirv-val` wrapper for recompiled SPIR-V.
- **`niddiag/`, `fetch_niddb.sh`** — NID (Sony symbol hash) resolution helpers.

Verification here is agentic-first (see `docs/VERIFICATION.md`): prefer a
programmatic check (ctest exit code, `spirv-val`, a snapshot hash) over eyeballing.

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
Per-target RTT is the normal renderer path. `PROSPER_RTT_SINGLE_TARGET=1` restores the obsolete flattened
single-framebuffer compositor only for diagnostic comparison; it cannot represent real post chains or
offscreen target dimensions.

Use `gpu_replay --inspect-only` to print fixed-function state, native color-target dimensions, resource
hashes, explicit clear intent, guest depth/stencil surface identities, and raw stencil-op provenance.
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
