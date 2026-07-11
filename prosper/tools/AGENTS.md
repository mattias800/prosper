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
Set `PROSPER_CAPTURE_REVISION` explicitly in WSL worktrees: WSL Git cannot resolve their Windows-path
gitdir links, so the build-time fallback revision is `unknown` there.

Use `gpu_replay --inspect-only` to print fixed-function state, resource hashes, explicit clear intent,
guest depth/stencil surface identities, and raw stencil-op provenance. `--draw N:M` replays an
inclusive contiguous draw range, which is useful for rendering one pass without its downstream
composite/scanout draws; a single `--draw N` remains supported.

For a differential replay, `PROSPER_STENCIL_CLEAR=<0..255>` overrides the initial stencil attachment
value and `PROSPER_STENCIL_REPLACE=<0..255>` overrides the replacement reference of an
ALWAYS+REPLACE stencil-prime draw. These are diagnostic controls only; they do not change guest-state
extraction or the default render path.
