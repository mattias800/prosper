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
- **`spv_validate/`** — `spirv-val` wrapper for recompiled SPIR-V.
- **`niddiag/`, `fetch_niddb.sh`** — NID (Sony symbol hash) resolution helpers.

Verification here is agentic-first (see `docs/VERIFICATION.md`): prefer a
programmatic check (ctest exit code, `spirv-val`, a snapshot hash) over eyeballing.
