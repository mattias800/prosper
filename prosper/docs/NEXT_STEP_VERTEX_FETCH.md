# NEXT STEP — bindless-dynamic vertex fetch → the first real game frame

> **SUPERSEDED (2026-07-11).** This document describes an old state where the Messenger vertex shader
> did not recompile. Dynamic vertex-fetch resolution, per-PC descriptor provenance, and the associated
> resource plumbing have since landed. Do not use the TL;DR, reproduction output, or expected result
> below as current status. Start from `MESSENGER_BLACK_RENDER.md` and GitHub issues #299, #300, #514,
> and #515. The remainder is retained as implementation history.
>
> **Project context (read `../../CLAUDE.md` first).** prosper is a **PS5→PC compatibility layer** —
> "Wine/Proton for PS5" — that runs a **legally-owned** game natively by reimplementing Sony's published
> library ABI and translating the console's GPU commands + RDNA2 shaders to Vulkan/SPIR-V. This is
> interoperability/preservation engineering in the same class as Wine, Proton, RPCS3, and Dolphin. The
> task below is **graphics programming**: make a vertex shader's bindless vertex fetch resolve so it
> recompiles to SPIR-V. "Reverse-engineer / descriptor / disassembly" here mean standard emulator work,
> in-process on the developer's own machine. See `CLAUDE.md` for full scope.

**Supersedes the earlier EUD write-up** (the EUD theory was disproven — `eud_size_dw=0`; the real blocker
was the `SET_SH_REG` range bug, now fixed, plus the dynamic vertex fetch described here).

---

## TL;DR

The game boots to its frame loop and submits real draws; a live Vulkan renderer is wired in. The
**pixel shader recompiles** to valid SPIR-V, sampling the real 1920×1080 render target. The **vertex
shader does not yet** because its vertex fetch is **bindless-dynamic**: it loads the vertex-buffer
descriptor (V#) from a table at a **runtime-computed** offset, and the recompiler only resolves
descriptors bound at a **static** offset. Implement dynamic-fetch resolution (constant-fold the setup, or
a higher-level vertex-input model) and both stages recompile → the first real frame renders to
`PROSPER_FRAME_DIR/frame_0000.bmp`.

---

## Current state (verified, all committed on `master`)

Boot + render chain (Linux 45/45, Windows 20/20):
- `fix(hle_file)` — PS5 fd semantics; deser crash gone; all shaders load.
- `fix(agc)` — ctx-init empty descriptor; **first draws submitted**.
- `feat(gpu)` — live Vulkan renderer wired into `boot_trace` (`PROSPER_RENDER=1`) + `v_rndne_f32`.
- `feat(gpu)` — resource-table bridge; **the pixel shader recompiles (737 dwords)**.
- `fix(gpu)` — **`SET_SH_REG` sets a register RANGE** (was writing only the first register of each
  packet). This populated the descriptor SGPRs: the VS's 4 constant buffers and the PS's **1920×1080**
  sampled texture now decode correctly (were `addr=0` / a degenerate 1×1).

### Reproduce
```bash
cd /mnt/c/Users/matti/repos/ps5ys/prosper/build-linux
cmake --build . --target boot_trace -j8
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
  PROSPER_GFXLOG=1 PROSPER_RENDER=1 PROSPER_FRAME_DIR=/tmp/frames \
  timeout 70 ./boot_trace 2>&1 | grep -E "\[restab\]|\[exec\]|\[render\] frame"
```
`PROSPER_RESDUMP=1` adds the raw user-data-struct + SGPR dump; `PROSPER_SHADER_DUMP=/tmp` writes the
bound shaders' bytecode to disk for `llvm-mc` disassembly.

### What that prints now
```
[restab] VS ... -> 4 constant buffers (real bases, stride=16)
[restab] PS ... -> constant buffer + Texture 1920x1080 (real render target)
[exec] skip: recompile failed (vs=0 fs=737 ...)      <- PS recompiles; VS does not
[exec]   vs coverage: total=157 alu=148 exp=6 tabledep=3 unsupported=0   <- 3 vertex-fetch ops unresolved
```

---

## Root cause (RE'd, precise)

The VS is an NGG vertex shader whose vertex fetch is **bindless with a computed index**. Disassembly
(`llvm-mc --arch=amdgcn --mcpu=gfx1030` on the dumped VS):
```
s_load_dwordx2 s[64:65], s[26:27]                 ; s64 = load from a table pointer (user data)
...
s_lshl_b32  vcc_hi, s64, 4
s_and_b32   vcc_hi, vcc_hi, 0x1f0                  ; vcc_hi = (s64<<4) & 0x1f0  — a COMPUTED byte offset
s_load_dwordx4 s[8:11], s[24:25], vcc_hi           ; load the vertex-buffer V# from table s[24:25] at vcc_hi
buffer_load_format_xyzw v[0:3], v0, s[8:11] idxen  ; fetch THROUGH the dynamically-loaded V#
```
The recompiler (`src/gpu/rdna2_to_spirv.cpp`) resolves a `buffer_load_format`'s descriptor by matching the
SRSRC SGPR's provenance: either a **static** s_load immediate offset (`rs.sreg_srt[dst] = in.literal`, then
`rt->by_srt_offset(literal)`) or a direct user-data SGPR index (`rt->by_sgpr_base(sgpr)`). Here the offset
is `vcc_hi` — a **runtime-computed** value, not an immediate — so neither matches, and the op stays
unresolved (`tabledep`), so `recompile_vertex` returns `{}`.

`direct_resource_offset[8]=16 / [10]=18` does **not** rescue it: the fetch's SRSRC is the dynamically
loaded `s[8:11]`, not `s[16:19]`; and s16/s18 don't hold clean V#s for this shader (plausibility-guarded
out in `build_shader_resources`).

---

## The task — resolve bindless-dynamic vertex fetch

Pick one approach (1 is more faithful; 2 may be simpler if the draw's vertex state is capturable):

### Approach 1 — constant-fold the descriptor setup (recommended)
The fetch offset is computed from **scalar constants** (`vcc_hi` ← `s64` ← a table load ← user-data
pointers, all uniform across the wave). Add a small **scalar interpreter** over the shader prologue that
tracks concrete SGPR values through: `s_load_dword*` (read from guest memory at `SBASE+offset`, 1:1
mapped), `s_mov_b32`, `s_lshl/lshr/and/or/add` on constants, `s_buffer_load*`. When it reaches the
`s_load_dwordx4 s[8:11], s[24:25], vcc_hi` that produces a vertex-fetch SRSRC, it now knows the concrete
address → read the 4-dword V# from guest memory → emit a `VertexBuffer` `ShaderResource` and tag
`s[8:11]` so `buffer_load_format` resolves. Bind its bytes at the vertex-buffer binding.
- Scope it to **uniform scalar** setup only (bail if a value depends on a VGPR / lane id). That's enough
  for descriptor-table indexing, which is wave-uniform.
- Files: `src/gpu/rdna2_to_spirv.cpp` (the SMEM/`sreg_srt` provenance + a new const-eval pass over the
  prologue), `src/gpu/agc_shader_layout.cpp` (`decode_buffer_descriptor` for the read-back V#),
  `src/gpu/gpu_executor.cpp` `build_stage_table` (supply the guest-memory reader + user-data pointers).
- Verify with a synthetic unit test: a hand-written prologue that computes an offset and s_loads a V#
  from an in-test table, asserting the recompiler resolves the fetch (SPIR-V non-empty, `spirv-val` clean).

### Approach 2 — higher-level vertex-input model
Recognize the NGG "fetch shader" pattern and bind the draw's vertex buffers from the **draw state**
(VGT/attribute registers + the vertex-buffer table the game set) rather than resolving per-lane V#s.
This mirrors how real drivers stage a fetch shader. Needs the draw's vertex-buffer table captured in
`GpuState` (it isn't yet — `command_processor.cpp` `GpuState::Draw` holds only `index_count`).

---

## Why this is the next step

- The **PS already recompiles** and samples the real texture. The VS's dynamic vertex fetch is the single
  thing left before `execute_gpustate` produces a frame (`render_triangle_rgba` renders both stages →
  `present_write_frame` → `frame_0000.bmp`).
- Everything upstream is now correct: full register state (post `SET_SH_REG` fix), real descriptors, real
  1920×1080 texture. No fabricated data.

## Expected result

- `[exec]` prints non-empty `vs=<N> fs=<M>`; no "recompile failed".
- `[render] frame 0 rendered (WxH) -> /tmp/frames/frame_0000.bmp` — **a screenshot exists.**
- The frame is the game's first composited draw. Caveats that are *separate* follow-ups: the sampled
  1920×1080 render target may be empty/zeroed if no prior pass has written it (we don't yet execute the
  full draw chain), and the texture may be **tiled** (`decode_image_descriptor` returns `tile_mode`;
  detile if non-zero). Getting the pipeline to render at all is the milestone here.

## How to verify (agentic-first)

1. **Unit:** the synthetic const-fold test above (deterministic; no boot needed).
2. **Recompile gate:** the boot run's `[exec] vs=…` line goes non-zero.
3. **SPIR-V validity:** both modules pass Vulkan submission (or run the VS through `spirv-val`).
4. **End-to-end:** `frame_0000.bmp` appears; inspect with `magick /tmp/frames/frame_0000.bmp out.png`.
5. **No regressions:** `ctest --test-dir build-linux` 45/45 and the Windows build 20/20 stay green.

## Known follow-ups (separate, do NOT scope-creep)
- **Texture detiling** (`tile_mode != 0`) — Kyty `Tile.cpp` reference.
- **Executing the full draw chain** so sampled render targets have real content.
- **Per-draw state** (`GpuState::Draw` currently only `index_count`).
- **Real swapchain/window present** (`videoout_present.cpp` scaffolding exists).

## Reference index
| Thing | Location |
|---|---|
| Executor entry (recompile+resolve+render) | `src/gpu/gpu_execute.hpp` `execute_gpustate` |
| Live-renderer registry + table builder | `src/gpu/gpu_executor.cpp` `build_stage_table` |
| Descriptor decode + table build | `src/gpu/agc_shader_layout.cpp` `build_shader_resources`, `decode_buffer_descriptor`, `decode_image_descriptor` |
| Recompiler SRT/SGPR provenance (extend here) | `src/gpu/rdna2_to_spirv.cpp` SMEM/`MUBUF` cases (`sreg_srt`, `by_srt_offset`, `by_sgpr_base`) |
| Register-state fold (SET_SH_REG range fix) | `src/gpu/pm4_decode.cpp` (`IT_SET_SH_REG`), `command_processor.cpp` (`SetShRegDirect`) |
| Shader-header lookup by code addr | `src/hle/hle_agc.cpp` `prosper_agc_shader_header_for_code` |
| Live renderer + BMP dump | `tools/boot_trace/boot_trace.cpp` (`PROSPER_RENDER`), `tests/render_runner.h` |
| Kyty reference | `../Kyty/source/emulator/src/Graphics/Shader.cpp` (`ShaderParseUsage`, fetch handling) |
| Repro switches | `PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_RESDUMP=1 PROSPER_SHADER_DUMP=/tmp PROSPER_FRAME_DIR=/tmp/frames` |

**Environment:** build/run in WSL Ubuntu-24.04 as root; game dump at
`/mnt/c/Users/matti/repos/ps5ys/PPSA24651-app0` (gitignored, never commit). `llvm-mc`, `spirv-val`,
`glslangValidator`, ImageMagick, Vulkan/llvmpipe are installed.
