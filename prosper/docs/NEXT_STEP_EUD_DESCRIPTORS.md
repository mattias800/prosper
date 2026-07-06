# NEXT STEP — Extended User Data (EUD) descriptor resolution → the first real game frame

**Audience:** the agent picking up the render frontier. This is a self-contained work order: current
state, the exact task, why, how, expected result, how to verify, and every file/offset/command you need.
Read `docs/RENDER_LOOP.md` (the running log) and `docs/GPU_EXECUTOR_DESIGN.md` (the executor design) for
background, but everything load-bearing is repeated here.

---

## TL;DR

The game (The Messenger, `PPSA24651`) now boots to its frame loop and **submits real draws**. The
executor recompiles and renders on each submit through a live Vulkan device. The **pixel shader already
recompiles** to valid SPIR-V. The **vertex shader does not yet**, because its vertex-buffer descriptors
(V#) live in an **Extended User Data (EUD)** buffer in guest memory, not inline in the shader's user
registers — and our resource-table builder only reads inline descriptors. **Implement EUD indirection in
`build_shader_resources` and both stages recompile → the live renderer produces the first real game
frame.** This is a bounded, well-characterized change with a clear verification path.

---

## Current state (verified, all committed on `master`)

Boot chain that reaches here (all green, Linux 45/45 + Windows 20/20):
1. `fix(hle_file)` (e5fd518) — PS5 fd semantics; the asset-deserialization crash is gone; all shaders load.
2. `fix(agc)` (3d6e015) — ctx-init empty descriptor; **first draws submitted** (`SubmitDcb #2`: 2 draws).
3. `feat(gpu)` (3cb13a0) — live Vulkan renderer wired into `boot_trace` (`PROSPER_RENDER=1`) + `v_rndne_f32`.
4. `feat(gpu)` (13d70f7) — resource-table bridge; **the pixel shader recompiles to 737 dwords of SPIR-V**.

### How to reproduce the current state (WSL, from repo root)
```bash
cd /mnt/c/Users/matti/repos/ps5ys/prosper/build-linux
cmake --build . --target boot_trace -j8
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
  PROSPER_GFXLOG=1 PROSPER_RENDER=1 PROSPER_FRAME_DIR=/tmp/frames \
  timeout 70 ./boot_trace 2>&1 | grep -E "\[restab\]|\[exec\]|\[render\] frame"
```
The two gated switches (`PROSPER_GUEST_FS=1` + `PROSPER_GUEST_ARGS=-force-gfx-direct`) are REQUIRED to
reach the frame loop; they are off by default so `master`'s plain boot stays stable. `PROSPER_RENDER=1`
registers the live renderer; `PROSPER_GFXLOG=1` turns on the `[restab]`/`[exec]` diagnostics.

### What that run currently prints (the problem, in one screen)
```
[restab] VS code=0x…180100 base=0x8c -> 4 resources:
[restab]   cls=0 binding=2 addr=0x4cf26ba0 size=0 0x0 fmt=0 stride=0    <- ConstantBuffer, but addr looks wrong
[restab]   cls=0 binding=2 addr=0x0 …                                    <- addr=0: descriptor NOT inline here
[restab]   cls=0 binding=2 addr=0x0 …
[restab]   cls=0 binding=2 addr=0x0 …
[restab] PS code=0x…180900 base=0xc -> 2 resources:
[restab]   cls=0 binding=2 addr=0x0 …                                    <- cbuf addr=0 (should be a real cbuf)
[restab]   cls=2 binding=4 addr=0x60… size=4 1x1 fmt=9 …                 <- T# decodes to a degenerate 1x1
[exec] skip: recompile failed (vs=0 fs=737 …)                            <- PS recompiles (737 dwords); VS fails
[exec]   vs coverage: total=157 alu=148 exp=6 tabledep=3 unsupported=0   <- 3 vertex-fetch ops unresolved
[exec]   ps coverage: total=37  alu=23  exp=1 tabledep=13 unsupported=0
```
`vs=0` (empty) means `recompile_vertex` bailed: the executor renders only if **both** stages produce
SPIR-V (`execute_gpustate` in `src/gpu/gpu_execute.hpp` returns `{}` if `vs.empty() || fs.empty()`).
The degenerate descriptors (addr=0, 1×1 texture) are the tell: the real descriptors are **not inline**.

---

## Root cause (RE'd, precise)

The vertex shader loads its vertex-buffer descriptors **indirectly**. Disassembly of the real VS
(`llvm-mc --arch=amdgcn --mcpu=gfx1030`, blob at `/tmp/exec_vs_*.bin.trim` if still present, else re-dump
with `PROSPER_SHADER_DUMP=/tmp` on the run above):
```
s_load_dwordx4 s[8:11], s[24:25], <soffset>        ; load a V# FROM a table pointer held in s[24:25]
buffer_load_format_xyzw v[0:3], v0, s[8:11] idxen  ; vertex fetch THROUGH the loaded V#
```
So `s[24:25]` is a pointer into guest memory (the EUD / descriptor table); the actual V# is read from it,
then used to fetch. Our current `build_shader_resources` (`src/gpu/agc_shader_layout.cpp`) assumes every
descriptor is **inline** in the 16-dword user-SGPR block (`user_sgprs[offset_dw]`). When the descriptor's
`offset_dw >= 16`, that read is out of range / garbage → the degenerate descriptors above.

### The EUD mechanism (from Kyty `source/emulator/src/Graphics/Shader.cpp`, `ShaderParseUsage`)
- Each descriptor sharp has an `offset_dw` (Kyty calls it `start_register`).
  - `offset_dw < 16`  → **inline**: descriptor dwords are `user_sgpr.value[offset_dw + j]` (what we do).
  - `offset_dw >= 16` → **extended**: descriptor dwords are `extended_buffer[offset_dw - 16 + j]`, where
    `extended_buffer` is a `uint32_t*` into **guest memory** (1:1-mapped, so a raw host pointer works).
- The **EUD base pointer** is itself a 2-dword value sitting in the user SGPRs, tagged by Kyty usage
  **type `0x1b`**: `extended.data.fields[0..1] = user_sgpr.value[start_register .. +1]`, and its
  `.Base()` (a V#-style 48-bit base extract) is the EUD address:
  `base = ((fields[0] | (uint64_t)fields[1] << 32)) & 0xFFFFFFFFFFFF`  (see `decode_buffer_descriptor`
  in `agc_shader_layout.cpp` for the identical Base48 extract we already use).

Kyty references: `ShaderGetStorageBuffer` / `ShaderGetTextureBuffer` (the `extended ? extended_buffer[...]
: user_sgpr.value[...]` split, ~line 1141–1228) and the type-`0x1b` case that sets `extended_buffer`
(~line 1474–1488).

---

## The task

Make `build_shader_resources` (and its caller `build_stage_table`) read descriptors from the EUD when
`offset_dw >= 16`, instead of only from the inline user-SGPR block.

### Files
- `src/gpu/agc_shader_layout.hpp` / `.cpp` — `build_shader_resources` (the descriptor decode). **Primary.**
- `src/gpu/gpu_executor.cpp` — `build_stage_table` (finds the header + reads user SGPRs + calls the above).
- `src/gpu/shader_resources.hpp` — `ShaderResource` / `ShaderResourceTable` (no change expected).
- `tests/test_build_shader_resources.cpp` — extend with an EUD case (see "Verify").

### Steps
1. **Find the EUD base.** Two options, in order of preference:
   - (a) **Preferred, faithful:** parse the shader's usage table to find the type-`0x1b` slot, exactly as
     Kyty does, and read the 2-dword pointer from `user_sgprs[start_register .. +1]`. This also gives you
     the authoritative inline-vs-extended split per resource. If you go this route, port Kyty's
     `ShaderParseUsage` usage-slot walk (it reads a small table embedded near the shader header).
   - (b) **Pragmatic bootstrap:** the EUD pointer is a 2-dword V#-base pair in the user SGPRs. In the
     observed VS it is `s[24:25]`. You can locate it heuristically (scan the user-SGPR block for a
     2-dword pair whose Base48 points into a mapped, readable guest region of plausible size) and validate
     by decoding the descriptors it yields (a real V# has a sane stride/num_records; a real T# has
     non-degenerate width/height). Log both and prefer (a) once it works.
2. **Read extended descriptors.** In `build_shader_resources`, for each sharp with `offset_dw >= 16`,
   read the descriptor dwords from `((const uint32_t*)eud_base)[offset_dw - 16 + j]` (guest memory is
   1:1-mapped; guard the read with a readability check — reuse the `/dev/null`-write probe pattern from
   `exec_image_linux.cpp::probe_readable`, or bounds-check against the mapped range). For `offset_dw < 16`
   keep the existing inline read. Do this for all three sharp kinds already handled: constant buffers
   (`sharp[3]`), textures (`sharp[0]`), and the direct vertex buffers (`direct_resource_offset` types
   8/10) — the vertex-buffer V#s are the ones the VS needs.
3. **Provenance keys stay as-is.** The recompiler already resolves indirect descriptors correctly:
   `src/gpu/rdna2_to_spirv.cpp` tags an `s_load_dwordx4/x8`'s destination SGPRs with the load's immediate
   byte offset (`rs.sreg_srt[dst+k] = in.literal`, ~line 1537), and `buffer_load_format` / `image_sample`
   / `s_buffer_load` resolve via `rt->by_srt_offset(that_offset)` (~lines 1575, 1671, 1528). So a resource
   whose `srt_offset` equals the shader's s_load immediate offset will match. **Set each extended
   resource's `srt_offset = offset_dw * 4`** (byte offset — the builder already does this for constant
   buffers). Confirm the shader's s_load immediate offsets line up with `offset_dw*4`; if the game uses a
   different base convention, adjust the key so they match (this is the one thing to verify empirically
   against the `[restab]` + disassembly).
4. **Keep the binding convention.** `assign_convention_bindings` in `gpu_executor.cpp` already maps
   ConstantBuffer→2, VertexBuffer→3, Texture→4+. Leave it; the recompiler routes cbuf slot by `binding>=3`
   and uses `binding` directly for image samplers.

---

## Why this is the right next step

- It is the **single** thing standing between "the pixel shader recompiles" and "both shaders recompile."
  The moment `recompile_vertex` returns non-empty, `execute_gpustate` calls the live renderer and
  `present_write_frame` — the first real frame is produced and dumped to `PROSPER_FRAME_DIR/frame_*.bmp`.
- It also **fixes the pixel shader's descriptor DATA** (its cbuf and T# are currently the degenerate inline
  reads; the real ones are in the EUD too), so the PS won't just recompile — it will sample the right
  texture and read the right constants.
- It is **correctness-first**, not a shim: EUD indirection is exactly how the real hardware/driver resolves
  these descriptors (Kyty-cross-checked). No fabricated data.

---

## Expected result

- `[restab]` shows real descriptors: constant buffers with non-zero `addr` and plausible `size`; a texture
  with real `WxH` (not 1×1) and a `gpu_addr` in a mapped guest region.
- `[exec]` no longer prints "recompile failed" for the VS: `vs=<N> fs=<M>` both non-zero.
- `[render] frame 0 rendered (WxH) -> /tmp/frames/frame_0000.bmp` appears. **A BMP screenshot exists.**
- The frame content: the game's first UI/splash draw. It may look wrong if the sampled texture is **tiled**
  (see "Known follow-ups") — that's the *next* step, not this one. Getting a rendered frame at all
  (correct geometry from real vertex data, real shader math) is the milestone.

---

## How to verify (agentic-first — no eyeballing required to know it works)

1. **Unit:** extend `tests/test_build_shader_resources.cpp` with a synthetic EUD case — build a fake user-
   SGPR block whose type-0x1b pair points at a small in-test `uint32_t[]` EUD holding a known V# at
   `offset_dw=16`, call `build_shader_resources`, assert the resource's `gpu_addr`/`stride`/`format` match
   the EUD V# (not the inline zeros). This is the deterministic regression gate. Add to `CMakeLists.txt`
   like the existing `build_shader_resources` test.
2. **Recompile gate:** the `[exec] vs coverage` line already exists; after the fix, `recompile_vertex`
   returns non-empty. Add a targeted check if useful, but the boot run's `[exec]` line is the live signal.
3. **SPIR-V validity:** the render backend (`render_triangle_rgba`) submits both modules to Vulkan; an
   invalid module → `{}` → no frame. If you want an offline gate, run the recompiled VS through
   `spirv-val` (installed in WSL) as the other recompiler tests do.
4. **End-to-end:** the boot run produces `frame_0000.bmp`. Convert/inspect:
   `magick /tmp/frames/frame_0000.bmp /tmp/frame0.png` (ImageMagick is available), or read the BMP header
   to confirm non-empty. A CRC/pixel-region assert can be added once the expected content is known.
5. **No regressions:** `ctest --test-dir build-linux` (45/45) and the Windows build (`build-win`, 20/20)
   must stay green. The change is gated behind the same `PROSPER_RENDER`/guest-fs switches, so `master`'s
   default boot is unaffected.

---

## Known follow-ups (do NOT scope-creep into these; they are separate next steps)

1. **Texture detiling.** `decode_image_descriptor` returns `tile_mode`; UI textures may be GPU-tiled
   (`tile_mode != 0`). If the first frame's texture looks scrambled, implement detiling
   (linear ⇄ tiled address swizzle) as its own increment. Reference: Kyty `Tile.cpp`. A linear texture
   (`tile_mode == 0`) needs nothing.
2. **Per-draw state.** `extract_render_state` reads the *final* register state, so the executor renders all
   draws of a submit with the last-bound pipeline. If frame 2's two draws use different shaders/state, add
   per-draw state capture in `command_processor.cpp` (`GpuState::Draw` currently holds only `index_count`).
3. **Multiple bindless tables.** The VS s_loads from several pointers (`s[8:11]`, `s[12:15]`, `s[16:19]`,
   `s[20:23]`, `s[24:25]`, `s[26:27]`). If not all are the single EUD, generalize (b) above to resolve each.
4. **Present to the real display.** Frames currently go to `present_write_frame` + BMP. Wiring an actual
   swapchain/window (`videoout_present.cpp` scaffolding exists) is a later, orthogonal step.

---

## Reference index (where everything is)

| Thing | Location |
|---|---|
| Executor entry (recompile+resolve+render) | `src/gpu/gpu_execute.hpp` `execute_gpustate` |
| Live-renderer registry + table builder | `src/gpu/gpu_executor.cpp` `build_stage_table`, `read_user_sgprs`, `assign_convention_bindings` |
| Descriptor decode + table build (**edit here**) | `src/gpu/agc_shader_layout.cpp` `build_shader_resources`, `decode_buffer_descriptor`, `decode_image_descriptor` |
| Descriptor layout / sharp struct | `src/gpu/agc_shader_layout.hpp` (`AgcShaderUserData`, `AgcShaderSharp`, `DecodedBufferDescriptor`, `DecodedImageDescriptor`) |
| Resource contract | `src/gpu/shader_resources.hpp` (`ShaderResource`, `by_srt_offset`, `by_sgpr_base`) |
| Recompiler SRT provenance (already correct) | `src/gpu/rdna2_to_spirv.cpp` SMEM/`MUBUF`/`MIMG` cases (`sreg_srt`, `by_srt_offset`) |
| Shader-header lookup by code addr | `src/hle/hle_agc.cpp` `prosper_agc_shader_header_for_code` |
| Live renderer + BMP dump | `tools/boot_trace/boot_trace.cpp` (`PROSPER_RENDER` block) + `tests/render_runner.h` `render_triangle_rgba`, `dump_bmp` |
| Kyty EUD reference | `../Kyty/source/emulator/src/Graphics/Shader.cpp` `ShaderParseUsage`, `ShaderGetStorageBuffer`, type-0x1b case |
| Repro switches | `PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FRAME_DIR=/tmp/frames PROSPER_SHADER_DUMP=/tmp` |

**Environment:** build/run in WSL Ubuntu-24.04 as root; game dump at
`/mnt/c/Users/matti/repos/ps5ys/PPSA24651-app0` (gitignored, never commit). `llvm-mc`, `spirv-val`,
`glslangValidator`, ImageMagick (`magick`), and Vulkan/llvmpipe are all installed in that env.
