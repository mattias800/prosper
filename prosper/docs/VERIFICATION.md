# Verification strategy (agentic-first — no manual eyeballing)

Every milestone is gated by a self-checking test whose **exit code is the truth**. Nobody looks at a
window; nobody diffs an image by hand. This keeps a long project cheap to run — `ctest` is the only
verification step. The graphics path is verified in layers, cheapest first.

## 1. Structural assertions (no GPU) — the bulk of correctness

Each pipeline stage is a pure function tested with exact-value assertions against an authoritative
reference. This catches most bugs deterministically and instantly:

| Stage | Test | Reference |
|-------|------|-----------|
| PM4 decode | `test_pm4_decode` | packets built by the real AGC Dcb functions |
| CommandProcessor apply | `test_command_processor` | Kyty `cp_op_indirect_cx_regs` |
| RenderState extract | `test_render_state` | Kyty `hw_ctx_*` register decodes |
| RDNA2→Vulkan translate | `test_render_state` | Kyty `GraphicsRender.cpp` enum maps |
| RDNA2 decode + operands | `test_rdna2_decode` | **`llvm-mc -mcpu=gfx1030` encodings** |

Using `llvm-mc` to assemble authoritative RDNA2 encodings (and `glslangValidator` for GLSL→SPIR-V)
means the test inputs are ground truth, not our own assumptions.

## 2. Render → readback → pixel/region assertions

`test_vulkan_triangle` renders offscreen on **llvmpipe** (deterministic software rasterizer),
copies the framebuffer to host memory, and asserts specific pixels/regions
(`center == red triangle`, `corner == blue clear`). No window, no eyeballing.

## 3. Golden-hash regression gate

The same test hashes the whole framebuffer (CRC32) and compares to a stored golden value. Any change
that flips a single pixel fails the test. llvmpipe is deterministic, so the hash is stable in CI.
`PROSPER_DUMP_PPM=1` writes `triangle.ppm` for *optional* human/debug inspection — never required.
(On a hardware GPU the raster may differ; the golden gate assumes the CI software rasterizer, while
the region checks in layer 2 stay portable.)

## 4. Execution-differential for the shader recompiler (LIVE)

This is how a recompiled shader is verified **semantically** without a PS5, fully automated:

```
llvm-mc assembles a known RDNA2 snippet  ->  our recompiler emits SPIR-V  ->
run it on Vulkan with known register inputs  ->  read back the result  ->
assert it equals the instruction's math   (e.g. v_add_f32(2.0, 3.0) == 5.0)
```

It proves the recompiler is *correct*, not merely structurally plausible, and every new opcode adds
one more cheap numeric assertion. Implemented as `test_rdna2_to_spirv` — 19 kernels covering
float/int/scalar ALU, convert/compare/select, bitfield, `pkrtz` (bit-exact f16), and **divergent
control flow** (EXEC per-lane predication, `saveexec`/restore, `s_cbranch_execz` if-then). The whole
`GpuState → recompiled shaders → VkPipeline → frame` spine is likewise pixel-verified
(`test_gpustate_render`, `test_pipeline_render`: topology/blend/depth/write-mask each driven from real
registers and asserted by readback).

## 4b. Recompiler coverage metric (data-driven roadmap)

`recompile_coverage()` reports per-instruction recompiler support without needing a complete
vertex/fragment; `shader_histo` runs it over the game's 41 **real** embedded shaders and prints
coverage % + the top first-unsupported opcodes. This turns "what to build next" into data (currently
~67% of instructions; the dominant remaining blocker is `SMEM`), and is regression-tested pure
(`test_recompile_coverage`).

## 5. Frame-CRC golden for the real game (once it boots)

When the completion-event fix lets the residency pass run and real draws flow through the spine,
capture the per-frame command-stream / framebuffer CRC once, then assert it thereafter. The game's
early frames are deterministic, so this detects any rendering regression frame-by-frame without review.

## CI

A GitHub Actions workflow builds + runs `ctest` on **Linux and Windows/MinGW** for every push/PR
(Vulkan discovery disabled on the runners, so the GPU-independent layers 1/4b gate CI; the
Vulkan-execution layers 2–4 run locally under llvmpipe). Dump-backed tests auto-skip when the private
game dump is absent.

## Rule of thumb

Prefer the cheapest layer that can catch a given bug: pure structural asserts for translation logic,
pixel/region + golden-hash for the raster path, execution-differential for shader semantics. Only the
final integrated frame needs the GPU; everything upstream is verified on the CPU.
