# Shader inspect

`shader_inspect` decodes one raw RDNA2 shader. It is the fast offline path for mapping shader control
flow before changing the recompiler.

```bash
cmake --build build-linux -j8 --target shader_inspect
./build-linux/shader_inspect /tmp/shaders/exec_ps_7f123400.bin

# Run the complete stage translator too; PROSPER_DBG identifies a contextual rejection PC.
PROSPER_DBG=1 ./build-linux/shader_inspect /tmp/shaders/exec_vs_7f123400.bin --stage vertex
PROSPER_DBG=1 ./build-linux/shader_inspect /tmp/shaders/exec_cs_7f123400.bin --stage compute
```

`PROSPER_SHADER_DUMP=DIR` writes raw shaders that fail recompilation. To inspect a shader that succeeds,
run the title or replay with `PROSPER_SHADER_DUMP_SUCCESS=DIR`. Each unique successful graphics or compute shader
produces a pair such as:

```text
success_ps_35956264829da0c6_62ad8faab4566b7a.bin
success_ps_35956264829da0c6_62ad8faab4566b7a.spv
success_cs_6f6a5fdd6e9f8d73_2339aa565126d182.bin
success_cs_6f6a5fdd6e9f8d73_2339aa565126d182.spv
```

For a large live compute program that reaches the backend but fails during Vulkan pipeline creation,
filter the backend trace by guest code address and dump only that exact translated module:

```sh
PROSPER_COMPUTELOG_CODE=0x50057b800 \
PROSPER_COMPUTELOG_SPIRV=~/prosper-shaders/target.spv \
  ./build-linux/prosper-app --dump <DUMP_ROOT>/PPSA21564-app0
spirv-val --target-env vulkan1.2 ~/prosper-shaders/target.spv
```

The destination directory must already exist. As with raw shader dumps, the resulting SPIR-V is
title-derived diagnostic data: keep it local and do not commit it.

The first hash identifies the translated SPIR-V and the second identifies the exact raw RDNA2 stream.
Pass the `.bin` file to `shader_inspect`; use the adjacent `.spv` for SPIR-V disassembly or validation.
The dump directory is created automatically. `vs`, `ps`, and `cs` identify vertex, pixel, and compute stages.

Each instruction row includes its dword PC, encoded length, format/opcode, exact raw words, decoded
operands, and, for SOPP branches, the signed immediate and resolved target PC. The header also prints
the stage-agnostic `recompile_coverage` result. That coverage is intentionally compute-safe; a VCC
control-flow shape may remain listed there even when the vertex/fragment structurizer accepts it.
`--stage vertex|fragment|compute` additionally runs the complete stage translator, which exposes contextual
resource and structured-control-flow failures that per-instruction coverage cannot decide.

## `--stage` cannot prove a shader is unsupported (#1571)

**`shader_inspect` has no resource table, and a table-less stage rejection is NOT evidence of a shader
defect.** A raw dump carries instructions but no descriptors, so the tool can never build a
`ShaderResourceTable`. The recompiler then refuses — by design — to lower anything that resolves a
V#/T#/S# through that table: `MIMG`, `MUBUF` and `MTBUF` in every stage, and additionally `SMEM` in the
**vertex and fragment** stages, which gate scalar memory on `allow_smem = (rt != nullptr)`. Compute
passes `allow_smem = true`, so constant-buffer loads are fine there — but compute image and buffer ops
still need the table.

The gates live in `emit_alu` in `src/gpu/rdna2_to_spirv.cpp` (MIMG rejects on `!allow_smem || !rt`,
SMEM on `!allow_smem`); the per-stage `allow_smem` values are set by `recompile_fragment_impl` and
`recompile_vertex_impl` (`rt != nullptr`) versus `recompile_compute` (`true`). Those functions are named
rather than cited by line because line numbers drift — grep the names.

When such an instruction is present the tool now reports:

```text
stage-recompile stage=fragment status=undetermined-no-resource-table spirv_dwords=0 table_dependent=17
stage-recompile NOTE: TOOL LIMITATION, NOT A SHADER DEFECT. ...
```

Treat `undetermined-no-resource-table` as "no verdict", never as a missing opcode. This mattered: on a
corpus of 114 shaders that had provably recompiled and rendered live, the pre-fix tool called **109 of
them `rejected`** — 33/35 compute, 27/29 fragment, 49/50 vertex. Agents acted on those false leads.

**For a table-accurate verdict, use `gpu_replay`**, which has the real descriptors from a capture:

```bash
gpu_replay <capture>.prgcap --inspect-only                     # per-draw realize/fail status
gpu_replay <capture>.prgcap --dump-failed-shader FAILURE:STAGE out.bin   # a genuinely failed stage
```

Exit codes: `0` recompiled, `1` genuine defect (truncated stream, or a rejection attributable to the
shader), `2` usage/IO error, `3` undetermined because no resource table could be supplied. `3` is
deliberately non-zero — a table-less run is never reported as a pass.
When a shader contains the fully proven bounded scalar `s_setpc_b64` jump-table idiom, the header also
prints its constant-buffer selector, adjustment/clamp, complete target list, merge PC, and owning span.

The input is bounded to 16 MiB and decoding stops at the first `s_endpgm` or unknown instruction.
Raw and SPIR-V dumps contain title-derived code, are local-only, and must not be committed.
