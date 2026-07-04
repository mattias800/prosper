# Resource binding — the front-half ↔ recompiler contract

**Purpose.** Unblock the format/descriptor-dependent shader memory instructions — `s_buffer_load_*`
(multi-buffer uniforms), `buffer_load_format_*` (vertex fetch), and `image_sample`/`image_load`
(textures) — *correctly*, without guessing formats. The seam is `src/gpu/shader_resources.hpp`.

## Why a contract (the core problem)

To translate `buffer_load_format_xyzw v[..], vaddr, s[8:11], …` the recompiler must know the vertex
attribute's **data format** — float32? unorm8? snorm16? — to emit the right conversion. A float32
attribute is a raw dword load (no conversion in our raw-32-bit-VGPR model); a unorm8×4 attribute is a
`load dword → unpack 4 bytes → /255.0`. **That format lives in the V# descriptor the game builds**,
which the recompiler never sees directly (it's in memory the game loads via `s_load`). So the
recompiler cannot, on its own, translate a format load correctly. It must be *told* the formats.

**Division of labor:**
- **Front-half (agent 2 / the AGC HLE):** knows the game's real GPU resources. From the shader's
  `user_data` / SRT (already traced — `ShaderUserData`: `direct_resource_offset`,
  `sharp_resource_offset[4]`, counts) and the game's bound resources, it reads the **V#/T#/S#
  descriptors** (base address, stride, `DFMT`/`NFMT`, dims, …) and produces a `ShaderResourceTable`.
- **Back-half (recompiler, me):** parameterized by that table. While translating a memory op it
  resolves *which* resource the op targets (see "provenance" below), emits the correct binding +
  format conversion, and records the binding layout.
- **Pipeline (back-half):** binds `size` bytes at each resource's `gpu_addr` (unified guest memory)
  to descriptor-set 0, `binding` — real game data flowing to the shader.

The result: real shaders get **real inputs** (turning the current near-black demo output real), and
vertex/texture shaders recompile correctly.

## Descriptor provenance — how a memory op maps to a resource

A shader uses a resource in two steps:
```
s_load_dwordx4  s[8:11], s[srt_ptr], 0x20      ; load a V# descriptor from user_data offset 0x20
buffer_load_format_xyzw v[0:3], v1, s[8:11], 0  ; use it to fetch a vertex attribute
```
So the recompiler tracks, per SGPR, **which SRT/user_data byte offset a descriptor came from**:
- On an `s_load*`/`s_buffer_load*` whose SBASE is the shader's user_data pointer, tag the destination
  SGPRs with `srt_offset = <the load's offset>`.
- On a `buffer_load_format_*` / `image_*` / `s_buffer_load_*`, read the tag off its SRSRC/SBASE SGPRs
  and look up `table.by_srt_offset(tag)` → the `ShaderResource` (format, binding, …).

This is a small SGPR-provenance side-table in the recompiler (analogous to the existing VGPR/SGPR SSA
maps). It keys the abstract "resource N" to the concrete descriptor the front-half described. The
front-half fills `ShaderResource::srt_offset` with the same offset so the two sides rendezvous.

### Two provenance modes (INDIRECT vs DIRECT)

Not every descriptor is loaded in-shader. Sony resources come in two flavours, and a `ShaderResource`
sets whichever key matches (leaving the other `0xFFFFFFFF`):
- **INDIRECT (`srt_offset`)** — the shader `s_load_dwordx4`s the V# from its user_data/SRT (above).
  Constant buffers (`s_buffer_load`) are typically this. Recompiler tags the load, resolves by offset.
- **DIRECT (`sgpr_base`)** — the driver places the V# *straight into the user-data SGPRs* at launch
  (Sony "direct" resources). **Vertex-buffer descriptors are this** — there is no in-shader load to
  tag. The recompiler resolves a memory op by matching its SRSRC/SBASE SGPR index to `sgpr_base`
  (`by_sgpr_base`). The recompiler must know the shader's user-data→SGPR layout at entry; the
  front-half provides `sgpr_base` = the SGPR the V# occupies.

So the recompiler resolves a memory op's descriptor by: (1) an `s_load` provenance tag → `by_srt_offset`,
else (2) the SRSRC/SBASE SGPR index → `by_sgpr_base`.

`DataFormat` is decoded from the descriptor's `DFMT`/`NFMT` (e.g. `DFMT=32_32_32_32,NFMT=FLOAT` →
`Float32`, `num_components=4`; `DFMT=8_8_8_8,NFMT=UNORM` → `Unorm8`, `num_components=4`). That decode
lives front-half (it owns the descriptor bit layout); the recompiler only consumes `DataFormat`.

## Binding scheme

- Descriptor set 0. Bindings assigned by the front-half when it builds the table (stable per shader).
  Convention: constant buffers first, then vertex buffers, then textures, then samplers — but the
  recompiler only relies on the `binding` value in each `ShaderResource`, not the order.
- The recompiler's existing single constant buffer (binding 2 in the compute shell) generalizes to
  "the `ShaderResource` with that binding."

## Staged implementation (each stage its own execution-verified commit)

1. **Multi-buffer `s_buffer_load`** (generalize today's single-cbuf model). Provenance-track the V#
   SGPRs → `by_srt_offset` → per-resource binding. Test: two constant buffers, distinct contents.
2. **`buffer_load_format_*` — float32 fast path.** For `DataFormat::Float32` (positions, most
   attrs) a format load is a raw dword load — reuse the MUBUF path, keyed to a `VertexBuffer`
   resource. This is what unblocks the game's real vertex shaders (op 0x0/0x3 dominate). Test: a
   float32 vertex fetch through the table.
3. **`buffer_load_format_*` — packed conversions.** `Unorm8`/`Snorm16`/… : load dword → unpack →
   normalize (SPIR-V `UnpackUnorm4x8` etc.). Test: unorm8×4 → 4 floats in [0,1].
4. **MIMG `image_sample`/`image_load`.** Bind a Vulkan sampled image + sampler from the T#/S#;
   emit `OpImageSampleImplicitLod`. Test: sample a known 2×2 texture.

Stage 2 is the high-value unlock (real VS recompile → real VS+PS frames from the game). Stages 1/2
depend only on this contract + the front-half filling the table for constant/vertex buffers.

## Front-half deliverable (agent 2's next piece)

`ShaderResourceTable build_shader_resources(const Shader& shdr)` (or equivalent): walk the shader's
`user_data`/SRT, read each V#/T#/S# descriptor, decode `DataFormat`/dims/base/stride, assign bindings,
and fill `srt_offset` with each descriptor's user_data byte offset. Provide the referenced guest
bytes to the pipeline via `gpu_addr` (1:1-mapped, so it's a host pointer). No recompiler knowledge
needed — just the descriptors → the table.
