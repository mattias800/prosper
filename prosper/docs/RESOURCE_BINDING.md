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
   **DONE** (kernel 22 — provenance routes two `s_buffer_load`s to bindings 2 & 3).
2. **`buffer_load_format_*` — float32 fast path.** For `DataFormat::Float32` (positions, most
   attrs) a format load is a raw dword load — reuse the MUBUF path, keyed to a `VertexBuffer`
   resource. This is what unblocks the game's real vertex shaders (op 0x0/0x3 dominate). Test: a
   float32 vertex fetch through the table. **DONE** (kernel 23 exec-diff; `vertex_fetch_render`
   renders real buffer-sourced geometry through a bound VkPipeline).
3. **`buffer_load_format_*` — packed conversions.** `Unorm8`/`Snorm16`/… : load dword → unpack →
   normalize (bitfield-extract → convert → /255|/32767, SNORM clamped to ≥ −1.0; Float16 via
   `UnpackHalf2x16`). Test: unorm8×4 → 4 floats in [0,1]. **DONE** — `unpack_norm`/`unpack_half`
   cover Unorm8/Snorm8/Unorm16/Snorm16/Float16; kernels 24 (unorm8×4) & 25 (snorm16×2 + clamp)
   verify the exact numbers. Integer sub-dword formats (Uint8/Sint8/…) are still rejected (no
   integer-attribute path yet) rather than mis-normalized.
4. **MIMG `image_sample`/`image_load`.** Bind a Vulkan sampled image + sampler from the T#/S#;
   emit `OpImageSampleImplicitLod`. Test: sample a known 2×2 texture. **DONE** — combined image+sampler
   (`OpTypeImage`/`OpTypeSampledImage`); `image_sample` (0x20, implicit LOD), `image_sample_lz` (0x27,
   LOD 0) / `image_sample_l` (0x24, explicit LOD) via `OpImageSampleExplicitLod`, `image_load` (0x00)
   via `OpImageFetch`; T# resolved through the same SRSRC provenance as vertex buffers (SMEM x8 tags it).
   Tests: `texture_sample_render` (2×2 texel, u/v routing + LOD variant + fetch) and `textured_interp_
   render` (VS-interpolated UVs → sample, the full textured-draw path). 2D float non-NSA; other dims /
   NSA / gradient / compare-shadow variants deferred (rejected, not faked).

## Beyond the staged plan (also DONE this line of work)

- **MUBUF stores** — `buffer_store_dword/x2/x4` + `buffer_store_format_*` (raw/Float32), with a real
  **EXEC-predicated conditional store** (selection-merge on the per-lane EXEC bool) + `robustBufferAccess`.
- **VINTRP** — pixel-shader attribute interpolation: VS `EXP PARAM_n` → Output varying, PS
  `v_interp_p1/p2/mov` → interpolated Input varying (deferred EntryPoint so varyings join the interface).
- **`s_cbranch_execz` guard-to-`s_endpgm`** linearization; **SCC** (`s_cmp`/`s_cselect`).

**Status:** the recompiler covers the full shape of a real textured, interpolated, buffer-reading/
writing draw. The remaining gap to recompiling the game's *own* format-dependent shaders is not the
recompiler — it is the **resource table**, which `build_shader_resources` (agc_shader_layout) can only
fill from the shader's *bound user-data SGPRs* (V#/T#/S# descriptors), which the game sets at DRAW time.
So it is gated on the game reaching draw submission = the parked GfxDevice boot wall (front-half). The
table-less coverage metric (~74%) therefore *understates* real render-path coverage: every MIMG /
`buffer_load_format` "blocker" it reports is a shape the recompiler already handles given a table.

Stage 2 was the high-value unlock (real VS recompile → real VS+PS frames from the game). Stages 1–3
depend only on this contract + the front-half filling the table for constant/vertex buffers; stage 4
adds textures.

## Front-half deliverable (agent 2's next piece)

`ShaderResourceTable build_shader_resources(const Shader& shdr)` (or equivalent): walk the shader's
`user_data`/SRT, read each V#/T#/S# descriptor, decode `DataFormat`/dims/base/stride, assign bindings,
and fill `srt_offset` with each descriptor's user_data byte offset. Provide the referenced guest
bytes to the pipeline via `gpu_addr` (1:1-mapped, so it's a host pointer). No recompiler knowledge
needed — just the descriptors → the table.

## Ruled out

Cross-title falsifications about **how the user-data / SH register block reaches a stage**. One line
per dead hypothesis, the evidence that killed it, and where that evidence lives. Do not re-derive
these without contradictory new evidence.

The umbrella is **#305** — a graphics stage resolves garbage descriptors exactly when the user-data
block the guest most recently programmed is **larger** than the bound pipeline's user-SGPR window
(`SPI_SHADER_PGM_RSRC2_{GS,PS}.USER_SGPR`, which equals the shader's own `user_data_range_end`). The
shader then dereferences a V# `num_records`/`dword3` tail as a pointer — the `0x0004dfac…` /
`0x0001d22c…` constant family — the const-fold correctly refuses to invent a descriptor, and the draw
is skipped fail-visibly. Confirmed on Nikoderiko (`PPSA23760`, #1607) and DOLL / Dragon Quest VII
(`PPSA17942`), both UE4; **not** reproduced on The Pathless or The Plucky Squire.

| Hypothesis | Verdict and evidence | Source |
|---|---|---|
| The **user-data block** is a previous pipeline's leftover (from the founding premise, "first draws run with the previous pipeline's PGM + user data" — the *user-data* half; the PGM half still holds, see **Still open** below) | **Falsified.** This was #305's founding premise and its original title; the issue was **retitled on 2026-08-01** so nobody starts from it. `PROSPER_UDPROV=1` records each SH register's last-write `command_order` and path and carries it into the per-draw snapshot: at **every** failing draw the dwords the shader dereferences were written by the **immediately preceding bind**, a handful of packets before the draw. Identical across 21 measured stages. | #305 |
| The **shader-header registry lookup is stale** (a recycled code allocation resolves an old layout) | **Falsified.** Real hazard in shape — `prosper_agc_shader_header_for_code` returns the *first* match in an append-only registry — but measured `registrations=1` for every failing address across a 2,725-entry registry. | #305 |
| A **bind packet is missing, dropped or mis-ordered** in the decoded stream | **Falsified.** `PROSPER_BINDTRACE=1` logs every Sh `Set*RegsIndirect` packet carrying a program register, in stream order, interleaved with draws: **0 of 141** register arrays apply more than one distinct `(es_lo, rsrc2)` over 193,397 packets, and **300,404 of 300,404** draws fold with the immediately preceding bind. Zero `SetRegsIndirect array unmapped`, zero out-of-range Sh writes. *(Use these corrected figures: the numbers first posted — 434,239 packets, 871,648 of 876,217 — were contaminated by compute dispatches mislabelled `DRAW` by the instrument itself, and the issue body still quotes them. The correction tightens the conclusion.)* | #305 correction comment |
| The stage's user data is the **tail** of the programmed block (a constant seed shift) | **Falsified despite a 9-of-9 numeric fit.** Declared descriptors do land on clean guest pointers exactly `programmed − user_data_range_end` dwords above `USER_DATA_GS_0` — but `USER_SGPR == user_data_range_end` for every stage measured (12/12, 8/8, 24/24, 20/20, 30/30, 32/32), so the hardware loads only that many registers and the stage physically **cannot see** anything above them. A live A/B with the shifted seed raised `exec-recompile-reject` from a 118–141 baseline to **521**. Retained, **off**, as `PROSPER_UD_TAIL_ALIGN`; it must stay off. | #305 |
| **#140 — TYPE-0 AGC data packets** carry the missing bind | **Falsified.** One of #305's two original candidates. The register writes that matter all arrive as ordinary decoded `SET_SH_REG` / `Set*RegsIndirect` packets whose provenance is now directly observable, and none is missing. #140 is closed and unrelated to this path. | #305 |
| **Cross-submit register inheritance** is itself the bug | **Falsified.** Opening a fold with a draw and inheriting the previous submit's bind is the *normal* pattern in this title — 10,806 `q1` folds and 653 `q3` folds do it, against 6,342 and 179 that open with a bind — and is correct on a shared ring. The defect is that the *inherited state* is wrong, not that inheritance happens. | #305 |
| **#1226's Acb register-file split** is dropping graphics user-data writes | **Falsified.** Measured 1,747 Acb folds with SH offsets confined to `[0x207,0x249]` and **zero** graphics user-data writes, so the split discards nothing hardware would have applied. Candidate closed. | #305 |
| A **wrong seeding origin** or **header decode** explains the unmapped pointers | **Falsified**, and the same measurement *confirms* two assumptions previously taken on faith: the seeding base `USER_DATA_<stage>_0 + user_data_range_start` (`range_start` is 0 for every shader, `range_end` = last declared direct offset + 2), and the merged-stage convention that user data begins at shader SGPR `s8`. The shader's own `SBASE` equals the header's declared offset in all ten stages disassembled. | #1607, #305 |

**Still open**, and the acceptance test for any proposed mechanism: **it must explain the
GS-versus-PS asymmetry.** Every traced pixel stage in these titles has its declared direct pointer
readable; only the vertex/GS block loses. The remaining candidate is cross-queue / multi-buffer
submit ordering — for every misfitting vertex stage the pipeline was bound in a `q1` (Dcb) stream at
fold *N* while the user-data block was written by a `q3` (`w1KFAHVqpaU` variant submit) stream at
fold *N+1*, and every sampled failing draw is the first bind-or-draw event of its own `q3` fold.

**Instruments** (`PROSPER_UDPROV`, `PROSPER_BINDTRACE`, `[udcand]`, `PROSPER_SHADER_HEADER_NEWEST`,
`PROSPER_UD_TAIL_ALIGN`) are on PR #1639 — reuse them rather than rebuilding this measurement.

`PROSPER_SHARPLOG=1` is the fourth one and answers a different question from the rest. They all
describe what the front half *produced*; this one prints what the shader **declared** — the stage's
raw sharp table (per-category counts, each slot's `bits`/`offset_dw`/`size`/EUD residency, and the
direct slots) — and then names the exact reason a declared texture slot was dropped. Between the two
sits a chain of about ten `continue`s, and without the input side "the front half never saw this
resource" and "it saw it and rejected it" produce identical evidence while pointing at different
files. Reach for it before concluding anything about descriptor *recovery*: on Earthion (#1590) it
overturned exactly that conclusion in one run.
