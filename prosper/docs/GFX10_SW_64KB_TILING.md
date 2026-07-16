# GFX10 64KB tiling (SW_64KB_S / SW_64KB_R_X) — the addrlib swizzle-pattern equation

**Status: SW_64KB_S SOLVED (2026-07-10, issue #288), validated against a live DOLL (PPSA17942)
texture. SW_64KB_R_X implemented with the exact addrlib equation; its one free parameter (the pipe
count) could not be pinned from captured data yet — see below.**

This documents how prosper detiles the PS5's 64KB swizzle modes — `tile_mode 9 = SW_64KB_S`
(standard; DOLL's BC material textures) and `tile_mode 27 = SW_64KB_R_X` (render-target; DOLL's
RT/post composites) — and how the equation was obtained and validated. Unlike `SW_4KB_S`
(`GFX10_SW_4KB_S_TILING.md`), which was derived empirically, the 64KB modes were taken directly
from an authoritative open-source reference and then validated against live captures.

## Why a flat bit-interleave cannot work (the failed first attempt)

A prior derivation attempt brute-forced all 1716 order-preserving X/Y bit-interleavings for a
one-block 512×256 BC1 texture, scoring decoded-image coherence: **no winner** (best ~4% over
row-major, still noise). The reason is now obvious from the real equation: the 64KB pattern is not
a pure interleave — the micro-tile (low 8 bits) uses a bpe-dependent mixed order, and `_X` modes
XOR *multiple* coordinate bits into single offset bits. Only the real equation family contains the
answer; it was never in the brute-force search space (the S pattern IS an interleave, but the
search scored a truncated single-dump wrongly... the addrlib pattern decides it exactly).

## The equation (from AMD addrlib)

Reference: Mesa `src/amd/addrlib/src/gfx10/gfx10SwizzlePattern.h` (tables) +
`src/core/addrlib.cpp:ComputeOffsetFromSwizzlePattern` (evaluation) +
`gfx10addrlib.cpp:ComputeSurfaceAddrFromCoordMacroTiled` (block layout). Older GCN/PS4 tiling models were
deliberately not used because they target the wrong hardware generation.

For a 2D single-sample surface:

```
addr(x, y) = blockIndex * 65536 + patternOffset(x, y)        [ ^ pipeBankXor for _X modes; we use 0 ]
blockIndex = (y / blockH) * ceil(pitch / blockW) + (x / blockW)
```

- x, y are **element** coords (texels, or 4×4 BC blocks), pitch in elements.
- 64KB block dims per element size: 1 B → 256×256, 2 B → 256×128, 4 B → 128×128, 8 B → 128×64,
  16 B → 64×64 (the 256-byte micro-block `Block256_2d` scaled ×16 each axis).
- `patternOffset` produces the 16 offset bits within the block; **each bit i is the XOR of the
  element-coordinate bits selected by an (x-mask, y-mask) pair** from the pattern tables
  (`ADDR_BIT_SETTING`; the z/slice and s/sample masks vanish for 2D 1-sample). The low
  `log2(bpe)` bits are zero (byte-within-element).
- The offset is XOR-separable: `off(x,y) = fx(x) ^ fy(y)` — prosper precomputes `fx[]`/`fy[]`
  per row/column (`src/gpu/tile.cpp: sw64kb_copy`).

### SW_64KB_S (tile_mode 9) — element order per bpe

Identical for every pipe count and for RB+ chips (verified across all 35/75 PATINFO rows):

| bpe | offset bits (low → high, above the byte bits) |
|-----|-----------------------------------------------|
| 1 B  | x0 x1 x2 x3 y0 y1 y2 y3 · y4 x4 y5 x5 · y6 x6 y7 x7 |
| 2 B  | x0 x1 x2 y0 y1 y2 x3 · y3 x4 y4 x5 · y5 x6 y6 x7 |
| 4 B  | x0 x1 y0 y1 y2 x2 · y3 x3 y4 x4 · y5 x5 y6 x6 |
| 8 B  | x0 y0 y1 x1 x2 · y2 x3 y3 x4 · y4 x5 y5 x6 |
| 16 B | y0 y1 x0 x1 · y2 x2 y3 x3 · y4 x4 y5 x5 |

**Structural fact:** SW_64KB_S = the SW_4KB_S pattern *continued upward with (y,x) Morton pairs*.
The addrlib 4 KB rows reproduce prosper's two pixel-verified SW_4KB_S orders **exactly** (32-bpp
`[x0 x1 y0 y1 y2 x2 y3 x3 y4 x4]` from #118, and the 8-bpp L=4 font-atlas order) — a strong
correctness cross-check of the whole table-extraction pipeline.

### SW_64KB_R_X (tile_mode 27) — adds pipe XOR

Same block layout; the micro-tile is display-ordered, and offset bits 8..(8+log2(pipes)−1) become
**pipe bits**: `bit(8+i) = x(3+i) ^ y(3+i) ^ z(...)` style XOR terms (z = 0 for 2D). E.g. at 4 bpe
with 8 pipes: bits 8..10 = `x3^y3, x4^y4, x5^y5`, then bit 11 = `x3`, 12 = `y4`, 13 = `x5`,
14 = `y6`, 15 = `x6`. The full table for pipes ∈ {1..64} × bpe ∈ {1..16} is in
`src/gpu/tile.cpp: kSw64kRX`.

The **pipe count is the one free parameter** (fixed PS5 hardware, not publicly documented).
Default: **16 pipes** — PS5's Oberon is Navi10-class (36-40 CU, 256-bit GDDR6, 64 ROPs / 16 RBs),
and Navi10's `GB_ADDR_CONFIG` is 16 pipes. Override with `PROSPER_RX_PIPES=<1|2|4|8|16|32|64>`
for live A/B. `pipeBankXor` is assumed 0. CONFIDENCE: MED.

### SW_64KB_R_X 3D volumes

DOLL's UE4 lighting pipeline also uses mode-27 thin 3D images (not array textures), including
`120x68x32` Float16 volumes. Mesa AddrLib's 16-pipe 3D pattern adds `z3,z2,z1,z0` to byte-offset
bits 8..11 respectively. Each Z slice owns its own padded grid of 64KB 2D blocks, while those Z
bits still participate in the XOR inside the slice:

```
addr(x, y, z) = z * blocksPerSlice * 65536
              + block2D(x, y) * 65536
              + patternOffset(x, y, z)
```

`tiled_volume_bytes`, `detile_volume`, and `tile_volume` implement this layout for mode 27 at
1/2/4/8/16 bytes per texel. Unit tests cover exact round-trips at every size plus golden Z-bit
positions derived from AddrLib. Capture v9 retains each resource's base-level depth so standalone
replay uses the same volume footprint. CONFIDENCE: HIGH for the observed 16-pipe PS5 layout.

## Validation (agentic, offline)

1. `PROSPER_DUMP_TILERAW=1 PROSPER_FRAME_DIR=<dir>` dumps every sampled tiled texture's raw guest
   bytes at T#-decode time (now content-gated: an all-zero surface is re-probed until something
   writes it).
2. Offline (scratchpad `addrlib/`): `extract_patterns.py` parses the addrlib header into JSON;
   `detile64.py` applies a pattern to a dump, decodes BC1/BC4/fp16, writes a BMP, and scores
   total-variation coherence.
3. **SW_64KB_S result: a live 1024×512 BC1_SRGB (fmt 170) material texture decodes into a fully
   coherent sprite atlas** (foliage strips; crisp silhouettes, smooth gradients, zero block-weave).
   TV coherence 20.1 (read-as-linear) → 6.9 (detiled). The C++ (`tile.cpp`) output byte-matches
   the Python reference on that dump. CONFIDENCE: HIGH.
4. **SW_64KB_R_X:** every captured mode-27 surface with content is either fp16 speckle
   (bloom-chain buffers) or checkerboard-sparse — all pipe variants score within noise of each
   other on TV, mip-consistency, and lag-2 autocorrelation (linear is clearly worse, confirming
   the surfaces ARE 64KB-tiled). The pipe count therefore remains hardware-analogy (Navi10 ⇒ 16).
   Re-pin it when a mode-27 surface with smooth authored content appears (e.g. a UI banner) —
   sweep `PROSPER_RX_PIPES` and keep the coherent one.

## Where it lives

`src/gpu/tile.cpp` — `kSw64kS` / `kSw64kRX` tables + `sw64kb_copy` walk; dispatched from
`tile_mode_is_tiled` / `tiled_surface_bytes` / `detile_surface` / `tile_surface` /
`tiled_elements_bytes` / `detile_elements`. The SW_4KB_S path is untouched (The Messenger
regression-verified). Tests: `tests/test_tile.cpp` (round-trips at all 5 element sizes for both
modes + addrlib golden byte positions). Refs: #288 (this work), #282 (the reframe), #119/#118
(the 4KB groundwork).

## Remaining texture walls after #288 (NOT tiling)

Two of the three walls were closed by #290:
- ~~fp16 textures upload as RGBA8~~ **FIXED (#290)**: fp16 surfaces (fmt 71 Float16×4, 29, 13) are
  now read at the real bytes-per-texel, detiled with the real element size, and converted
  half→UNORM8 (exact binary16 decode, unit-tested; values clamp to [0,1] — native
  `R16G16B16A16_SFLOAT` upload for HDR >1.0 energy is a documented follow-up).
- ~~BC4/BC5/BC7 bindings are skipped~~ **FIXED (#290)**: full BC4/BC5/BC7 decoders in
  `bc_decode.cpp`, fuzz-validated byte-exact vs texture2ddecoder (4000 BC7 blocks covering all 8
  modes + 1000 each BC4/BC5). Live DOLL BC7_SRGB atlases (fmt 182, 1024×1024 / 2048×1024, mode 9)
  decode through `detile_elements` + `bc_decode_surface` into fully coherent cloud sprite atlases
  (TV ≈ 1.6). Still skipped: **BC6H** (fmt 179/180, HDR half-float — no decode) and **SNORM
  BC4/BC5** (fmt 176/178 — the UNORM8 upload can't carry signed samples).
- **RT sampling — largely addressed (#294)**: rendered pixels live host-side in `g_rtt`
  (`PROSPER_RTT` / `PROSPER_RTT_PERTARGET` injection). Previously fmt 36 (`10_11_11_FLOAT`, the
  UE4 R11G11B10F scene color) was unmapped and dropped, and the layout-level RTT bind only checked
  `PROSPER_RTT` — so a pertarget-only run skipped the fmt-36 T# entirely. Now:
  - **fmt 36 is mapped** to `DataFormat::Float10_11_11` (3 comps, 4 B/texel) unconditionally in
    `gen5_image_format`; the live_renderer unpacks the packed word to RGBA8 (`f11_to_float` /
    `f10_to_float`, same NaN/clamp semantics as the fp16 path) for the non-RTT-hit case. Kept out
    of `rdna2_buffer_format` (vertex fetch has no packed conversion). Unit-tested.
  - **The RTT env gate is unified**: `agc_shader_layout`'s unmapped-format bind now accepts
    `PROSPER_RTT_PERTARGET` too, matching `live_renderer`'s `rtt_on`.
  - **Result** (live DOLL capture, `PROSPER_RTT_PERTARGET=1`): 0 `UNMAPPED` skips, the scene-color
    fmt-36 targets (1920×1080 / 3840×2160) and the fp16 bloom chain (960×540…60×34) all resolve as
    RTT **HITs** (497 fmt-20 scene-color HITs) — the composite samples real rendered pixels, not
    NaN speckle. The `PROSPER_RTT=1` single-framebuffer flatten renders a coherent smooth
    yellow-green radial bloom gradient (587 distinct colors) instead of striped garbage.
  - **Remaining**: the presented frame is not yet a recognizable scene. Under pertarget, the
    "present the last group" heuristic surfaces a solid-blue clear/UI pass; under single-FB the
    output is a smooth bloom gradient with no scene geometry detail — i.e. the scene-color RT
    carries coherent post-process content but the underlying scene draws don't yet write
    recognizable geometry into it. Next wall is scene-geometry detail in the scene-color RT (and a
    present-selection heuristic that picks the composite group, not the last small pass), not the
    RT-sampling plumbing.
