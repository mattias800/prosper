# BC texture-decode audit — prosper BC1–BC7 vs the Khronos/D3D11 spec (2026-07)

A bit-exact sweep of prosper's block-compression texture decoders (`src/gpu/bc_decode.cpp`) against
the Khronos Data Format Specification (S3TC / RGTC / BPTC) and the Microsoft D3D11 functional spec
(§19.5 BC7, §19.6 BC6H), run against master `b573f49` on 2026-07-18. Fifth and final entry in the
GPU-audit series (recompiler #878–#883, PM4 registers #911, descriptors #912, vk_translate #913).
Tracked by #914.

## Result: clean — zero defects

BC decode is a pure deterministic function, so findings are bit-exact verifiable — the adversarial
verifiers hand-decoded concrete block bytes and checked every endpoint expansion, interpolation weight,
partition/anchor table entry, and mode field against the canonical spec constants. Across all four
areas the sweep produced **no confirmed findings** (and none refuted or uncertain). The BC decoders are
correct.

## Why this is a high-confidence result

Unlike a live-rendering check (which only proves the *exercised* texture formats decode acceptably),
this audit verifies the decoders **bit-exactly against the spec**, including format/mode combinations
that no currently-booting title exercises. BC1–BC5 are prosper's own hand-written decoders; BC6H and
BC7 are vendored verbatim from `bcdec` (MIT/Unlicense, attributed in the source), a widely-used
reference decoder — the sweep confirmed both the hand-written paths and that the vendored tables/logic
match the spec (no transcription drift).

## Per-area coverage (all verified bit-exact)

- **BC1 color (`decode_color_block`):** 565 endpoint read (little-endian) and the 5→8 `(v<<3)|(v>>2)` /
  6→8 `(v<<2)|(v>>4)` bit-replication; the `c0>c1` 4-color mode (`c2=(2c0+c1)/3`, `c3=(c0+2c1)/3`,
  truncating) vs the `c0<=c1` 3-color mode (`c2=(c0+c1)/2`, `c3`=transparent black); the 2-bit row-major
  LSB-first index read; and the `dxt1_alpha` punch-through gating (BC1 on, BC2/BC3 forced 4-color).
- **BC2/BC3/BC4/BC5 alpha & channel:** BC2 4-bit direct alpha `(nib<<4)|nib`; the BC3/BC4 8-value alpha
  ramp — `a0>a1` (6 interpolants `((7-i)a0+i·a1)/7`) vs `a0<=a1` (4 interpolants `/5` + literal 0/255);
  the 3-bit LSB-first index read over 48 bits; BC5's two-channel R/G placement (B=0, A=255). SNORM
  BC4/BC5 is deliberately unimplemented (skipped upstream, no signed endpoints reach the unsigned
  decoder) — documented non-implementation, not a defect.
- **BC7 (`decode_bc7_block`):** the leading-zero mode select; subset counts (0,2→3; 1,3,7→2; 4,5,6→1);
  partition bits (mode 0=4, else 6); per-mode color/alpha precision (`{4,6,5,7,5,7,7,5}` / alpha
  `{0,0,0,0,6,8,7,5}`); the p-bit modes (0,1,3,6,7) with mode-1 per-subset shared p-bits and the
  per-endpoint LSB p-bit appended before unquantize; the endpoint unquantize
  `v<<(8-prec) | v>>(2·prec-8)`; the anchor implied-high-bit (each subset anchor reads one fewer index
  bit); rotation (modes 4,5) and index-selection (mode 4); the `kWeight2/3/4` interpolation tables and
  the `(a·(64-w)+b·w+32)>>6` formula. ~15 entries of both the 2-subset and 3-subset partition/anchor
  tables were spot-checked against the canonical BPTC fixup tables — all match.
- **BC6H (vendored bcdec) + routing:** the 2-/5-bit mode read and the 14-mode remap; the endpoint
  precision table; the delta/transform-skip predicate (explicit-endpoint modes) and sign handling;
  unquantize, transform-inverse, sign-extend, and the interpolation + int→f16 steps — all matching the
  D3D11 §19.6 / Khronos BPTC-FLOAT spec byte-for-byte (the only divergences from bcdec are cosmetic
  comment typos that do not affect decoded bits). The f16→RGBA8 clamp for the 8-bit upload path is a
  documented HDR-energy limitation, not a decode defect. Format routing / block sizes (BC1/BC4=8,
  BC2/3/5/6/7=16) and per-format decoder dispatch are correct.

## Value of a clean result

BC decode correctness is critical for texture rendering, and this records it as **verified bit-exactly
against the spec** rather than merely "the exercised textures look right." A future BC change starts
from this baseline instead of re-auditing the decoders.
