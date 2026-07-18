# BC texture decode audit — Khronos S3TC/RGTC/BPTC (2026-07)

A systematic correctness sweep of `src/gpu/bc_decode.cpp`, run against master `ebe6ab8` on
2026-07-18 and tracked by #914. The review covered every decoder that the upload path exposes:
BC1, BC2, BC3, unsigned BC4/BC5, BC6H UF16, and BC7.

## Result

No wrong-texel discrepancy was found in the implemented formats. Endpoint order, index order,
interpolation, partitions, anchors, p-bits, BC7 rotation/index selection, and all 14 valid BC6H modes
agree with the normative format descriptions and checked-in known-block expectations.

The audit and PR review found three compiler-portability correctness defects and two misleading
declarations:

- BC6H delta sign extension used `(val << (32 - bits)) >> (32 - bits)`. A valid negative delta can
  make the signed left shift overflow `int`, which is undefined C++ behavior even though the tested
  compilers happened to produce the intended result. The decoder now subtracts the field modulus
  when the sign bit is set, which is defined for every BC6H endpoint width.
- BC6H loaded its 128-bit input by casting arbitrary byte storage to `unsigned long long*`, which
  could violate both alignment and aliasing requirements. It now copies the two words with `memcpy`,
  and a regression test decodes a known block from an intentionally misaligned address.
- BC6H's half-to-float helper reinterpreted integer and float bits by reading inactive union members.
  It now uses C++20 `std::bit_cast`, including unsigned operands for the bit shifts.
- The BC7 "per-endpoint p-bit" mask was `0xCB`, which also set mode 1. Mode 1 actually has shared
  per-subset p-bits; a preceding mode-1 branch prevented wrong output, but the mask contradicted its
  name and the spec. It is now the exact per-endpoint mask `0xC9` and mode 1 remains explicit.
- The public header's supported-format list omitted the already-supported BC6H decoder. The list now
  matches the implementation.

## Authoritative references

The primary source was Khronos Data Format Specification 1.4 at commit
[`661f4ef`](https://github.com/KhronosGroup/DataFormat/tree/661f4ef60a16c428fa1ed00e2e436b96bf7c51f7):

- [S3TC / BC1–BC3](https://github.com/KhronosGroup/DataFormat/blob/661f4ef60a16c428fa1ed00e2e436b96bf7c51f7/s3tc.txt)
- [RGTC / BC4–BC5](https://github.com/KhronosGroup/DataFormat/blob/661f4ef60a16c428fa1ed00e2e436b96bf7c51f7/rgtc.txt)
- [BPTC / BC6H–BC7](https://github.com/KhronosGroup/DataFormat/blob/661f4ef60a16c428fa1ed00e2e436b96bf7c51f7/bptc.txt)

The independent reference implementation inspected during the audit was the Khronos CTS decoder at commit
[`5fa4361`](https://github.com/KhronosGroup/VK-GL-CTS/blob/5fa43613e238890232280400eac48723f5afea53/framework/common/tcuCompressedTexture.cpp).

The existing decoder predates this audit and carries its own legacy provenance in the source:
the BC7 constants are attributed to `bcdec` (Sergii Kudlai, MIT/Unlicense), while the BC6H logic and
constants are described as adapted from it. The legacy comments do not pin a `bcdec` revision, and
this audit does not claim otherwise. This PR copies no new external implementation or test corpus;
its new vectors are locally generated blocks with expected bytes derived from the Khronos equations.

## Coverage and evidence

| Format | Fields and cases checked | Bit-exact evidence |
| --- | --- | --- |
| BC1 | RGB565 expansion, little-endian endpoints/indices, four-color interpolation, three-color + punch-through branch | Complete known-block output plus focused endpoint/interpolation/transparent-index cases |
| BC2 | BC1 forced four-color path, 16 direct alpha nibbles, nibble expansion and texel order | Complete block containing alpha values 0 through 15 |
| BC3 | Forced four-color path, both alpha endpoint orderings, 48-bit 3-bit-index stream, 7-step and 5-step ramps | Complete 0..7 ramp block plus endpoint-index cases |
| BC4 | Unsigned endpoint comparison, both ramps, literal 0/1 entries, red-only channel result | Complete ramp block with every decoded texel pinned |
| BC5 | Two independent BC4 streams, channel order, `(R,G,0,1)` result | Complete opposing-ramp block with both channels pinned |
| BC6H UF16 | 14 mode codes, endpoint bit layouts, transformed/direct endpoints, 32 partitions and anchors, 3/4-bit indices, unquantize/interpolate/final-unquantize, reserved modes, HDR-to-RGBA8 clamp | One complete non-trivial known block per valid mode, plus a misaligned-input regression |
| BC7 | Eight mode codes, 2/3-subset partitions and anchors, endpoint widths, shared/per-endpoint p-bits, 2/3/4-bit weights, secondary indices, rotation, index selection, reserved mode | One complete non-trivial known block per mode |

The permanent test contains only fixed block bytes and expected RGBA bytes. Its expected values were
derived from the Khronos equations, so CI needs no reference decoder, downloaded corpus, GPU, game
dump, renderer, or snapshot.

## Intentional boundaries

- BC4/BC5 SNORM and BC6H SF16 are recognized by descriptor decode but deliberately skipped before
  this unsigned-RGBA8 upload adapter. This audit does not claim support for those signed variants.
- BC6H's final conversion to clamped RGBA8 is prosper's existing upload policy, not part of BC6H's
  native HDR representation. The permanent vectors pin that policy rather than claiming it as a
  native BC6H representation.
- S3TC/RGTC implementations can differ by one unit when their normalized results are converted back
  to 8-bit channels. Prosper retains the Khronos CTS choices already used by its decoder; the new
  known blocks pin those choices instead of silently switching to a vendor-specific rounding rule.

## Permanent regression gate

`tests/test_bc_decode.cpp` now checks complete 4×4 RGBA output for BC1–BC5 and for every BC6H/BC7
mode, in addition to its focused hand-built cases. Run it directly with:

```text
cmake --build build-windows --target test_bc_decode
build-windows/test_bc_decode.exe
```

No snapshot or rendered-output workflow is required for this pure decoder audit.
