# Descriptor value-decode audit — prosper V#/T#/S# vs the RDNA2 ISA (2026-07)

A systematic sweep of prosper's resource-descriptor **value tables and enum mappings** against the
RDNA2 ISA, run against master `c24475d` on 2026-07-18. Third in the GPU-audit series after the RDNA2
ISA recompiler audit (`RDNA2_ISA_AUDIT_2026_07.md`, #878–#883) and the PM4 register audit
(`PM4_REGISTER_AUDIT_2026_07.md`, #911). Tracked by #912.

## Result: clean — zero defects

Across all five reviewed areas the sweep produced **no confirmed findings** (and none refuted or
uncertain). The descriptor value-decode layer is correct.

## Scope and the Gen5 boundary

prosper's descriptor decode is a **mix** of vanilla-RDNA2 behavior and PS5 **Gen5-specific** behavior.
The Gen5-specific parts — the combined 7-bit V# `FORMAT` field at dword3[18:12] (vs the vanilla
NFMT/DFMT split; cited to Kyty `Shader.h:585`), the `MAX_MIP=BASE-1` storage-view form (cited to a
live Astro 1920×1080 descriptor), the width/height split fields, the `dst_sel` bit positions — are
**validated by live PS5 boots** and are *deliberately excluded*: this project's rule is to never
weaken behavior demonstrated by a live boot to match a secondary/vanilla reference. Sweeping those
field positions against vanilla RDNA2 would produce false positives, so the audit targeted only the
genuinely doc-verifiable, value-level semantics (a wrong entry there silently corrupts decode):

- **Buffer format value table** (`rdna2_buffer_format`) — the format-code → (DataFormat, component
  count) mapping. The code self-declared `CONFIDENCE: HIGH` only on the four anchors 56/64/74/77 and
  `MED` on the rest, so the non-anchor entries were the real target.
- **Image format table + size helpers** — `Gen5ImageFormatInfo`, `data_format_bytes`, BC block sizes.
- **Swizzle** — the `dst_sel` code → `VkComponentSwizzle` enum mapping.
- **Sampler (S#) enums** — depth-compare function, mag/min/mip filter, address/clamp modes.
- **RDNA2-inherited buffer/image fields** — `num_records`/stride semantics, base-48, bounds.

## Per-area coverage (all verified against the ISA)

- **Buffer formats vs Table 47:** all 44 case labels (1,2,5,6,7,8,11,12,13,14,15,18,19,20,21,22,23,24,
  27,28,29,36,50,51,54,55,56,57,60,61,62,63,64,65,66,69,70,71,72,73,74,75,76,77) map to the correct
  format family, signed/norm/float class, **and** component count. The three called-out traps are all
  handled correctly: the `2_10_10_10` codes 50/51/54/55 map to the 2_10_10_10 DataFormats (the
  `10_10_10_2` codes 44/45/48/49 are deliberately unmapped, not mis-ordered); the USCALED/SSCALED
  codes fall through to `Unknown` rather than aliasing UINT/SINT (and 52/53 additionally fail closed);
  code 36 (`10_11_11_FLOAT`) is distinguished from code 43 (`11_11_10_FLOAT`, unmapped).
- **Image formats / bytes:** `data_format_bytes` is internally consistent; `plain()` computes
  `bytes = components × component_bytes` by construction; SRGB (128/129/130) carry the right base
  format and size; every BCn block byte size matches Table 47 and the standard (BC1/BC4 = 8;
  BC2/3/5/6/7 = 16) with 4×4 block dims. (BC *decoders* themselves are audited separately under #914.)
- **Swizzle (Table 45):** `dst_sel` decoded from WORD3 [2:0]/[5:3]/[8:6]/[11:9], and the sole
  code→component site maps 0→ZERO, 1→ONE, 4→R, 5→G, 6→B, 7→A verbatim per the ISA; reserved 2/3 fall
  to IDENTITY (never emitted).
- **Sampler enums (Table 46):** `DEPTH_COMPARE_FUNC` 0..7 is numerically identical to `VkCompareOp`
  and cast directly (correct for all 8); mag/min filter point(0)/linear(1) map correctly (aniso codes
  2/3 fold to linear — a documented approximation tracked under #275, not a value error); address modes
  correct.
- **RDNA2-inherited fields:** `num_records`/stride sizing (64-bit, wrap-clamped) and base-48 are
  correct; the Gen5-specific fields were excluded as live-validated per the scope rule.

## Value of a clean result

The descriptor value tables were already partly proven by live rendering (the exercised titles decode
textures and vertices correctly, so the anchor formats are demonstrably right). This audit extends that
confidence to the less-common codes — packed/scaled formats, component order, SRGB, BC block sizing,
and the sampler/swizzle enums — that live rendering does not routinely exercise. Recording it here so a
future descriptor change starts from "the value layer is verified against the ISA" rather than
re-auditing it. The Gen5-specific *layout* remains governed by live evidence, not this reference.
