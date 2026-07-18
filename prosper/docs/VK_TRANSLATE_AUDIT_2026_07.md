# AGC→Vulkan enum-translation audit — prosper vs RDNA2 enums + Vulkan spec (2026-07)

A systematic sweep of prosper's render-state **enum translation** (RDNA2 register enum value → Vulkan
enum value), run against master `a449baf` on 2026-07-18. Fourth in the GPU-audit series (after the RDNA2
ISA recompiler #878–#883, PM4 registers #911, and descriptors #912). Tracked by #913.

## Scope and the two-sided reference

Every mapping here takes an RDNA2 register enum **code** and must produce the correct **Vulkan enum
value** — verified on both sides:

- **RDNA2 side:** the gfx10.3 enums extracted from the Mesa AMD register database — `BlendOp` (blend
  factor codes 0–20), `CombFunc` (blend combine 0–4), `CompareFrag` (0–7), `StencilOp` (0–15),
  `VGT_DI_PRIM_TYPE`, plus ISA Table 47 for color formats.
- **Vulkan side:** the stable spec enum values (`VkBlendFactor`, `VkBlendOp`, `VkCompareOp`,
  `VkStencilOp`, `VkPrimitiveTopology`, `VkCullModeFlags`, `VkFrontFace`, `VkFormat`).

This subsystem has a proven history of silent enum-mapping bugs (#534 reversed `VkFrontFace`, #654
RectList topology), which is why it was swept.

## Result: 1 confirmed fix, otherwise clean

Across the blend-factor table (21 codes), blend combine, compare/stencil ops, cull/front-face,
topology, and the color-format matrix, the sweep produced **one confirmed defect** (fixed here) and
verified everything else correct.

### Confirmed (fixed in this PR)

**[MEDIUM] `COLOR_5_6_5` `comp_swap` STD/ALT reversed** — `vk_translate.cpp:128`.

- **Convention (verified against the live-anchored rows):** `SWAP_STD` places R at the LOW end. This is
  proven by the two anchor rows the sweep cross-checked — `0xA` `8_8_8_8` (live-verified, HIGH
  confidence) STD → `R8G8B8A8` (R at byte 0), and `0x9` `2_10_10_10` STD → `A2B10G10R10` (R in bits
  [9:0]). For a 565 packed word, R-at-low is Vulkan `B5G6R5_UNORM_PACK16` (R in bits [4:0]).
- **Defect:** the 565 row returned `R5G6B5` (R in the HIGH bits [15:11]) for STD and `B5G6R5` for ALT —
  the two arms swapped, the lone violator of prosper's own STD=R-low convention. The pre-existing test
  had *enshrined* the reversed STD mapping.
- **Consequence:** a guest 5_6_5 UNORM color/render target programmed with `SWAP_STD` renders with red
  and blue channels swapped (and vice-versa for `SWAP_ALT`). Latent — no currently-booting title
  (Messenger, Dead Cells, Blasphemous 2) uses a 565 CB surface (the exercised rows are `0xA` 8888 and
  the packed HDR/2_10_10_10 rows), so rendered output for exercised titles is unchanged.
- **Fix:** swap the arms (STD → `B5G6R5`, ALT → `R5G6B5`), matching the convention; the regression test
  is corrected to the STD=R-low mapping and gains the ALT case.

## Per-area coverage (all other mappings verified clean)

- **Blend factor (`vk_blend_factor`, all 0–20):** correct, including the classic traps — DstColor
  RDNA2=8 → VK `DST_COLOR`=4 (not 8), `SRC_ALPHA_SATURATE` RDNA2=10 → VK=14, the dual-source `SRC1_*`
  factors (15–18 → 15–18), and the reordered CONSTANT_COLOR/ALPHA quad (13/14/19/20 → VK 10/11/12/13).
  Codes 11/12 (`BOTH_SRC_ALPHA`/`BOTH_INV_SRC_ALPHA`) have no single VkBlendFactor equivalent and fall
  through to ZERO — incompleteness, tracked in **#919**.
- **Blend combine (`vk_blend_op`):** correct including the subtract/reverse-subtract direction —
  `SRC_MINUS_DST`(1) → `SUBTRACT`(1, src-dst) and `DST_MINUS_SRC`(4) → `REVERSE_SUBTRACT`(2, dst-src),
  NOT transposed; MIN/MAX remap (2/3 → 3/4). `CB_COLOR_CONTROL.MODE`/`ROP3` (hardware logic-op) are not
  translated — incompleteness, tracked in **#919**.
- **Compare / stencil / cull / front-face:** `vk_compare_op` (direct mask, valid because `CompareFrag`
  0–7 == `VkCompareOp` 0–7); `vk_stencil_op` correctly remaps the RDNA2 `StencilOp` order to VkStencilOp
  (ONES/REPLACE_TEST/REPLACE_OP → REPLACE with the ref value chosen by #466; ADD/SUB_CLAMP → INCR/DECR);
  `cull_mode` bit assembly; and — critically — the `front_face` sense (the #534-class trap) is correct,
  verified against the register definition and the `test_pipeline_render` regression.
- **Topology / format:** `vk_topology` correct (RectList `7` → TriangleStrip is the #654 special-case,
  not a finding); `vk_color_format` `number_type` mapping and the local `VkFormat` enum values are
  correct (the 565 STD/ALT reversal above was the sole defect).
