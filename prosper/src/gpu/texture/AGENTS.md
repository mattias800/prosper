# `texture` — guest texture memory

How pixels are actually laid out in guest memory, and how to get them into a linear Vulkan image.

- `tile` — the tiling/swizzle modes, including mip-tail handling.
- `guest_texture_layout` — per-level offsets, pitches and sizes.
- `bc_decode` — block-compressed format decode.

Tiling bugs are **visually distinctive and diagnostically misleading**: the content is present and
correctly coloured but spatially scrambled, which reads as a geometry or UV problem. If a surface
looks like the right image cut into blocks, start here.

Layouts are generic where possible, and a title-specific tiling special case is a strong signal the
general rule is wrong. Worked example: #1578 (`13908a66`) fixed a wrong block-to-element multiplier
in the **shared generic** path and resolved a whole title at once — the defect was in the rule
everyone used, not in a per-title hack.
