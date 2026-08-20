# `texture` — guest texture memory

How pixels are actually laid out in guest memory, and how to get them into a linear Vulkan image.

- `tile` — the tiling/swizzle modes, including mip-tail handling.
- `guest_texture_layout` — per-level offsets, pitches and sizes.
- `bc_decode` — block-compressed format decode.

Tiling bugs are **visually distinctive and diagnostically misleading**: the content is present and
correctly coloured but spatially scrambled, which reads as a geometry or UV problem. If a surface
looks like the right image cut into blocks, start here.

Layouts are generic where possible. A title-specific tiling special case is a strong signal the
general rule is wrong — one recorded fix replaced a per-title hack with generic 4 KiB mip-tail tiling
and resolved a whole title at once.
