# `texture` — guest texture memory

How pixels are actually laid out in guest memory, and how to get them into a linear Vulkan image.

- `tile` — the tiling/swizzle modes, including mip-tail handling.
- `guest_texture_layout` — per-level offsets, pitches and sizes.
- `bc_decode` — block-compressed format decode.

The 64 KiB mip-tail walker hoists separable x/y swizzle contributions into a bounded stack table
for views up to 256 columns. It retains global coordinates, partial backing checks and neighboring
mip bytes; wider callers use the scalar equation. `PROSPER_NO_SEPARABLE_MIP_TAIL=1` restores the
scalar path for both reads and writes. `test_mip_tail` checks placement against the full-surface
walker; its Python companion compares every output byte against a fresh scalar process across
pipe counts. Compute writeback timing includes tail coordinates and backing bytes so an array
fallback can be distinguished from an actual packed-tail conversion before proposing GPU retile.

Tiling bugs are **visually distinctive and diagnostically misleading**: the content is present and
correctly coloured but spatially scrambled, which reads as a geometry or UV problem. If a surface
looks like the right image cut into blocks, start here.

Layouts are generic where possible, and a title-specific tiling special case is a strong signal the
general rule is wrong. Worked example: #1578 (`13908a66`) fixed a wrong block-to-element multiplier
in the **shared generic** path and resolved a whole title at once — the defect was in the rule
everyone used, not in a per-title hack.
