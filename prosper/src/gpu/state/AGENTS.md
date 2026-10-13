# `state` — fixed-function render state

- `render_state` — the guest's render state as decoded from its register writes: blend, depth,
  raster, target formats and extents.
- `vk_translate` — mapping that state onto Vulkan enums, formats and pipeline structures.
- `guest_depth_plane` — how many bytes of guest memory a depth plane occupies, from
  `DB_Z_INFO.FORMAT`. The host image is D32 for every guest format, so this is the only place that
  knows a Z_16 plane is half the size.
- `fragment_entry_facts` and `raster_launch_facts` live in `gpu/recompiler` now: they are inputs the
  recompiler reads (ADR 0006 / GPU-1), so they sit with it rather than in the layer that calls it.

Small folder, outsized blast radius: state that never reaches the GPU produces output that looks like
a *shading* bug. One recorded case had solid-block glyphs, a black logo panel and a flat sky that were
all one defect — the RT0 blend state was never submitted.

When adding a translation, prefer failing visibly on an unknown enum over mapping it to a plausible
default.
