# `state` — fixed-function render state

- `render_state` — the guest's render state as decoded from its register writes: blend, depth,
  raster, target formats and extents.
- `vk_translate` — mapping that state onto Vulkan enums, formats and pipeline structures.

Small folder, outsized blast radius: state that never reaches the GPU produces output that looks like
a *shading* bug. One recorded case had solid-block glyphs, a black logo panel and a flat sky that were
all one defect — the RT0 blend state was never submitted.

When adding a translation, prefer failing visibly on an unknown enum over mapping it to a plausible
default.
