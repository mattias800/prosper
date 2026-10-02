# `state` — fixed-function render state

- `render_state` — the guest's render state as decoded from its register writes: blend, depth,
  raster, target formats and extents.
- `vk_translate` — mapping that state onto Vulkan enums, formats and pipeline structures.
- `fragment_entry_facts` — physical per-word PS USER_DATA presence and raw RSRC2 knownness from
  the producing draw. A present zero is not an absent word. This does not map observations to
  launched guest SGPRs, system parameters, M0 or VGPRs; the packet preflight names those gaps.

Small folder, outsized blast radius: state that never reaches the GPU produces output that looks like
a *shading* bug. One recorded case had solid-block glyphs, a black logo panel and a flat sky that were
all one defect — the RT0 blend state was never submitted.

When adding a translation, prefer failing visibly on an unknown enum over mapping it to a plausible
default.
