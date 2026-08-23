# AGENTS.md — prosper/tools/index_fetch_probe

Standalone bare-Vulkan reproducer for **#2961**: on RADV/STRIX_HALO, indexed
draws hand the vertex shader the sequential vertex ordinal instead of the
fetched index-buffer value (deterministic, no GPU load needed). Deliberately
shares **no code** with vkprobe — the point of the tool is to answer whether
the divergence lives in the driver or in vkprobe's harness, so importing
vkprobe's pipeline setup would defeat the experiment.

- `probe.c` — the whole program. Vulkan 1.0, no extensions, no layers, one
  64×64 offscreen render pass. A POINTS draw records `(gl_VertexIndex,
  fetched attribute)` per vertex into an SSBO keyed by the fetched value; the
  vertex buffer is identity, so a correct driver writes every indexed slot as
  its own index. Exit 0 fetched / 1 divergence / 2 tool error. The identity
  control list runs first: both hypotheses agree on it, so it proves the
  harness drew without discriminating.
- `shaders/*.spvasm` — readable SPIR-V assembly (assemble with `spirv-as
  --target-env vulkan1.0`); `probe_vert_spv.h` / `probe_frag_spv.h` are the
  generated word lists the binary embeds, so building needs no shader
  compiler. Regenerate the headers after touching the asm.

Results and the run environment are recorded on #2961; update that issue when
a new driver generation is tested.
