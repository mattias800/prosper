# `recompiler/indirect` — bounded guest pointers loaded at runtime

Proves producer/guard/consumer structure for pointers a shader loads **into VGPRs** and dereferences
per invocation, and derives one bounded record per real invocation.

- `rdna2_indirect_pointer_analysis` — the proof: which fetch produced the pointer, what guards it,
  which consumers use it, and at what widths.
- `rdna2_indirect_pointer_relocation` / `_descriptor_range` — turning that proof into something
  bindable.
- `rdna2_indirect_buffer_shadow` — the carrier that owns snapshot layout and lifetime.

**This is not the descriptor-table path.** A descriptor loaded with `s_load`/`s_buffer_load` from a
user-data pointer at a constant immediate is a *scalar*, compile-time-constant offset, and is
const-folded in `execute/` (see the `SrtUse` contract in `execute/gpu_execute.hpp`). This folder
exists for the harder case where the address is not known until the wave runs.

It deliberately does **not** grant arbitrary FLAT access: every relocated consumer keeps its exact PC,
packet, address pair, width and immediate in the proof.
