# `resources` — what a shader actually binds

Turns descriptors into resolved, bindable resources, and answers questions about their provenance.

- `shader_resources` — `ShaderResource` / `ShaderResourceTable`. The table exposes **five** lookups
  (`shader_resources.hpp:705,708,713,716,718`); the recompiler uses four of them — by fetch PC, by
  SRT byte offset (`by_srt_offset`), and by SGPR base in both the class-qualified
  (`by_sgpr_base_cls`) and bare (`by_sgpr_base`) forms.
  A resource carrying `srt=0xffffffff sgpr=0xffffffff` cannot be reached by any of those four
  however correct its address and size are — which is the usual ending of a "the descriptor is
  right but nothing binds" investigation. It is still reachable by `by_binding`, which matches
  anything in the table, so "invisible to every lookup" would be too strong.
- `gpu_resources` — the resolved resource layer over guest memory.
- `compressed_source_authority` — who is authoritative for a compressed surface's bytes.
- `metadata_kind_correlation` — correlating a surface's metadata kind with how it is used.

The three-lookup contract is the thing to understand first: most "the descriptor is right but nothing
binds" investigations end at a resource whose key does not match the key its consumer looks up.
