# `resources` — what a shader actually binds

Turns descriptors into resolved, bindable resources, and answers questions about their provenance.

- `shader_resources` — `ShaderResource` / `ShaderResourceTable` and the three lookups the recompiler
  matches against: by fetch PC, by SRT byte offset (`by_srt_offset`), and by SGPR base
  (`by_sgpr_base_cls`). A resource matched by none of them is invisible to every lookup however
  correct its address and size are.
- `gpu_resources` — the resolved resource layer over guest memory.
- `compressed_source_authority` — who is authoritative for a compressed surface's bytes.
- `metadata_kind_correlation` — correlating a surface's metadata kind with how it is used.

The three-lookup contract is the thing to understand first: most "the descriptor is right but nothing
binds" investigations end at a resource whose key does not match the key its consumer looks up.
