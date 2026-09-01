# `resources` — what a shader actually binds

Turns descriptors into resolved, bindable resources, and answers questions about their provenance.

- `shader_resources` — `ShaderResource` / `ShaderResourceTable`. The table exposes **five** lookups
  (`by_srt_offset`, `by_sgpr_base`, `by_sgpr_base_cls`, `by_fetch_pc`, `by_binding`, all declared
  together in `shader_resources.hpp`); the recompiler uses four of them — by fetch PC, by
  SRT byte offset (`by_srt_offset`), and by SGPR base in both the class-qualified
  (`by_sgpr_base_cls`) and bare (`by_sgpr_base`) forms.
  A resource carrying `srt=0xffffffff sgpr=0xffffffff` is unreachable by the three **key-based**
  lookups however correct its address and size are — which is the usual ending of a "the
  descriptor is right but nothing binds" investigation.
  It is **not** unreachable in general, and getting that wrong sends you hunting a binding that is
  in fact resolving: `by_fetch_pc` consults neither field, so a resource with both sentinels plus a
  valid fetch PC still resolves — the header defines four such shapes, and the recompiler retrieves
  them live — `grep -rn by_fetch_pc src/gpu/recompiler/` finds the call sites, including the
  `by_fetch_pc(153u)` selected-descriptor lookup. `by_binding` matches anything in the table at
  all.
- `mip_chain_plan` — where every level of a declared mip chain lives, and how many of them a
  compute image materializes. `ShaderResource` describes the SELECTED level (`image_base_level_view`
  shifts `gpu_addr`/`width`/`height` onto it), so the other levels have no placement anywhere else;
  this is the one derivation that recovers them. It is deliberately read by **both** the backend
  that creates the image and the recompiler that lowers `IMAGE_LOAD_MIP` — if those two ever answer
  the level count differently, a shader fetches a level that was never created.
- `gpu_resources` — the resolved resource layer over guest memory.
- `compressed_source_authority` — who is authoritative for a compressed surface's bytes.
- `metadata_kind_correlation` — correlating a surface's metadata kind with how it is used.

The lookup contract is the thing to understand first: most "the descriptor is right but nothing
binds" investigations end at a resource whose key does not match the key its consumer looks up.
