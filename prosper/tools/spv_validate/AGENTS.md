# SPIR-V emitter validation

`spv_validate` emits representative modules and validates them with spirv-val; its coverage
inventory also rejects a producing entry point without a validated module.

`arbitrary_placeholder_witness` exercises the delegating admission entry directly using the
shared synthetic register fixture. It requires actual chosen-module admission metadata; an
ordinary strict success cannot stand in for this path. No Vulkan device or game is needed.

The allocated MOVRELS witnesses validate both Wave32 and Wave64 declared-bank/OOB modules
through the existing `recompile_compute` selector. Unknown allocations stay outside this route.

`native_fragment_collective_witness` validates the actually selected native READFIRST retry in
both widths and exact/WQM execution. Required module modes and admission metadata exclude an
ordinary legacy success from its coverage; it delegates through the normal fragment compiler.
