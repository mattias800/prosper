# SPIR-V emitter validation

`spv_validate` emits representative modules and validates them with spirv-val; its coverage
inventory also rejects a producing entry point without a validated module.

`arbitrary_placeholder_witness` exercises the delegating admission entry directly using the
shared synthetic register fixture. It requires actual chosen-module admission metadata; an
ordinary strict success cannot stand in for this path. No Vulkan device or game is needed.
