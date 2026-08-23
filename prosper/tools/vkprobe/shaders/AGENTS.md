# tools/vkprobe/shaders

**SPIR-V that prosper did not generate.**

`tools/vkprobe` separates prosper's host-side Vulkan usage from everything else by executing
prosper's own modules in a bare process. That leaves the modules themselves inside the suspect set —
if the recompiler emits SPIR-V that is malformed, or that leans on something the spec does not
guarantee, the control fails for a reason the control cannot see. This folder is the arm that closes
that gap: hand-written modules that do the *same work* through the *same descriptor shape*, and were
written from the SPIR-V specification rather than emitted by any part of prosper.

They are deliberately assembly (`.spvasm`, assembled by `spirv-as`) and not GLSL. The question these
answer is about decorations, storage classes and access chains — `NonWritable`, `Aliased`, the exact
`OpAccessChain` arithmetic — and a GLSL compiler chooses those for you. Hand assembly is the only
way to vary one of them at a time, which is what the family below is for.

## The family, and what each one isolates

| module | what it does | the question it answers |
| --- | --- | --- |
| `minimal_ssbo_vs.spvasm` | the same storage-buffer loads as prosper's dumped vertex module, same set 0 binding 3, same `gl_VertexIndex * 8` addressing | is prosper's *generated* SPIR-V the cause? |
| `no_ssbo_vs.spvasm` | identical descriptor layout, positions from `gl_VertexIndex` alone — the storage buffer is declared and never read | does the failure need a storage-buffer load at all? |
| `index_readback_vs.spvasm` | stores `gl_VertexIndex + 1` into the record buffer, so the host can see which vertex indices the shader was handed | did the shader run, and with which indices? |
| `minimal_green_fs.spvasm` | opaque green, matching what prosper's dumped fragment module writes | keeps `vkprobe`'s coverage predicate reading both families identically |

Every one of them renders the *same* triangle when it works — the geometry `vkprobe`'s default
records produce at an 8-byte stride, 496 covered pixels of a 64x64 target — so coverage is
comparable straight across the table and against prosper's own module.

## What belongs here

Only modules whose value comes from **not** being prosper's output. If you find yourself dumping a
module out of a capture, it belongs in a scratch directory and gets passed to `vkprobe` with
`--vs`; the point of these is that they have no provenance in the code under test.

Keep them minimal in the strict sense: a module here should contain nothing whose removal would
still let it express the failure. Every instruction that survives is one a reader has to rule out.

## They are validated by a test, and that is not ceremony

`check_controls.py` assembles each source, pins the result to SPIR-V 1.3 (the version prosper's
recompiler emits, so the two families are compared at the same version), and runs `spirv-val` over
it. It is registered as the ctest case `vkprobe_hand_written_controls`.

The reason is the parent folder's own history: `vkprobe` spent weeks reporting confident coverage
numbers from pipelines that were **invalid**, because nothing checked and it loads no layers. A
control's whole claim is "this one is known good", and an unvalidated control cannot make it. If
`spirv-as` or `spirv-val` is missing the case is not registered and CMake says so out loud — a gate
that passes because its checker is absent is worse than no gate.
