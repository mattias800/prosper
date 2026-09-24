# Mesh backend workgroup fixture

`mesh_workgroup_probe.mesh` is the source for four module arrays in the tracked
`mesh_workgroup_spirv.h` fixture:

| Module | Difference from source default | Expected center pixel |
| --- | --- | --- |
| `mesh_workgroup_probe.spv` | none | red |
| `mesh_workgroup_probe_one_lane.spv` | `ONE_LANE_PROJECTION=1` reads its own shared slot | blue clear |
| `mesh_workgroup_probe_wrong_prefix.spv` | `WRONG_PREFIX=1` replaces the active-lane prefix | blue clear |
| `mesh_workgroup_probe_texture_input.spv` | `TEXTURE_INPUT=1` requires a freshly uploaded sampled image in the mesh stage | red for a red texel; blue for a black texel |

The fragment module comes from `mesh_workgroup_probe.frag`. All four arrays are SPIR-V 1.6 targeting Vulkan 1.3. Generate a mesh variant with `glslangValidator -V --target-env vulkan1.3 -S mesh` and the optional `-DNAME=1` flag. Glslang emits `OpExecutionModeId %main LocalSizeId %uint_64 %uint_1 %uint_1`; replace exactly that one disassembled instruction with `OpExecutionMode %main LocalSize 64 1 1`, reassemble with `spirv-as --target-env vulkan1.3`, then run `spirv-opt --strip-debug --compact-ids` and `spirv-val --target-env vulkan1.3`. The literal mode avoids requiring the separately enabled `maintenance4` feature for this fixed-size fixture. An absent or multiply matched instruction is a generation error. The tracked header contains little-endian 32-bit words from the validated modules; temporary `.spv` files are ignored by Git.

The test executes the backend used by the app. It proves host mesh workgroups can exchange shared values, read an uploaded texture, and emit the expected primitive on this device. The fixture writes only layer zero into a one-layer image, so it does **not** test layer routing, recompile guest RDNA, test a 3D target, or establish Kena's output values. `PROSPER_NO_MESH_SHADER=1` exercises the unavailable-feature path even on capable hosts.
