`volume_target_probe.mesh` and `.frag` produce the four arrays in
`volume_target_spirv.h`. They test a layered mesh producer and a direct 3D sampled
consumer in the production Vulkan backend.

Generate with glslangValidator targeting Vulkan 1.3. Compile the fragment source
three times: no definitions, `-DGREEN=1`, and `-DSAMPLE_VOLUME=1`. In the mesh
disassembly replace exactly one
`OpExecutionModeId %main LocalSizeId %uint_32 %uint_1 %uint_1` with
`OpExecutionMode %main LocalSize 32 1 1`, then reassemble with `spirv-as`.
The literal mode avoids a separate maintenance4 feature requirement. Run every
module through `spirv-opt --strip-debug --compact-ids` and
`spirv-val --target-env vulkan1.3`. Encode the resulting little-endian 32-bit
words in the header. A missing or repeated execution-mode instruction is a
generation error.
