# `agc` — Sony AGC / Gen5 shader descriptor formats

Reads the PS5-specific shader header and turns a stage's *declared* resources into a
`ShaderResourceTable`.

`agc_shader_layout` is the whole folder. It parses `AgcShaderUserData` — the sharp arrays, the
Extended User Data (EUD), the direct-resource offsets — and decodes individual descriptors —
`decode_buffer_descriptor` for a V#, with named decoders for the T# and BVH forms. There is no
`decode_sampler_descriptor`: an S# is decoded inline.

Two things to know before working here:

- **A declared resource is not the only kind.** Some titles describe resources through a
  shader-resource table the header only sizes (`srt_size_dw`), or load descriptors with SMEM from a
  user-data pointer. Those are recovered by a const-fold in `execute/`, not here — so "this function
  returned an empty table" does not mean the dispatch has no resources.
- **`user_data` can be non-null and point at unmapped guest memory.** `build_shader_resources` probes
  it with `guest_readable` before dereferencing, and documents the observed case. Anything new that
  reads it needs the same probe.

PS5-specific formats need direct title evidence; PS4-era layouts are a hypothesis, not a source.
