# `capture` — frame capture and replay serialization

Serializes a live frame — commands, shaders, resources, and the renderer-owned targets it samples —
into a `.prgbundle` / `.prgcap` that `tools/gpu_replay` can reproduce offline and deterministically.

- `gpu_capture` — the capture itself.
- `gpu_capture_bundle` — bundle format and manifest.
- `capture_compute_policy` — the policy governing compute capture.
- `writer_provenance` — which pass wrote a given range, so a replay can attribute a pixel.

This is the highest-leverage debugging path in the project: press **F9** in `prosper-app` (or use the
`PROSPER_GRAB_BUNDLE_AFTER_MS` / `_AT_FRAME` triggers for a headless run), then iterate on the frozen
frame instead of re-routing to catch the moment live.

**A capture owns ALLOCATIONS, not descriptor ranges.** A resource's captured range is anchored at
its `gpu_addr`, but the range may legitimately start below it — a tiled GFX10 mip chain stores level
zero last, so the rest of the chain lies underneath the address the descriptor names. The serialized
`blob_offset` is what carries that: it is the count of owned bytes preceding `gpu_addr`, and replay
publishes it as `ShaderResource::host_data_prefix_bytes` (#3202). Anything that repoints a replayed
resource's `host_data` at a different buffer must reset that field with it.

**A manifest strips payloads by design.** Validators and consumers must not assume a referenced
payload is present; several were fixed for dereferencing exactly what the manifest omits.
