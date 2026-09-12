# `capture` — frame capture and replay serialization

Serializes a live frame — commands, shaders, resources, and the renderer-owned targets it samples —
into a `.prgbundle` / `.prgcap` that `tools/gpu_replay` can reproduce offline and deterministically.

- `gpu_capture` — the capture lifecycle: begin, finish, materialize, and the pending-capture
  plumbing behind the F9 grab and the scheduled triggers.
- `capture_collect` — gathering a frame's contents: intervals, tables, submit items, raw shader
  versions, failure diagnostics.
- `serialize/` — the file format, in its own folder: byte cursors, per-field codecs, and the two
  entry points. See its `AGENTS.md`; the boundary against this folder is the version contract.
- `gpu_capture_internal` — the vocabulary those two share: magic/version/limits, footprint
  and blob-size helpers, and the validators. Internal to this folder; not a public header.
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


**Bundle serialization keeps only the owned chunk store in memory.** `write_gpu_capture_bundle`
streams the existing v1/v2 layout into a private temporary file and computes its trailing checksum
incrementally. Keep the historical checksum basis and exclude the trailer from its own hash. Preserve
chunk/resource integrity checks and the complete-file size bound; removing a duplicate allocation
does not authorize raising collection limits.

Install only after successful writes, flush and close. Failed validation or I/O must preserve an
existing destination and clean only this writer's private temporary. Concurrent writers may replace
one another with complete valid files; they must never share a truncatable temporary or delete the
previous destination to recover from a failed rename. The collector retains payload ownership until
the asynchronous writer completes, including failure and shutdown.
