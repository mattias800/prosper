# `capture` — frame capture and replay serialization

Serializes a live frame — commands, shaders, resources, and the renderer-owned targets it samples —
into a `.prgbundle` / `.prgcap` that `tools/gpu_replay` can reproduce offline and deterministically.

- `gpu_capture` — the capture itself.
- `gpu_capture_bundle` — bundle format and manifest.
- `capture_compute_policy` — what compute state a capture retains.
- `writer_provenance` — which pass wrote a given range, so a replay can attribute a pixel.

This is the highest-leverage debugging path in the project: press **F9** in `prosper-app` (or use the
`PROSPER_GRAB_BUNDLE_AFTER_MS` / `_AT_FRAME` triggers for a headless run), then iterate on the frozen
frame instead of re-routing to catch the moment live.

**A manifest strips payloads by design.** Validators and consumers must not assume a referenced
payload is present; several were fixed for dereferencing exactly what the manifest omits.
