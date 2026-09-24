# Failed-draw NGG input probe

`ngg_capture_probe` executes a **compile-only** workgroup/export SPIR-V module against the small buffer inputs retained for one failed draw. It does not rasterize, infer the guest merged-stage ABI, or verify a presented frame. It refuses a capture without the exact buffer blobs; the normal metadata-only failure record is insufficient.

Capture with `PROSPER_CAPTURE_FAILED_INPUTS_PROGRAM=0xPROGRAM`. The ordered executor samples at the failed operation. Full-submit capture reconstructs draws in the capture callback and logs `capture-time`; those bytes are not proven to be the earlier draw-time contents. Each stage snapshot is bounded to 4 KiB per buffer and 64 KiB in total, and an incomplete set is declined rather than partly serialized.

Compile the selected failed vertex chain using `gpu_replay --retry-failed-chain N --probe-ngg-workgroup-s3 VALUE --retry-failed-chain-spv module.spv capture.prgcap`. `VALUE` is deliberately explicit: #2072 records an unresolved merged-wave-info ABI disagreement. Validate with `spirv-val --target-env vulkan1.3 module.spv`, then run:

```text
ngg_capture_probe capture.prgcap N module.spv outputs.bin
ngg_capture_probe capture.prgcap N module.spv zeroed.bin --zero-inputs
ngg_capture_probe capture.prgcap N module.spv one-muted.bin --zero-binding=4
```

The output contains 13 little-endian `uint32_t` words per flattened input lane: PRIM, POS0.xyzw, POS1.xyzw, PARAM0.xyzw. The printed nonzero PRIM count is a raw slot census, not a valid rendered primitive count. A changed output under a muted binding proves that the compile-only module reads that input; an unchanged output cannot prove the input is unused in the actual guest draw. The tool currently accepts the five small bindings used by the selected producer (2..6) and refuses other layouts.
