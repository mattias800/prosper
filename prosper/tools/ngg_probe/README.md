# Failed-draw NGG input probe

`ngg_capture_probe` executes a **compile-only** workgroup/export SPIR-V module against the small buffer inputs retained for one failed draw. It does not rasterize, infer the guest merged-stage ABI, or verify a presented frame. It refuses a capture without the exact buffer blobs; the normal metadata-only failure record is insufficient.

Capture with `PROSPER_CAPTURE_FAILED_INPUTS_PROGRAM=0xPROGRAM`. The ordered executor calls the snapshot helper while handling the failed operation. Full-submit capture reconstructs draws in the capture callback and logs `capture-time`; those bytes are not proven to be the earlier draw-time contents. The snapshot helper has direct tests; an integration test has not yet checked the exact timing of either call site. Each stage snapshot is bounded to 4 KiB per buffer and 64 KiB in total, and an incomplete set is declined rather than partly serialized.

Compile the selected failed vertex chain using `gpu_replay --retry-failed-chain N --probe-ngg-workgroup-s3 VALUE --retry-failed-chain-spv module.spv capture.prgcap`. `VALUE` is deliberately explicit: #2072 records an unresolved merged-wave-info ABI disagreement. Validate with `spirv-val --target-env vulkan1.3 module.spv`, then run:

```text
ngg_capture_probe capture.prgcap N module.spv outputs.bin
ngg_capture_probe capture.prgcap N module.spv zeroed.bin --zero-inputs
ngg_capture_probe capture.prgcap N module.spv one-muted.bin --zero-binding=4
```

To test a proposed merged-GS launch layout, add `--probe-ngg-packed-offsets` to the `gpu_replay` compile command. Supply `ngg_capture_probe` with `--packed-offsets=FILE`, where `FILE` contains exactly two little-endian `uint32_t` words per flattened lane, interleaved as initial VGPR0 then VGPR1. The compiler still derives the synthetic ES vertex/instance indices from the captured draw count; the file changes only the initial GS offset words. The option is compile-only and intentionally does not choose a layout or admit a live draw. A proposed layout needs a discriminating wrong-layout control and evidence for the guest ES/GS wave partition before its results can be interpreted as guest output. Pair the module and input file explicitly: the probe does not embed a layout signature in SPIR-V.

`gpu_replay --retry-failed-chain N --probe-ngg-full-four-wave --retry-failed-chain-spv module.spv capture.prgcap` can also **compile** a 256-invocation module that expects ten raw input words per lane: initial v0..v8 followed by wave-uniform s3. This models four guest waves sharing one LDS workgroup; it does not infer any register values or prove the guest's partition. `ngg_capture_probe` deliberately refuses modules whose declared local size is not 64x1x1. There is no captured-input execution command for the four-wave module yet: the real shader's workgroup-barrier arrival must be proved before running a candidate that disables ES/GS lanes by wave. The synthetic four-wave LDS test exercises the shell without this hazard.

The output contains 13 little-endian `uint32_t` words per flattened input lane: PRIM, POS0.xyzw, POS1.xyzw, PARAM0.xyzw. The printed nonzero PRIM count is a raw slot census, not a valid rendered primitive count. A changed output under a muted binding proves that the compile-only module reads that input; an unchanged output cannot prove the input is unused in the actual guest draw. The tool currently accepts the five small, dword-aligned bindings used by the selected producer: ConstantBuffer at 2/5/6, VertexBuffer at 3/4. It refuses other layouts or resource classes.
