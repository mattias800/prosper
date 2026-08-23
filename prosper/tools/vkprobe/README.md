# vkprobe

Drive a dumped SPIR-V vertex/fragment pair through a bare Vulkan pipeline, with **no prosper code in
the process**, and report whether the draw rasterizes.

```
vkprobe --vs vs.spv --fs fs.spv [--iterations 300] [--records FILE] [--indices 0,1,2]
        [--extent 64x64] [--device N] [--verbose]
```

Each iteration renders twice into two fresh targets — once with `vkCmdDraw`, once with
`vkCmdBindIndexBuffer` + `vkCmdDrawIndexed` — and counts pixels that differ from the blue clear.
The non-indexed arm draws `--indices`-many vertices starting at 0, so **the two arms are directly
comparable only when the indices are the identity sequence** (the default `0,1,2`); pass a capture's
real indices when you want the indexed arm to reproduce a specific draw, and read the arms
separately rather than comparing them. The two arms are the point: an indexed arm that fails beside
a passing non-indexed arm is #2937's signature; either arm failing alone is a device or shader
problem rather than an indexing one.

Exit status is `0` when every iteration of both arms covered pixels, `1` otherwise, and **`2` on a
setup error** — a probe that could not run must never read as a negative result.

## Getting the inputs out of a capture

```bash
# the modules a draw actually used
gpu_replay <capture.prgcap> --dump-shader <DRAW>:vs vs.spv out.bmp
gpu_replay <capture.prgcap> --dump-shader <DRAW>:fs fs.spv out.bmp
# the buffer its vertex fetch reads (binding number from --list-resources <DRAW>:vs)
gpu_replay <capture.prgcap> --dump-resource <DRAW>:vs:<BINDING> records.bin out.bmp
gpu_replay <capture.prgcap> --inspect-only | grep '^draw\[<DRAW>\] '   # indices are on this line

vkprobe --vs vs.spv --fs fs.spv --records records.bin --indices 0,4,5 --iterations 300
```

The set-0 descriptor layout is read out of the vertex module (`OpDecorate … DescriptorSet 0` +
`Binding`), so the tool adapts to whatever the dumped shader declares instead of encoding one
capture's layout. Every such binding is given a `STORAGE_BUFFER` descriptor over the same record
buffer, which is what prosper's recompiled vertex shaders want.

`--records` defaults to three 32-byte records forming a fullscreen triangle, in the layout
prosper's vertex-fetch shaders address. If the shader you dumped uses a different record stride, the
geometry will be arbitrary — which is still a valid *determinism* test, but not a correctness one.

## Reading a result

| observation | what it means |
| --- | --- |
| both arms cover the same pixel count on every iteration | **nothing, on its own.** This tool has produced thousands of such iterations and then failed; see below. It is a negative only across tens of RUNS, and even then it shows prosper's host-side Vulkan usage is not NECESSARY for a failure — never that it is exonerated |
| the indexed arm is empty while the non-indexed arm is not | the index path itself; compare against #2937 |
| coverage varies between iterations | the driver or hardware is not deterministic on this input — take it to the driver, and re-run at high `--iterations` before believing it |
| the indexed arm is empty on some iterations while the non-indexed arm is constant | the #2945 signature, reproduced with no prosper code in the process. Measured 3 of ~63 runs on a valid pipeline. The order of the two arms alternates every iteration and the by-position line is printed beside the by-arm line **so this attribution is checkable** — if the position counts are lopsided and the arm counts are not, it is submission order, not indexing |
| exit 2 | the probe never ran; nothing has been measured — a hung wait, an unsupported input pair, or a malformed argument all land here rather than reading as a failing draw |

The coverage predicate counts pixels that differ from the **blue clear**, ignoring alpha. A fragment
shader whose output happens to be `(0, 0, 1, x)` therefore reads as EMPTY — a false negative in the
alarming direction. Change the clear or the shader if that is your case.

The probe models **descriptor set 0, storage buffers only**, and reflects both modules. Any other
set, or a binding that is not a `StorageBuffer` variable, exits 2 with a message naming it rather
than building a pipeline layout inconsistent with the shaders — with no validation layers loaded,
nothing else would report that, and the coverage number would be undefined.

### Before any of that: the pipeline has to be VALID

The single most important thing this tool has established is about itself. Its first version created
its device **without `vertexPipelineStoresAndAtomics`**, and prosper's recompiled vertex shaders
fetch through `STORAGE_BUFFER` descriptors — which Vulkan requires to be `NonWritable` in the vertex
stage unless that feature is on (`VUID-RuntimeSpirv-NonWritable-06341`). So every pipeline it built
from a prosper vertex module was **invalid**, and because it loads no layers it printed coverage
numbers for them anyway. Every reading taken before that was fixed is void — the clean ones and the
failures alike, including the 1,500-draw result on #2937 that turned "RADV is broken" into "prosper
is broken".

Two guards now exist so it cannot happen again: the feature is enabled and its absence is `exit 2`,
and a vertex module whose output interface exceeds the device's `maxVertexOutputComponents` is
refused by name (this is #2945's own subject — a 132-against-128 interface — and a module dumped
before that bound was applied is refused with those exact numbers).

**Run it under the validation layers when the answer matters** — and note that the plain form below
enables CORE validation only, so "zero findings" from it says nothing about synchronization. Add a
`vk_layer_settings.txt` with `khronos_validation.validate_sync = true` if that is the question:

```bash
VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation vkprobe --vs vs.spv --fs fs.spv --iterations 200
```

**This probe has itself reproduced the #2945 class on a VALID pipeline, and that is its most
important result so far.** On the corrected build, running
`vkprobe --vs vs.spv --fs vs.spv.frag --iterations N` against #2937's module dump with default
records and default indices: 3 of 5 iterations empty in one run under the validation layers with
**zero core-validation findings**, 38 of 200 in another, 13 of 150 in a third — **3 failing runs of
about 63** — while the non-indexed arm beside them stayed constant at 496 throughout. So the run-level rate
is low, the control is **not** a clean negative, and "prosper's Vulkan usage is the defect" does not
follow from a quiet afternoon with this tool.

Practical consequence: **a clean run here proves nothing on its own.** Budget tens of runs before
reading a negative, and quote the number of RUNS as well as iterations — the failure is per-process
in shape, not per-iteration. Both failures happened to be the first execution after a relink; the
obvious mechanism, a cold Mesa shader cache, is falsified (`MESA_SHADER_CACHE_DISABLE=true`, 0 of
8x150 either way), so that correlation is unexplained at n=2.
