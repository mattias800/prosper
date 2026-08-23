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
| both arms cover the same pixel count on every iteration | the modules, ACO and the device are fine; a defect seen through prosper is prosper's |
| the indexed arm is empty while the non-indexed arm is not | the index path itself; compare against #2937 |
| coverage varies between iterations | the driver or hardware is not deterministic on this input — take it to the driver, and re-run at high `--iterations` before believing it |
| exit 2 | the probe never ran; nothing has been measured — a hung wait, an unsupported input pair, or a malformed argument all land here rather than reading as a failing draw |

The coverage predicate counts pixels that differ from the **blue clear**, ignoring alpha. A fragment
shader whose output happens to be `(0, 0, 1, x)` therefore reads as EMPTY — a false negative in the
alarming direction. Change the clear or the shader if that is your case.

The probe models **descriptor set 0, storage buffers only**, and reflects both modules. Any other
set, or a binding that is not a `StorageBuffer` variable, exits 2 with a message naming it rather
than building a pipeline layout inconsistent with the shaders — with no validation layers loaded,
nothing else would report that, and the coverage number would be undefined.

**This probe has itself reproduced the #2945 class, twice, and that is its most important result
so far.** One 300-iteration run reported 80 arm disagreements with indexed coverage ranging
496-2731; one 200-iteration run had 76 of 200 indexed draws cover ZERO pixels while the non-indexed
arm beside it stayed at 496. Against roughly 7,500 clean iterations over ~45 runs. So the run-level
rate is low, the control is **not** a clean negative, and "prosper's Vulkan usage is the defect" does
not follow from a quiet afternoon with this tool.

Practical consequence: **a clean run here proves nothing on its own.** Budget tens of runs before
reading a negative, and quote the number of RUNS as well as iterations — the failure is per-process
in shape, not per-iteration. Both failures happened to be the first execution after a relink; the
obvious mechanism, a cold Mesa shader cache, is falsified (`MESA_SHADER_CACHE_DISABLE=true`, 0 of
8x150 either way), so that correlation is unexplained at n=2.
