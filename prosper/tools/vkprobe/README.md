# vkprobe

Drive a dumped SPIR-V vertex/fragment pair through a bare Vulkan pipeline, with **no prosper code in
the process**, and report whether the draw rasterizes.

```
vkprobe --vs vs.spv --fs fs.spv [--iterations 300] [--records FILE] [--indices 0,1,2]
        [--extent 64x64] [--device N] [--verbose]
```

Each iteration renders the same triangle twice into two fresh targets — once with `vkCmdDraw`, once
with `vkCmdBindIndexBuffer` + `vkCmdDrawIndexed` over indices naming the same vertices — and counts
pixels that differ from the blue clear. The two arms are the point: an indexed arm that fails beside
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
| exit 2 | the probe never ran; nothing has been measured |

Re-run before believing an unstable result. During #2945 one 300-iteration run reported 80
disagreements between the arms and was not reproduced by 3×200 and 1×3000 iterations immediately
afterwards on the same binary — the same slow, machine-wide drift that made every A/B in that
investigation worthless unless the arms were interleaved.
