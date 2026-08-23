# vkprobe

Drive a dumped SPIR-V vertex/fragment pair through a bare Vulkan pipeline, with **no prosper code in
the process**, and report whether the draw rasterizes.

```
vkprobe --vs vs.spv --fs fs.spv [--iterations 300] [--records FILE] [--indices 0,1,2]
        [--extent 64x64] [--device N] [--verbose] [--device-local-indices]
        [--vs-b vs2.spv --fs-b fs2.spv] [--readback-dwords FIRST:COUNT] [--index-bits 16|32]
```

**`--vs-b`/`--fs-b` render a SECOND module pair in the same process, interleaved render-by-render
with the first, and report the two separately.** Use it for any comparison between two modules. The
failure rate here drifts machine-wide over minutes — the identical command measured 0/20 in one
ten-minute window and 12/12 in the next with nothing changed — so "run module A for a while, then
module B" is void rather than negative, and two pipelines alive in one process is the only shape of
the comparison that survives that. Both pipelines share one descriptor set layout, the union of what
all the supplied modules declare, which is what makes it a comparison of the modules; it also means
a paired run is not bit-identical in setup to either module run alone, so run both ways when that
could matter.

**`--readback-dwords FIRST:COUNT` turns an empty draw into three distinguishable answers.** Coverage
alone cannot say whether an empty indexed draw means the vertex shader never ran, ran with all-zero
indices, or ran correctly and lost the primitive afterwards — a degenerate triangle looks the same
in all three. Those dwords of the record buffer are reset to `0xdeadbeef` before every render and
read back after it, so a vertex shader that stores what it saw (`shaders/index_readback_vs.spvasm`)
reports out of band. The tool prints the patterns per arm with `--` for a dword nothing wrote. A
module that does not write leaves `[--,--,--]`, which is the readback path's own positive control:
if it ever printed something else for a non-writing module it would be reporting stale bytes.

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
| the indexed arm is empty while the non-indexed arm is not | the index path — but read the row below before attributing it to *indexing*: the arms also differ in where the index memory lives (`--device-local-indices`) |
| coverage varies between iterations | the driver or hardware is not deterministic on this input — but check the by-position line first, and re-run across many RUNS, not many iterations, before believing it |
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
numbers for them anyway. Every reading **this tool** took before that fix is void, the clean ones
and the failures alike. #2937's 1,500-draw result came from a *different, earlier* program that was
deleted before this one was written, so whether it shared the gap is **inference** — likely, since a
hand-written control has no reason to enable that feature, but not checkable.

Two guards now exist so it cannot happen again: the feature is enabled and its absence is `exit 2`,
and a vertex module whose output interface exceeds the device's `maxVertexOutputComponents` is
refused by name (this is #2945's own subject — a 132-against-128 interface — and a module dumped
before that bound was applied is refused with those exact numbers).

`--device-local-indices` stages the index data into `DEVICE_LOCAL` memory through a transfer instead
of binding a host-coherent allocation directly, which is what a real application does. It exists
because the two arms differ in more than indexedness — only the indexed one binds an index buffer at
all — so a failure attributed to "indexed draws" could be an attribution to host-coherent index
memory. **Result: the DEVICE_LOCAL arm fails too** — 2 of 10 iterations in a failing window, with
the selected memory type's flags printed as `0x1`, so host-coherent index memory is not necessary
for the defect. A 20-run-per-arm interleaved A/B did not separate the rates (1 failing run against
0, which is underpowered at this base rate). The probe prints the memory type it actually selected
and refuses if the device has no DEVICE_LOCAL type that is not also HOST_VISIBLE — without that,
this arm silently measures nothing on an APU while printing a confident label.

**Run it under the validation layers when the answer matters** — and note that the plain form below
enables CORE validation only, so "zero findings" from it says nothing about synchronization. Add a
`vk_layer_settings.txt` with `khronos_validation.validate_sync = true` if that is the question:

```bash
VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation vkprobe --vs vs.spv --fs fs.spv --iterations 200
```

## The verdict it reached, and the arm that made it possible (2026-08-23)

Everything below this heading was written before the question was settled; keep it, because the ways
this tool has been wrong are the reason to trust the answer. The answer itself:

**#2945 is the driver or the hardware, not prosper's generated SPIR-V.** The module in
`shaders/minimal_ssbo_vs.spvasm` performs the same storage-buffer loads through the same descriptor
shape as prosper's dumped vertex module and was hand-written from the specification; run beside that
module **in the same process** with `--vs-b`, it fails at the same rate. `shaders/no_ssbo_vs.spvasm`,
which never reads the storage buffer at all, fails too — so the failure does not even require a
storage-buffer load. All of them render correctly, on both arms, on **lavapipe**, from the same
binary and the same files. Every module involved is `spirv-val` clean.

Two things that follow for anyone using this tool:

- **Reach for `--vs-b` with a `shaders/` module first.** Running a dumped module alone can never
  separate "the driver is wrong" from "prosper emitted bad SPIR-V", because both are in every run.
  A paired run separates them in one command.
- **Concurrent GPU work from another process is what opens the failing window.** The "drift over
  minutes" this README and `GRAPHICS.md` describe is not weather: three heavy `gpu_replay` full-submit
  replays flipped a quiet machine into failing within one round and it recovered within ~15 s of
  them stopping, and a second `vkprobe` used purely as a load reproduces it more weakly. So run the
  control against a *quiet* GPU when you want a negative, and put a load beside it when you want the
  defect. `pgrep -x 'prosper-app|screenshot|boot_trace'` will not see another agent's `gpu_replay`.

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

**And the practical form of that warning, measured 2026-08-23: GPU load from another process
INDUCES the failing regime on demand, so put load beside the measurement before believing a null.**
Three heavy `gpu_replay` full-submit replays flipped a passing box into failing within one round of
a 3-second detector loop, and it recovered ~15 s after they stopped; a second `vkprobe` used purely
as load reproduces it more weakly. **The converse does not hold** — an independent reviewer measured
13/20 and 8/20 failing with no other GPU process running — so load is *sufficient*, not *necessary*,
and a quiet-box null is undecided rather than clean. Note also that
`pgrep -x 'prosper-app|screenshot|boot_trace'` does not see another agent's `gpu_replay`, `vkprobe`
or `ctest`, which is precisely the load that matters.

### The rate table, so nobody re-derives it

All 2026-08-23, Linux / AMD Radeon 8060S (RADV STRIX_HALO `0x1586`), Mesa 26.1.6 host, kernel 7.1.5.
Paired runs are one process, interleaved render-by-render, 20 iterations per run.

| arm | condition | failing RUNS | empty indexed iterations |
| --- | --- | --- | --- |
| prosper's dumped VS | sustained GPU load | 52 of 400 | 220 of 8,000 |
| `minimal_ssbo_vs` (hand-written) | sustained GPU load, paired with the above | 54 of 400 | 228 of 8,000 |
| `no_ssbo_vs` (hand-written) | sustained GPU load | 142 of 400 | 675 of 8,000 |
| `index_readback_vs` (hand-written) | sustained GPU load, paired with the above | 138 of 400 | 669 of 8,000 |
| all four, NON-indexed arm | sustained GPU load | — | **0 of 32,000** |
| all four | idle GPU (see the caveat below) | 0 of 48 each | 0 |
| prosper's VS, `minimal_ssbo_vs`, `no_ssbo_vs` | **lavapipe**, during a window RADV was failing 20/20 | 0 of 1 each | 0 of 20 each |

Two limits on that table, both of which matter more than the totals do. **The paired rows are the
evidence; the cross-campaign comparison is not** — 142-vs-52 is two different processes with a
different option set in different windows, and nothing makes them comparable. Within a campaign the
sharpest statement is not the two totals but their run-by-run agreement: **prosper's module and the
hand-written one disagreed on 2 of 400 runs.** And **each campaign is a single ~3-minute window**
(48 rounds over 206 s; 400 rounds over 173 s), so "0 of 48" is one drift period rather than 48
independent samples — read it as *undecided*, not as a clean arm. An independent reviewer measured
13/20 and 8/20 failing on a box with no other GPU process running, so load is sufficient to induce
the regime, not necessary for it.

**Use NON-IDENTITY indices for anything diagnostic, and treat every rate above as a lower bound.**
The campaign that produced that table ran `--indices 0,1,2`, where "the index was fetched correctly"
and "the shader got the ordinal and the index buffer was never read" give the *same* readback — so
it scored 91.6% correct. Re-run with `--indices 3,4,5`, where the only correct answer is
`[--,--,--,4,5,6,--,--]`:

| readback, 1,400 indexed iterations under load, `--indices 3,4,5` | count | |
| --- | --- | --- |
| `[--,--,--,4,5,6,--,--]` | 17 (1.2%) | correct |
| `[--,--,--,--,--,--,--,--]` | 1,076 (76.9%) | nothing written in the watched window |
| `[1,2,3,--,--,--,--,--]` | 288 (20.6%) | the shader saw the ordinals, not the uploaded indices |
| other | 19 | |

**What the readback does not tell you.** A shader handed an out-of-range index writes outside the
descriptor's range and the write is dropped, which is indistinguishable from never running; an index
lost as `0` and the same index arriving as `7` both land outside the watched window. And `[1,2,3]`
on an indexed arm is equally consistent with the shader having been handed the ordinals and with the
host's sentinel reset never becoming visible, leaving the previous render's bytes to be read back.
So do not turn these into a mechanism — what they establish is that **the vertex indices reaching
the shader are usually not the ones the host uploaded**, while lavapipe returns the correct distinct
answer for every index set from the same binary and files.

The non-indexed arm read `[1,2,3]` on 1,400 of 1,400 — which is **undiagnostic by construction**,
since for that arm the correct answer and a stale read of the previous render are the same bytes.
