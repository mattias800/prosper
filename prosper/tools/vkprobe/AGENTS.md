# tools/vkprobe

The **control** for "is this prosper, or is this the driver?".

Every other instrument in this repository runs inside prosper: `gpu_replay` links the real backend,
the `PROSPER_*` diagnostics are compiled into it, and the Vulkan-execution tests use
`tests/fixtures/render_runner.h`. So when a draw stops producing pixels, none of them can separate a
defect in prosper's Vulkan usage from a defect in the shader, in ACO, or in the hardware — they all
share the suspect.

`vkprobe` is the arm that does not. It links **no prosper code**, creates its own instance, device
and queue, and executes the *same SPIR-V modules prosper produced* through a bare pipeline. A draw
that renders correctly here and wrongly through the backend rules out the recompiler, the SPIR-V,
ACO and the GPU in a single run, and leaves only what prosper does around the draw: descriptor
wiring, synchronisation, resource lifetime, pass configuration.

## What belongs here

Only self-contained Vulkan controls of this kind — programs whose value comes from *not* being
prosper. The moment one of them includes a prosper header it stops being a control and belongs in
`tests/` instead. Keep them dependency-free (Vulkan and the C++ standard library), and keep them
reading their inputs from files so a capture's real shaders and buffers can be fed in without
editing code.

## Boundary against its siblings

- `tools/gpu_replay/` replays a capture **through prosper's backend**. That is the subject, not the
  control.
- `tools/spv_validate/` asks whether a module is *valid*; `vkprobe` asks whether it *renders*.
- `tests/gpu/**` are assertions about prosper's behaviour and are compiled against it.

## History worth knowing

The first version of this program was written for #2937 and established that 1,500 indexed draws of
prosper's own modules render with zero failures on the same device where the backend rasterizes
nothing — which is what turned "RADV is broken" into "prosper is broken". It was then **deleted
during cleanup**, and #2945 had to rebuild it from the issue comment. That is why it lives in the
tree now: a control that has to be re-derived every time is a control nobody runs.

**And the rebuilt version then falsified that conclusion.** Run enough times, it reproduces the
class itself — twice so far, once with 76 of 200 indexed draws covering nothing. A control whose
negative result was believed after a few hundred iterations turned out to need tens of runs before
its first positive. Read the README's "Reading a result" section before quoting a clean run from
this tool; that is the whole reason it says what it says.
