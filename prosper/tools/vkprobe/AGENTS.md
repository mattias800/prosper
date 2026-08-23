# tools/vkprobe

The **control** for "is this prosper, or is this the driver?".

Every other instrument in this repository runs inside prosper: `gpu_replay` links the real backend,
the `PROSPER_*` diagnostics are compiled into it, and the Vulkan-execution tests use
`tests/fixtures/render_runner.h`. So when a draw stops producing pixels, none of them can separate a
defect in prosper's Vulkan usage from a defect in the shader, in ACO, or in the hardware — they all
share the suspect.

`vkprobe` is the arm that does not. It links **no prosper code**, creates its own instance, device
and queue, and executes the *same SPIR-V modules prosper produced* through a bare pipeline. What
that separates is narrower than it first appears, and getting it wrong has cost this project twice:
it *avoids* prosper's **host-side Vulkan usage** — descriptor wiring, synchronisation, resource
lifetime, pass configuration — rather than reproducing it. So a failure here shows that usage is not
NECESSARY for the defect; it never shows it is exonerated, and the difference matters because the
same run also suggests something on prosper's side is amplifying. It does **not** clear the
recompiler, because the SPIR-V under test is the recompiler's own output, and it does **not** clear
the driver, because on a valid pipeline it reproduces #2945 itself.

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

**And the rebuilt version then falsified that conclusion twice over.**

First on rate: run enough times, it reproduces the class itself. A control whose negative was
believed after a few hundred iterations needed tens of runs before its first positive.

Then on validity, which is the one worth remembering. The rebuild — like, almost certainly, the
original — created its device without `vertexPipelineStoresAndAtomics`, so **every pipeline it built
from a prosper vertex module was invalid** (`VUID-RuntimeSpirv-NonWritable-06341`), and with no
layers loaded it reported coverage for them anyway. Every reading **this program** took before that
fix is void, in both directions. #2937's 1,500-draw result came from a different, earlier program
that was deleted before this one existed, so whether it shared the gap is inference — likely, but
not checkable.

The lesson the folder exists to carry: **a control is only a control once it is valid, and a program
that loads no validation layers cannot tell you that it is.** Read the README's "Reading a result"
before quoting anything from this tool.
