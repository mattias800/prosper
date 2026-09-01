# AGENTS.md — prosper/tools/shader_conformance

One question, asked of any capture, with no per-title knowledge:

> **does the recompiled shader still address the image the way the guest asked?**

A GCN/RDNA image instruction carries its own `DIM` field, so the guest states the addressing it wants
in the machine code. When the emitted SPIR-V declares a different image shape, coordinates are
silently dropped — and the result is *plausible but wrong output*, never an error. That is the
expensive failure mode: nothing fails, nothing logs, and the picture just looks a bit off.

## Why this folder exists

It was written after #325 cost a full session of hand tracing. Tomb Raider I-III Remastered textures
its entire world from one 256-slice `Texture2DArray`; prosper lowered the `2D_ARRAY` sample to `2D`,
dropping the layer, so every surface sampled slice 0 and the world rendered flat. Finding that by eye
took hours of overrides, shader dumps and hypotheses — three of which were wrong. **The scanner finds
the same defect in one command, and names the issue.**

The point is not this one bug. It is that the check needs no knowledge of the title, so it turns
"deep-dive each game" into "capture a frame, scan it, read the list".

## The workflow it belongs to

Both halves are already cheap and headless, which is what makes this scale:

1. **Capture** — the F9 frame grab is schedulable: `PROSPER_GRAB_BUNDLE_AFTER_MS=<ms>` on
   `prosper-app` writes a `.prgbundle` with no human at the keyboard (see `tools/AGENTS.md`).
2. **Scan** — point this at the bundle. It extracts the last submit itself.

```bash
python3 tools/shader_conformance/scan.py <capture.prgbundle|.prgcap> \
        --gpu-replay ./build-linux/gpu_replay
python3 tools/shader_conformance/scan.py --self-test     # no capture, no GPU needed
```

Exit codes are the interface: **0** scanned and clean, **1** mismatches found, **2** could not scan.
`--json` for machine consumption.

## What it will and will not tell you

It reports a *structural* mismatch between the guest instruction and the emitted module. It does
**not** claim the picture is wrong, and it cannot: a title may never sample a non-zero slice, in
which case a dropped layer costs nothing — that is exactly why #325 sat open harmlessly on The
Messenger, whose array happened to be uniformly white. Treat a finding as **a place worth looking**,
and confirm impact against the frame.

It covers vertex, fragment **and compute** stages. Compute matters: array sampling worked *only* in
compute for a long time, so a graphics-only scanner would have reported the healthy half.

## It refuses to report a clean result it did not earn

This is the design point worth preserving — and the first version of this tool **did not actually
hold it**, which is why the claim is now spelled out with its limits attached.

It exits **2**, never printing a clean result, when `spirv-dis` is missing, `gpu_replay` is not where
it was told, a capture holds nothing with shaders, a capture could not be inspected, or **any single
capture yielded zero examined shaders**. That last clause is the one that was wrong: the guard was
global, so one readable capture certified an entire run, and a barren capture alongside it exited 0.
Findings could also be attributed to a capture whose dumps had all failed, because one temp directory
was shared across captures and success was judged by whether a file existed rather than by the
child's exit code. Both are fixed, and each half is pinned **independently**: a stub capture that
writes a file and exits non-zero is caught only by the return-code check, and one that exits 0
having written nothing is caught only by the per-capture temp directory.

Every run prints, per capture, how many shaders it examined and how many it could not read. **Quote
those counts alongside any clean result** — an exit code alone is not falsifiable.

### What a finding is, and what a clean result is

**The ISA walk is length-aware (#3040).** `decode_mimg_sites` decodes each instruction's real
per-instruction length — mirroring the format dispatch and length rules in prosper's own
`rdna2_decode.cpp` (`rdna2_decode_one` / `rdna2_walk`) — and only ever tests a dword for the MIMG
top-6-bit encoding when it is genuinely the first dword of an instruction. An operand or trailing
literal dword can no longer be misread as an instruction start, and a real MIMG's own address/NSA
dwords can no longer be misread as separate instructions either.

Before #3040 the walk tested *every* dword unconditionally, which was deliberately over-approximating
in one direction (never hide a real MIMG) but paid for that by inventing MIMG instructions from
operand and literal dwords whose top bits happened to alias the encoding — a finding for a defect
that did not exist. Review of #3039 had already found the opposite failure mode in the pre-#3039
walk (a phantom swallowing a real MIMG that followed it, scanning CLEAN); the length-aware walk fixes
both directions at once, because both come from the same root cause — testing a dword that is not
actually an instruction boundary. `self_test` pins both: a hand-built literal aliasing an arrayed DIM
that must NOT be found, and the #3039 swallow sequence whose real MIMG must still be found (and, now,
found *alone*).

Residual risk is different in kind, not degree: this Python walk is a **second, independent
implementation** of `rdna2_decode_one`'s length dispatch rather than a call into it, so a future
change to the real decoder's length rules needs a matching update here or the two can drift apart
silently. #3040's own suggested fix — teaching `shader_histo` to emit real per-instruction `(pc,
opcode, dim)` and having `scan.py` consume that instead — removes the duplication entirely and
remains the more robust direction; it just was not required to fix the length-awareness defect
itself. Until it lands:

- **A finding is still a candidate**, not a defect — confirm it against the shader before acting on
  it. The remaining reason is `classify()`'s module-wide match (below), not the ISA walk.
- **A clean result is sound for the classes it checks** — the walk can neither invent an instruction
  nor miss a real one it can decode.

Two further recall limits, both deliberate and unrelated to the ISA walk: `OpTypeImage` is parsed
module-wide, so a shader that declares *any* arrayed image clears *every* arrayed sample in it
(without per-binding attribution, claiming otherwise would report shaders that are actually correct);
and `max_coord_arity` is reported for information only — it feeds no decision, and reads 0 when a
coordinate is not built by `OpCompositeConstruct`.

## Self-test

`scan.py --self-test` is hermetic: it drives the scanner end to end against a stub `gpu_replay` and a
`spirv-dis` shim, so it needs no GPU and no Vulkan toolchain. It is registered with ctest as
`shader_conformance_scan_logic`. It covers both parsers, the classifier in both directions, and the
exit contract; ten mutations of the covered logic each turn it red, including `DIM_ARRAYED` dropping
the 2D_ARRAY case that is the tool's entire reason to exist.

The suite it replaced passed under all of those mutations. Its MIMG fixture was assembled from the
same constant it was checking, so mutating the constant mutated the fixture — a same-source positive
control, which tests the discriminator and never the domain.
