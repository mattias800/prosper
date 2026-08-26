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

These two directions do **not** carry the same weight, and the difference is structural.

The ISA walk is **not length-aware**. A correct walk needs per-instruction lengths (prosper's own
`rdna2_decode.cpp` has them, and records what mis-sizing costs: a phantom instruction that derails
the stream walk). Without them, a literal constant whose top six bits happen to read as the MIMG
encoding looks like an instruction.

**Every dword is tested, including the one after a match.** Skipping it looks free — a real MIMG is
at minimum two dwords, so its dword1 cannot start an instruction — but that is true of a *real*
MIMG and false of a *phantom*, and a phantom that swallows the next dword can hide a genuine
instruction. Review of #3039 demonstrated exactly that: an aliasing literal followed by a real
`image_sample dim:2D_ARRAY` against a non-arrayed image scanned CLEAN. `--json` reports
`phantom_risk` per shader (candidates sitting immediately after another candidate) so the suspicion
is visible; it never suppresses a finding.

So the error is **one-directional**: the finder can invent an MIMG that is not there, and cannot hide
one that is.

- **A finding is a candidate**, not a defect. Confirm it against the shader before acting on it.
- **A clean result is sound for the classes it checks** — a superset that found nothing means there
  was nothing to find.

Two further recall limits, both deliberate: `OpTypeImage` is parsed module-wide, so a shader that
declares *any* arrayed image clears *every* arrayed sample in it (without per-binding attribution,
claiming otherwise would report shaders that are actually correct); and `max_coord_arity` is
reported for information only — it feeds no decision, and reads 0 when a coordinate is not built by
`OpCompositeConstruct`.

## Self-test

`scan.py --self-test` is hermetic: it drives the scanner end to end against a stub `gpu_replay` and a
`spirv-dis` shim, so it needs no GPU and no Vulkan toolchain. It is registered with ctest as
`shader_conformance_scan_logic`. It covers both parsers, the classifier in both directions, and the
exit contract; ten mutations of the covered logic each turn it red, including `DIM_ARRAYED` dropping
the 2D_ARRAY case that is the tool's entire reason to exist.

The suite it replaced passed under all of those mutations. Its MIMG fixture was assembled from the
same constant it was checking, so mutating the constant mutated the fixture — a same-source positive
control, which tests the discriminator and never the domain.
