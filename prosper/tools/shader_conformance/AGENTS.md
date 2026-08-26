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

This is the design point worth preserving. If `spirv-dis` is missing, `gpu_replay` is not where it
was told, no shader could be dumped, or a capture holds nothing with shaders, it exits **2** and says
so — it never prints `mismatches: 0`. A silent scanner and a healthy codebase produce identical
output, and this project has lost real time to exactly that. Every run prints the number of shaders
it actually examined; quote that count alongside any clean result.

That guard has already earned itself: the first cross-title run hit a bundle it could not read and
refused to clear the title.
