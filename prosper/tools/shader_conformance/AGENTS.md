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
        --gpu-replay ./build-linux/gpu_replay --mimg-decoder ./build-linux/shader_inspect
python3 tools/shader_conformance/scan.py --self-test \
        --mimg-decoder ./build-linux/shader_inspect          # no capture, no GPU needed
```

Both binaries are required, and both default to `./build-linux/…`. `shader_inspect` is not an
optional accelerator: since #3184 it **is** the ISA walk (see below), so without it the scanner
cannot find a single image instruction — and it exits 2 rather than reporting a clean run.

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

**The ISA walk IS prosper's decoder (#3184).** `decode_mimg_sites` shells out to
`shader_inspect --mimg-sites`, which is `rdna2_walk` over `rdna2_decode_one`
(`src/gpu/recompiler/rdna2_decode.cpp`) and nothing else, and reads back one
`mimg-site pc=… op=… dim=…` line per image instruction. There is no RDNA2 decoding left in
`scan.py`: where an instruction boundary is, where DIM sits, and how the opcode's split MSB is
reassembled are stated in exactly one place in this repository. So the walk only ever tests a dword
for MIMG-ness when it is genuinely the first dword of an instruction, and it cannot fall behind the
recompiler when the decoder learns a new literal-forcing opcode or a wider NSA field.

Two earlier failure modes are therefore gone by construction rather than by two decoders happening
to agree. Before #3040 the walk tested *every* dword unconditionally, over-approximating in one
direction (never hide a real MIMG) but inventing MIMG instructions from operand and literal dwords
whose top bits happened to alias the encoding — a finding for a defect that did not exist. Review of
#3039 had found the opposite failure in that same walk: a phantom swallowing a real MIMG that
followed it, scanning CLEAN. Both came from one root cause, testing a dword that is not an
instruction boundary. `self_test` still pins both, and now pins them against the real decoder — a
hand-built literal aliasing an arrayed DIM that must NOT be found, and the #3039 swallow sequence
whose real MIMG must still be found, and found alone.

#3040 fixed those by **porting** `rdna2_decode_one`'s length dispatch into Python, and that port is
what #3184 removed. It was correct — see *Ruled out* below — and correctness was never the argument
against it: a copy of a decoding rule is the thing nobody updates when the original changes, and it
fails by silently returning a plausible instruction stream, which is precisely the failure this tool
exists to make impossible. The same folder had already drifted that way in the other direction, and
it surfaced while landing #3184: `shader_histo`'s hand-maintained `Rdna2Format` name table lost
track of the enum when `VOP3P` was added, mislabelling every VOP3P instruction as unknown and
reading out of bounds for `Unknown` itself (#3229).

What that leaves:

- **A finding is still a candidate**, not a defect — confirm it against the shader before acting on
  it. The reason is `classify()`'s module-wide match (below); it was never the ISA walk.
- **A clean result is sound for the classes it checks** — the walk can neither invent an instruction
  nor miss a real one the recompiler itself can decode. That last clause is now exact rather than
  approximate: it is the same walk, not an equivalent one.
- **A shader whose census could not be taken is not an examined shader.** The decoder prints a
  `mimg-sites-end` sentinel and the scanner refuses any output without one, so a decoder that dies
  before printing anything can never be read as "this shader has no image instruction". The sentinel
  carries the decoder's own site count and the scanner cross-checks it against the lines it parsed,
  which closes the same hole on the reading side: a site line the regex stops matching would
  otherwise be dropped silently, and an under-count reads as a cleaner shader. A raw dump that is
  empty, or not a whole number of dwords, is likewise refused rather than truncated — the
  in-process walk used to truncate it silently.

Two further recall limits, both deliberate and unrelated to the ISA walk: `OpTypeImage` is parsed
module-wide, so a shader that declares *any* arrayed image clears *every* arrayed sample in it
(without per-binding attribution, claiming otherwise would report shaders that are actually correct);
and `max_coord_arity` is reported for information only — it feeds no decision, and reads 0 when a
coordinate is not built by `OpCompositeConstruct`.

## Self-test

`scan.py --self-test` drives the scanner end to end against a stub `gpu_replay` and a `spirv-dis`
shim, so it needs no GPU and no Vulkan toolchain. It is registered with ctest as
`shader_conformance_scan_logic`. It covers both parsers, the classifier in both directions, the exit
contract, and the decode arms; ten mutations of the covered logic each turn it red, including
`DIM_ARRAYED` dropping the 2D_ARRAY case that is the tool's entire reason to exist.

It does need the **built `shader_inspect`**, which is why ctest passes
`--mimg-decoder $<TARGET_FILE:shader_inspect>`. That is a deliberate loss of hermeticity and it is
the point of #3184: the decode arms now exercise `rdna2_decode.cpp` itself, so reverting the real
decoder's MIMG NSA length to a flat 2 dwords reddens this suite. It could not before — the arms
tested the Python copy, and a suite that only tests the copy is exactly how a copy drifts unnoticed.
Without the binary the suite exits **2** (could not test); it never reports a pass it did not earn.

The suite it replaced passed under all of those mutations. Its MIMG fixture was assembled from the
same constant it was checking, so mutating the constant mutated the fixture — a same-source positive
control, which tests the discriminator and never the domain.

## Ruled out

One line per already-falsified hypothesis, the evidence that killed it, and the issue — so nobody
re-derives a dead answer at full cost.

- **"The Python length port and the real decoder have already drifted, so the scan has been
  reporting wrong MIMG sites"** — false, measured before the swap. Differential over the whole local
  dump library: every `.shader_text` embedded in all 55 `PPSA*-app0` eboots (30 of them carry any),
  **11,266 shaders carrying 199,521 MIMG sites**, decoded by the pre-#3184 `scan.py:decode_mimg_sites` and by
  `shader_inspect --mimg-sites`, with `(pc, opcode, dim)` compared elementwise: **0 disagreements,
  0 decoder failures**. So the swap is output-equivalent on real input, and no historical finding is
  invalidated by it. The argument for #3184 is the drift a copy invites, never a defect it had
  (#3184).
- **"Deleting the port is a recall risk, because the decoder may refuse streams the Python walk
  accepted"** — false for whole-dword streams, which is every real dump: the same 11,266-shader
  differential ran the decoder over all of them with zero refusals. It IS true for a zero-byte or
  partial-dword dump, and that difference is deliberate — those are now refused as unreadable rather
  than truncated into a clean-looking empty census (#3184).

**And one limit of that differential, so nobody reads more coverage into it than it has: the eboot
corpus contains no NSA MIMG instruction.** Mutating the real decoder to ignore the NSA extra-dword
count produced *zero* disagreements over all 11,266 shaders, which is only possible if every MIMG in
them has `NSA == 0`. Dropping two of the six VOP2 K-literal opcodes likewise produced zero. So the
equivalence result above covers the length rules this corpus exercises, and the `self_test` NSA arm
is hand-built precisely because the corpus cannot supply one. The instrument itself is not blind —
lengthening MIMG by one dword makes the same differential report **5,680 of 11,266 shaders
disagreeing across 29 titles** — which is the control that makes the zero a real null rather than a
same-source one (#3184; CLAUDE.md's instrument-trap 122).
