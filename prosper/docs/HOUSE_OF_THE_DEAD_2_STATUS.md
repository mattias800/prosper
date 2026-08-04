# The House of the Dead 2: Remake status

Tracking: [#1896](https://github.com/mattias800/prosper/issues/1896)

Visual defect: [#1907](https://github.com/mattias800/prosper/issues/1907)

Title ID: `PPSA24203`

Engine: Unity / IL2CPP

Current verified rung: **3 — gameplay with real GPU draws, with severe world-rendering defects**

## Current symptom

A fresh-save route reaches Training 1 gameplay at native 1920x1080. The HUD, weapon and ammunition
display, crosshair, objective text, sky, changing camera frames and hit effects render, but most of
the expected environment and actors are black, absent or reduced to sparse outlines. The published
title and gameplay images in `COMPATIBILITY.md` remain the current visual reference.

The captured gameplay frame contains 809 draws, 106 dispatches and 1,978 ordered operations. Its
historical realization record has 1,063 failures. Of 1,056 failed fragment stages, 1,006 first
reject at MIMG opcode `0x60`, independently disassembled as `IMAGE_GET_LOD`. AMD's RDNA2 ISA defines
its two results as sampler-clamped and raw implicit LOD; SPIR-V `OpImageQueryLod` exposes the same
ordered pair in fragment shaders.

The generic 2D fragment form is now lowered to a real `OpImageQueryLod`. The accepted contract is
deliberately narrow: ordinary non-NSA 2D texture and sampler resources, normalized coordinates,
only x/y result channels, and no auxiliary MIMG address, cache-policy, descriptor/result, D16 or
reserved controls. Other dimensions, stages and forms remain visible recompiler failures.

Three representative failed fragment programs now recompile through their original resource
tables: the 663-dword shader that supplied the exact instruction test produces 14,542 SPIR-V dwords
with 22 resources, and two larger scene shaders produce 10,621/11,379 dwords with 12/13 resources.
A complete current-recompiler replay substituted all 809 vertex and fragment stages and all 106
compute stages without retaining stored SPIR-V.

## Current frontier

The complete replay remains visually corrupted. The first large scene families now advance past
`IMAGE_GET_LOD` and stop at `s_cmp_eq_u64` comparing two saved wave-mask SGPR pairs. One exact live
sequence creates `s[44:45]` and `s[46:47]` with `s_and_b64` against EXEC, then compares the pairs.
The fragment CFG dispatcher handles mask-versus-zero comparisons, while this two-mask form remains
fail-visible. That is the next distinct recompiler frontier; it is not part of the `IMAGE_GET_LOD`
change.

## Ruled out

- **`IMAGE_GET_LOD` being unsupported is not, by itself, the cause of the corrupt red-triangle
  frame.** At the exact target immediately after operation 1,937/draw 1,846, a current-recompiler
  replay changed the BMP SHA-256 from `9b579dee...` to `cb33895b...`, but the red-pixel fraction was
  effectively unchanged (`0.486912` to `0.486932`) and human inspection showed the same corrupt
  scene. The hypothesis that implementing the first rejection alone restores this frame is false.
- **Draw 1,846 is not explained by non-finite or wholly clipped geometry.** Its 9,600 transformed
  vertices are finite; 4,872 are on-screen and 4,728 clipped. Of 3,200 triangles, 3,057 are
  non-degenerate and the screen-spanning triangles are present in the transformed output.
- **Draw 1,846's red overwrite is not caused by alpha blending.** Its fragment alpha is constant
  one, so the decoded normal alpha blend reduces algebraically to an overwrite.
- **The draw's primary texture is not using the old 4 KiB detiler.** The current `SW_4KB` detile
  produces a triangle-coherence mean of `1.08505e-05`; the prior mapping and plain Morton candidates
  produce `0.102495` and `0.133787`. The current mapping is the coherent one.

## Verification

Focused CPU suites pass in the `ps5ys` distrobox:

```text
recompile_coverage
rdna2_decode_walk
rdna2_spirv_struct
shader_resource_contract
```

The structural test uses the exact title-live packet `f1800108 01480809` in x, y and xy dmask forms.
It traces distinct U/V bit patterns into the real `OpImageQueryLod`, then traces clamped/raw query
components through compact consecutive VDATA writes, EXEC predication and the fragment export. An
isolated compute-stage arm requires the derivative-consuming operation to remain rejected outside
fragment shaders. Exact or raw negative packets separately cover NSA, UNRM, A16, DLC, GLC, SLC,
R128, TFE, LWE, D16 and all six reserved Table 100 bit positions in both production and table-less
coverage.

Defect-shaped mutations separately swapped U/V, swapped clamped/raw results, broke y-only VDATA
compaction, removed EXEC predication, admitted the operation in compute, disabled each shared
address/cache/result/reserved control family, and erased reserved-bit decode. Each mutation broke
its named check and the clean implementation passed again after restoration.

The single authorized Vulkan replay completed in 2.689 seconds with return code zero and empty
pre/post process censuses. It recompiled all retained stages, read back the exact 1920x1080 FP16
target at operation 1,937, and produced the unchanged corrupt-scene result recorded above. This is
generic shader-coverage progress, not a title-screen, gameplay-visual or performance improvement.
