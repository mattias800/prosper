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

The saved-mask comparison and adjacent DPP reduction frontier is now implemented. The fragment CFG
dispatcher proves both ordinary B64 operands are live masks on every incoming path, reduces their
EQ/LG result over the exact guest wave and persists SCC into the later `s_cselect_b64`. Fragment
cases publish a nonzero static event identity; every event's ungated mask mismatch is voted in the
uniform common phase, so mismatch bits on lanes parked at another dispatcher PC still participate,
while the publishing event alone selects SCC. It also executes the exact in-place
`v_min_u32_dpp row_shr:{1,2,4,8}` family in that common phase. Value, source activity and static
event identity use the same subgroup source lane, and the shuffled source tag must match the
destination tag before UMin can update the persistent VGPR. The 83-resource scene shader that
previously stopped at pc 1,276 now recompiles completely to 288,728 SPIR-V dwords; a second copy of
both instruction families later in the same shader is covered as well.

The last current-recompiler deterministic replay, made before the event-isolation review hardening
above, remained byte-for-byte visually corrupted while every retained raw VS, FS and compute stage
substituted successfully. No current-recompiler Vulkan replay of the corrected lowering has been
run. There is no next unsupported retained shader instruction to quote from this capture; the next
useful GPU arm is to rerun operation 1,937 once and determine whether exact cross-PC semantics change
the target before returning to draw/resource/state localization.

## Ruled out

- **`IMAGE_GET_LOD` being unsupported is not, by itself, the cause of the corrupt red-triangle
  frame.** At the exact target immediately after operation 1,937/draw 1,846, a current-recompiler
  replay changed the BMP SHA-256 from `9b579dee...` to `cb33895b...`, but the red-pixel fraction was
  effectively unchanged (`0.486912` to `0.486932`) and human inspection showed the same corrupt
  scene. The hypothesis that implementing the first rejection alone restores this frame is false.
- **Merely admitting the saved-mask EQ/LG comparisons and adjacent DPP unsigned-minimum row
  reductions with the pre-review lowering was not sufficient.** That operation-1,937 replay kept
  backend hash `dce8e600954195f4`; its BMP was byte-for-byte identical to the prior
  post-`IMAGE_GET_LOD` image (SHA-256 `cb33895b...`, red-pixel fraction `0.486932`). Review then
  found that pair votes ran inside lane-divergent switch cases and that DPP neighbor mailboxes lacked
  static-event isolation. The old replay therefore does **not** rule out the corrected semantics
  changing the frame; do not quote it as a negative result for the hardened lowering.
  [#1907](https://github.com/mattias800/prosper/issues/1907)
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
spv_validate
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

The saved-mask tests use the exact House EQ/LG encodings in Wave32 and Wave64 arbitrary CFGs. They
trace the one required whole-wave vote through the dispatcher's Function-memory SCC and into the
later scalar select. High-half overwrite, numeric-pair and path-dependent-join controls prevent a
stale or non-mask pair from acquiring that lowering. Production mutations that removed EQ
admission, inverted its polarity or weakened the two-source MUST proof each broke the corresponding
named dataflow check. A second defect-shaped fixture places disjoint EQ and LG pairs at alternate
static dispatcher PCs. It proves both votes execute after the switch merge from ungated persistent
mask loads, tags 1 and 2 select the matching polarity, tag 0 remains reserved for no pending event,
and SCC still reaches `s_cselect_b64`. Gating the vote predicate by the current event makes the named
Wave32/Wave64 divergent-PC check red.

The DPP test uses the exact `v_min_u32_dpp v3,v3,v3 row_shr:{1,2,4,8}` packets in the same crossing
graphics CFG for both wave sizes. It proves the shared dynamic amount feeds both the row-bound check
and lane subtraction, pending plus EXEC gate source participation, and subgroup shuffle feeds UMin
and the persistent VGPR store. A four-way alternate-site fixture proves tags 1/2/3/4 are published,
tag 0 is reserved, value/activity/event use the same shuffle lane and source-event equality reaches
the final write in Wave32 and Wave64. Opcode, distinct-source and amount packet mutations fail the
exact contract; production mutations to opcode admission, row direction, source participation and
static-event equality each make the named House DPP check red.

The last authorized Vulkan replay completed in 2.803 seconds with return code zero and empty
pre/post process censuses. It substituted all 809 VS/FS pairs and 106 compute stages with zero stored
modules retained, read back the exact 1920x1080 FP16 target at operation 1,937, and produced the
byte-identical corrupt-scene result recorded above. It predates the common-phase/event-isolation
review fixes and must not be treated as visual evidence for the corrected head. This remains generic
shader-coverage progress, not a title-screen, gameplay-visual or performance improvement.
