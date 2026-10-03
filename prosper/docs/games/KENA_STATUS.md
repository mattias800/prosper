# Kena: Bridge of Spirits (`PPSA01802`) — status

Unreal Engine 4 (Ember Lab), one 28.5 GB `kena-ps5.pak` (no IoStore), Wwise, SDK `0x03000000`. Tracker
[#3787](https://github.com/mattias800/prosper/issues/3787). Brought up on Windows 11 / RTX 4090;
Linux/AMD title-menu investigations are recorded below.

## Current state — rung 2, gameplay reached with the world absent (2026-09-23)

On `main` (which has both crash fixes, [#3817](https://github.com/mattias800/prosper/pull/3817) and
[#3814](https://github.com/mattias800/prosper/pull/3814)), `screenshot.exe` with the route
`prosper/scripts/kena/reach-first-gameplay-prompt.pad` reaches, in order:

| t (one 540 s run) | what renders |
| --- | --- |
| 0-70 s | black (the Ember Lab logo movie is decoded and delivered to the guest but never visible — open) |
| ~80 s | **main menu** — `New Game / Load Game / Options`, `Version 1.16`, over black |
| ~160 s | **Choose Difficulty** |
| ~170-220 s | **brightness calibration** (the map art and slider render) |
| ~240 s | loading spinner — the first level loads |
| ~270-330 s | **intro narration** (*"Unique wooden masks are carved to honor the dead…"*), legible, with the blue *Spirit Guides* highlight |
| ~410-540 s | the first gameplay prompt, *"Press … to Pulse"* — **over a black world** |

The guest is in the game loop and asking for gameplay input, but the scene behind the prompt does not draw
(6,817 non-black pixels of 8.3 M per sample, all of them the prompt). Under the ladder that is **rung 2**, not rung 3.

**Likely cause of the black world, not yet established per draw:** the title's fragment shaders need wave64.
The run logs `[render] skip draw=... fragment shader requires subgroup size 64 (device range 32..32 ...)` for 402
distinct shaders (400 `why=0x2 wave-any`, two `lane-id`), and an NVIDIA device is 32-wide, so every one of those
draws is skipped — the cross-title gap [#2147](https://github.com/mattias800/prosper/issues/2147). The main menu's
missing animated background is probably the same thing. Confirming it needs either a 64-wide (AMD) run or a
census of which draws in a gameplay frame are skipped; neither has been done.

Every menu frame also alternates with a fully black composited frame in about half the samples; not investigated.

### Linux/AMD title-menu boundary (2026-09-24)

The live menu still shows its wordmark and UI over black. A retained 1600×900 RGBA16F target on the normal
render path contains nonzero scene data (one observed target has every RGB channel at half-float maximum), while
the later 3200×1800 compositor output is black. Other same-extent retained images contain recognizable forest
content. The saturated target is not by itself evidence of the correct scene: raw half-float values and converted
BMP previews have different meanings. The ordered submit capsule is explicitly **unverified** because it was not
matched to its own presented frame; its draw/dispatch lineage is useful for localization, not a visual oracle.

In that capsule, the final compositor samples a zero 32³ LUT at PS binding 39. The immediately preceding LUT
producer is unrealized because its merged-NGG vertex chain hits the existing wave/mesh gate
[#3135](https://github.com/mattias800/prosper/issues/3135). A corrected **one-draw-only** live bracket
confirms the boundary on current main: the final compositor's retained scene and half-resolution inputs have
visible content, its bound LUT has 0/131,072 nonzero decoded bytes, and the executed draw turns its 3200×1800
target black. Replacing only that draw's LUT sample with a constant or its final output with its existing scene
sample makes the menu endpoint nonblack. Those are diagnostic interventions, not rendering fixes: the constant
produces an almost white image, and the direct sample is a distorted gradient. The earlier saved-module override
changed over a hundred draws and its black result was not an isolated LUT experiment.

A fresh same-binary v58 ordered capture of the missing producer records **4 vertices × 32 instances**. The
old capture omitted failed-draw instance count and now reports it as unknown when opened by current replay.
The producer's terminal PARAM output is transformed after shared-memory reads, so prosper's proven no-GS
per-vertex shortcut rejects it. A mesh/workgroup execution path must preserve its cross-lane and LDS behavior;
lifting the vertex `v_mbcnt` gate would not do that. The ordered captures remain unverified against their own
presented frame. The live same-draw bracket establishes the bound zero LUT and black output independently.

The compile-only four-wave NGG probe now offers a raw VGPR value plus execution-hit trace at a selected linked
instruction. On a **speculative** one-subgroup input for this producer, it exposed a bounded DPP reduction bug
in the probe's native Wave64 dispatcher: a `BOUND_CTRL` marker was passed as part of the lane shift, so the
reduction never collected neighboring lanes. With that corrected, the same input changes from zero to 32
nonzero raw PRIM records. A byte-matched two-instruction synthetic test guards the reduction and wave isolation.
The referenced vertex slots clarify those records: their packed indices cover 96 distinct slots, with
nondegenerate clip-space triangles and two candidate triangles on each of layers 0–15. Supplying instance
indices 16–31 to a second otherwise-identical offline subgroup produces the corresponding layers 16–31;
changing only the proposed primitive IDs leaves the readback byte-identical. The earlier count of just six
POS1.z values examined only lanes carrying PRIM records and missed the vertex records they reference.
These are still **supplied** launch inputs, not validated hardware subgroups, primitives, pixels, or a title
fix. The captured blobs are callback-time copies and the live renderer still rejects the producer. #3135
remains open.

The LUT fragment program has a separate missing input. A live exact-program trace records
`SPI_PS_INPUT_ENA=SPI_PS_INPUT_ADDR=0x2020`, enabling fields 5 and 13. Its first vector instruction,
`v_bfe_u32 v3, v2, 16, 11`, extracts the array index from the ancillary VGPR (v2 after field 5's
two VGPRs). The fragment recompiler previously reserved that register but left it at zero, and a
private execution of the exact live fragment module with supplied full-screen geometry produced
32 byte-identical LUT slices. A controlled fragment `Layer` input and view-base specialization
produced 32 distinct slices in that fixture; neither result establishes the correct guest LUT.

The four eight-slice batches were **fixture-only**, not production behavior. Live Kena passes one
32-slice view to the backend, which refuses mesh draws exceeding this device's
`maxMeshOutputLayers`; live draw construction does not yet wire the guest NGG chain into MeshEXT.
Partitioning will need a proved mapping for guest workgroups, primitive layer output and side
effects. The guest view's starting slice and an internal host batch offset must remain distinct;
the fixture's base values cannot be copied into production without that contract.

A targeted live state log for that same producer records `SPI_SHADER_PGM_RSRC2_GS=0x008b0000`
(2,176 LDS dwords, 8.5 KiB), `VGT_GS_ONCHIP_CNTL=0x10020040`
(64 ES vertices, 64 GS primitives and 64 instanced primitives per subgroup using the
[gfx10 register layout](https://chromium.googlesource.com/chromiumos/third_party/mesa/+/refs/heads/stabilize-13982.70.B-chromeos-amd/src/amd/registers/gfx10.json)),
`GE_NGG_SUBGRP_CNTL=1`, `VGT_GS_MAX_VERT_OUT=3`,
`CB_COLOR0_VIEW=0x00040000`, and `CB_COLOR0_ATTRIB3=0x4606c01f`. The latter's
`MIP0_DEPTH=31` encodes 32 slices and its resource-type field is 2; the immediately following
consumer binds the same address as a 32³ texture. PR #3842 added a generic retained 3D color-target
path; the shader and draw launch still need to produce valid layered output for it. The slice-max
register is recorded raw rather than interpreted as a layer count until its boundary convention is
established. On this AMD host, a standalone mesh draw rendered 32 layers in four eight-layer batches
with Khronos validation loaded and no reported errors. This proves only that the host API route exists:
the guest wave/primitive mapping and correct LUT values still need implementation and game verification.

An exact-program-and-output-shape live witness on the 32×32, 32-instance LUT draw reported the same
state eight times: `SPI_SHADER_PGM_RSRC1_GS=0x622c0047`, `RSRC2_GS=0x008b0000`,
`VGT_GS_ONCHIP_CNTL=0x10020040`, `VGT_ESGS_RING_ITEMSIZE=4`, and `GE_NGG_SUBGRP_CNTL=1`.
The `GS_VGPR_COMP_CNT` and `ES_VGPR_COMP_CNT` fields are both 3, so the hardware loads v0–v3 and
v5–v8. `VGT_SHADER_STAGES_EN=0x2030` has `GS_W32_EN=0`, selecting Wave64. The separately guessed
packed GS offsets use units of four, matching this programmed ring item size; this does not validate
the exact lane allocation. An independent review found the first logger wrongly read `GE_CNTL` and
`VGT_PRIMITIVE_TYPE` from the context file: both are **UCONFIG** registers. A corrected live run
records `GE_CNTL=0x00008040` (both programmed group-size fields are 64),
`VGT_PRIMITIVE_TYPE=6` (the capture reports Vulkan triangle strip),
`GE_MAX_OUTPUT_PER_SUBGROUP=0x000000c0` (192), `VGT_GS_MAX_VERT_OUT=3`, and raw
`VGT_GS_OUT_PRIM_TYPE=2`, all present. These are programmed bounds and types, not proof of the
actual subgroup split. The original program-only diagnostic filled its eight-line cap on other
draws using the same shader, which is why the extent/instance filter is needed.

An offline raster check follows the 9-bit indices of every PRIM record in the two **supplied**
16-instance groups. It finds two opposite-winding, full-screen triangles on each candidate layer
0–31, with no uncovered 32×32 pixel sample centers and PARAM.xy matching the normalized screen
coordinates at every referenced vertex. The captured failed draw has `raster=0/0/0` (no culling,
CCW front, fill), so the opposite windings do not by themselves discard half the lookup. A
wrong packed-offset input fails this checker with mixed-layer primitives. This is candidate
geometry only: the actual hardware launch, fragment output, and live title pixels are still
unverified. The raster-field print is included for failed draws in `gpu_replay --inspect-only`,
which previously printed it only for realized draws (#3135).

The input-default probe logged changes to earlier fragment programs `0x3007c60000` and `0x3007c80000`
paired with vertex program `0x3007060000`; it did **not** change the final compositor's input wiring.
A separate pair trace of `0x3007060000` with fragment `0x30096e0000` found five guest-programmed
`SPI_PS_INPUT_CNTL_0..4=0` values and header-derived constant defaults. The guest writes these registers
through indirect **physical** offsets (17,420 observed writes, none through the virtual AGC bank).
That vertex program exports PRIM/POS0 without PARAM exports; it is a fused front shader, with no hidden
linked back/chain body. This traced pair is also **not** the final compositor. The captured compositor
vertex SPIR-V stores Locations 0–3. Trace the exact earlier pass changed by the probe and establish its
PS5 input contract before changing generic interpolation semantics.

## Reproduction

```bash
PROSPER_NULL_PAGE=1 PROSPER_GUEST_ARGS= \
PROSPER_SAVE0=<FRESH>/save0 PROSPER_SAVEDATA_DIR=<FRESH>/savedata \
PROSPER_PAD_SCRIPT=@scripts/kena/reach-first-gameplay-prompt.pad PROSPER_PAD_SCRIPT_LOG=1 \
  ./build-mingw-app/screenshot.exe <DUMP_ROOT>/PPSA01802-app0 --seconds 10 --count 54 --timeout 560 --out <OUT>
```

`PROSPER_NULL_PAGE=1` and an empty `PROSPER_GUEST_ARGS` are the UE4 recipe (`DRAGON_QUEST_STATUS.md`). The route's
anchors are pad reads — see `prosper/scripts/kena/README.md`. `tools/dump_hygiene.py` on the dump is clean; its
`_original_files/` (the untouched encrypted originals) is retained and never read.

About 2 in 12 launches hang before their first frame (no raw scanout either). Not yet investigated; relaunch.

## What was fixed to get here

1. **`sceAvPlayerStartEx` was unregistered** ([#3781](https://github.com/mattias800/prosper/issues/3781), merged
   as [#3788](https://github.com/mattias800/prosper/pull/3788)). The title adds its logo movie with
   `auto_start = 0` and starts it only through this call; the dispatcher's `return 0` answered `SCE_OK` without
   starting anything, and the title waited forever on a black screen. 0 of 2 → 2 of 2 runs reach the menu.
2. **A fixed `sceKernelBatchMap` MAP_DIRECT failed over pages prosper had lazily committed**
   ([#3812](https://github.com/mattias800/prosper/issues/3812), PR [#3814](https://github.com/mattias800/prosper/pull/3814),
   Windows only). Host-side reads first-touched unmapped pages of the guest's own reservation; the VEH replaced
   those placeholder pieces with private memory, and `MapViewOfFile3` cannot place a view over them. The guest got
   ENOMEM and UE4 aborted (`sceKernelBatchMap failed with error code: 0x8002000c`) 50-65 s in, 5 of 6 runs. A fixed
   map now releases the range's guest-owned contents first, as `mmap(MAP_FIXED)` does on Linux.
3. **The recompiler emitted invalid SPIR-V for a fragment `s_bfm_b64`**
   ([#3816](https://github.com/mattias800/prosper/issues/3816), PR [#3817](https://github.com/mattias800/prosper/pull/3817)).
   Its lane test used `linear_localid`, which is id 0 in a fragment module; NVIDIA's driver dereferenced it while
   compiling the pipeline and the process died inside `nvoglv64.dll` on the first level load, 4 of 4 runs.

## Ruled out

- **The black pre-menu screen is a renderer failure** — false. The composite was black because the logo movie was
  never started (unregistered `sceAvPlayerStartEx`, #3781); registering it reaches the menu with no renderer change.
- **The BatchMap ENOMEM is #2424's too-small free placeholder** — false. `VirtualQuery` on the failing range shows
  `MEM_COMMIT | MEM_PRIVATE` pages in 16 KiB pieces with non-64 KiB allocation bases, and the registry holds
  **0** free placeholders at that moment; it is prosper's own lazy commit, not a placeholder size mismatch (#3812).
- **The guest mapped or touched the failing range before the fatal map** — false. `PROSPER_MEMLOG=1` shows no guest
  map, flexible map or unmap of that range before the fatal MAP_DIRECT, and a RIP-tagged log of every VEH lazy
  commit (5,120+ in 145 s; the first 60 and every 1024th logged, 65 lines) found **all 65** in a host
  system DLL's copy routine and none in guest code (#3812).
- **The level-load crash is an NVIDIA driver bug** — false as the root cause. Of all 11,710 graphics SPIR-V
  modules created in a crashing run, `spirv-val` rejects exactly one — the last one created — and its only
  defect is an operand naming id 0 emitted by prosper's `s_bfm_b64` lowering (#3816). The driver crashing on
  invalid SPIR-V rather than rejecting it is a driver robustness gap, not the cause.
- **Seconds-anchored pad pulses are enough for this title's menus** — false. An 11-press seconds-based route
  delivered 2 presses, because the guest reads the pad about once a second there and a 300 ms pulse mostly falls
  between reads; pad-read anchors (`pN`) deliver every press.
- **Any nonzero 3D LUT makes the title oracle-correct** — false. An exact one-draw constant-LUT control changes
  the live compositor and menu endpoint, but the result is nearly white. Directly outputting the already bound
  scene sample instead gives a distorted gradient. The LUT producer is a real blocked dependency; the scene
  feeding it is also spatially wrong, so both paths need separate repairs (#3135, #3835).
- **The traced earlier fragment inputs are zero because virtual AGC register translation dropped their writes**
  — false for the traced `0x30096e0000` pair. A path-labelled register watch observed 17,420 direct physical
  indirect-array writes to the controls and zero virtual-bank writes (#3835).
- **The traced earlier vertex program's missing PARAM stores are in an unlinked back half** — false for its
  `0x30096e0000` pair. Exact-pair logging names one fused front program, no chain, and no linked second body;
  its decoded entry ends without a PARAM export. This says nothing about the separate final compositor pair (#3835).

## Next

- Confirm that the black gameplay world (and the menu background) is the wave64 skip, on a 64-wide device or with
  a per-draw skip census of a gameplay frame; if it is, the title waits on #2147.
  A Linux lane has since traced the menu scene ([#3813](https://github.com/mattias800/prosper/pull/3813)): the
  forest is present in a retained MRT3 image that a one-block fragment shader samples at binding 37 — so on Linux
  the background scene is rendered somewhere. The Linux title-menu boundary above now narrows the later loss;
  do not assume the Windows black background is solely the wave64 skip. Resolve the generic missing-PARAM
  contract and merged-NGG LUT producer separately, then compare a live default-path image with the oracle.
- Host-side renderer reads of never-mapped guest memory (#3820), and `SCE_KERNEL_MAP_NO_OVERWRITE` (#3819).
- The invisible logo movie: 150 frames decoded and delivered, none visible; delivered at ~2x real time.
- The early-boot hang (about 2 in 12 launches).
- Derive the actual ES/GS subgroup allocation and every per-wave `s3`/v0..v8 value for the 4×32 draw.
  The current source-bounded **candidate** uses two 64-ES/32-GS subgroups and produces two offline triangles
  on each of 32 candidate layers. An output pattern that fits the target is not a hardware launch oracle;
  require a wrong-partition control and independent ABI evidence before live shader admission (#3135, #2072).
- Lower the proved output into MeshEXT with the existing retained 3D target path, preserving the 96 indexed
  vertex slots and 32 primitive records **per candidate subgroup**, then compare the live default-path
  title frame with the oracle. Devices without the optional subgroup/mesh features must continue to decline
  safely. The separate scene/interpolation defect (#3835) may still affect visual correctness.
